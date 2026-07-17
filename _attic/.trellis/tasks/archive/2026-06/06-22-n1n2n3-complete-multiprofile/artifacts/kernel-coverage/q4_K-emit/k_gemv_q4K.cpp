#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 2304;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 2304;
      const uint8_t* v19 = v12 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 292;
      const uint8_t* v21 = v4 + v20;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const float* v22 = (const float*) v21;
      float v23 = *(const float *)(v22);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v24 = v19 + 32;
      const _Float16* v25 = (const _Float16*) v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v26 = __riscv_vle16_v_f16m1(v25, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v27 = __riscv_vfwcvt_f_f_v_f32m2(v26, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v28 = __riscv_vfmul_vf_f32m2(v27, v23, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v29 = v19 + 48;
      const _Float16* v30 = (const _Float16*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v31 = __riscv_vle16_v_f16m1(v30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v32 = __riscv_vfwcvt_f_f_v_f32m2(v31, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v33 = __riscv_vfmul_vf_f32m2(v32, v23, 8);
      vint32m2_t v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v35 = __riscv_vmv_v_x_i32m2(0, 8);
      v34 = v35;
      vint32m2_t v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v37 = __riscv_vmv_v_x_i32m2(0, 8);
      v36 = v37;
      vint32m2_t v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v39 = __riscv_vmv_v_x_i32m2(0, 8);
      v38 = v39;
      vint32m2_t v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v41 = __riscv_vmv_v_x_i32m2(0, 8);
      v40 = v41;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v42 = v19 + 64;
      const uint8_t* v43 = (const uint8_t*) v42;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v44 = __riscv_vle8_v_u8mf2(v43, 8);
      const uint8_t* v45 = v19 + 192;
      const uint8_t* v46 = (const uint8_t*) v45;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v47 = __riscv_vle8_v_u8mf2(v46, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v48 = __riscv_vand_vx_u8mf2(v44, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v49 = __riscv_vsrl_vx_u8mf2(v44, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v50 = __riscv_vand_vx_u8mf2(v47, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v51 = __riscv_vsll_vx_u8mf2(v50, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v52 = __riscv_vand_vx_u8mf2(v47, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v53 = __riscv_vsll_vx_u8mf2(v52, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v54 = __riscv_vor_vv_u8mf2(v51, v48, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v55 = __riscv_vor_vv_u8mf2(v53, v49, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v56 = __riscv_vzext_vf2_u16m1(v54, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v57 = __riscv_vreinterpret_v_u16m1_i16m1(v56);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v58 = __riscv_vzext_vf2_u16m1(v55, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v59 = __riscv_vreinterpret_v_u16m1_i16m1(v58);
      const uint8_t* v60 = v19 + 80;
      const uint8_t* v61 = (const uint8_t*) v60;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v62 = __riscv_vle8_v_u8mf2(v61, 8);
      const uint8_t* v63 = v19 + 208;
      const uint8_t* v64 = (const uint8_t*) v63;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v65 = __riscv_vle8_v_u8mf2(v64, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v66 = __riscv_vand_vx_u8mf2(v62, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v67 = __riscv_vsrl_vx_u8mf2(v62, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v68 = __riscv_vand_vx_u8mf2(v65, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v69 = __riscv_vsll_vx_u8mf2(v68, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v70 = __riscv_vand_vx_u8mf2(v65, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v71 = __riscv_vsll_vx_u8mf2(v70, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v72 = __riscv_vor_vv_u8mf2(v69, v66, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v73 = __riscv_vor_vv_u8mf2(v71, v67, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v74 = __riscv_vzext_vf2_u16m1(v72, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v75 = __riscv_vreinterpret_v_u16m1_i16m1(v74);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v76 = __riscv_vzext_vf2_u16m1(v73, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v77 = __riscv_vreinterpret_v_u16m1_i16m1(v76);
      const uint8_t* v78 = v19 + 96;
      const uint8_t* v79 = (const uint8_t*) v78;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v80 = __riscv_vle8_v_u8mf2(v79, 8);
      const uint8_t* v81 = v19 + 224;
      const uint8_t* v82 = (const uint8_t*) v81;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v83 = __riscv_vle8_v_u8mf2(v82, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v84 = __riscv_vand_vx_u8mf2(v80, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v85 = __riscv_vsrl_vx_u8mf2(v80, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v86 = __riscv_vand_vx_u8mf2(v83, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v87 = __riscv_vsll_vx_u8mf2(v86, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v88 = __riscv_vand_vx_u8mf2(v83, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v89 = __riscv_vsll_vx_u8mf2(v88, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v90 = __riscv_vor_vv_u8mf2(v87, v84, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v91 = __riscv_vor_vv_u8mf2(v89, v85, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v92 = __riscv_vzext_vf2_u16m1(v90, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v93 = __riscv_vreinterpret_v_u16m1_i16m1(v92);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v94 = __riscv_vzext_vf2_u16m1(v91, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v95 = __riscv_vreinterpret_v_u16m1_i16m1(v94);
      const uint8_t* v96 = v19 + 112;
      const uint8_t* v97 = (const uint8_t*) v96;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v98 = __riscv_vle8_v_u8mf2(v97, 8);
      const uint8_t* v99 = v19 + 240;
      const uint8_t* v100 = (const uint8_t*) v99;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v101 = __riscv_vle8_v_u8mf2(v100, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v102 = __riscv_vand_vx_u8mf2(v98, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v103 = __riscv_vsrl_vx_u8mf2(v98, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v104 = __riscv_vand_vx_u8mf2(v101, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v105 = __riscv_vsll_vx_u8mf2(v104, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v106 = __riscv_vand_vx_u8mf2(v101, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v107 = __riscv_vsll_vx_u8mf2(v106, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v108 = __riscv_vor_vv_u8mf2(v105, v102, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v109 = __riscv_vor_vv_u8mf2(v107, v103, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v110 = __riscv_vzext_vf2_u16m1(v108, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v111 = __riscv_vreinterpret_v_u16m1_i16m1(v110);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v112 = __riscv_vzext_vf2_u16m1(v109, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v113 = __riscv_vreinterpret_v_u16m1_i16m1(v112);
      const uint8_t* v114 = v19 + 72;
      const uint8_t* v115 = (const uint8_t*) v114;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v116 = __riscv_vle8_v_u8mf2(v115, 8);
      const uint8_t* v117 = v19 + 200;
      const uint8_t* v118 = (const uint8_t*) v117;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v119 = __riscv_vle8_v_u8mf2(v118, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v120 = __riscv_vand_vx_u8mf2(v116, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v121 = __riscv_vsrl_vx_u8mf2(v116, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v122 = __riscv_vand_vx_u8mf2(v119, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v123 = __riscv_vsll_vx_u8mf2(v122, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v124 = __riscv_vand_vx_u8mf2(v119, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v125 = __riscv_vsll_vx_u8mf2(v124, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v126 = __riscv_vor_vv_u8mf2(v123, v120, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v127 = __riscv_vor_vv_u8mf2(v125, v121, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v128 = __riscv_vzext_vf2_u16m1(v126, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v129 = __riscv_vreinterpret_v_u16m1_i16m1(v128);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v130 = __riscv_vzext_vf2_u16m1(v127, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v131 = __riscv_vreinterpret_v_u16m1_i16m1(v130);
      const uint8_t* v132 = v19 + 88;
      const uint8_t* v133 = (const uint8_t*) v132;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v134 = __riscv_vle8_v_u8mf2(v133, 8);
      const uint8_t* v135 = v19 + 216;
      const uint8_t* v136 = (const uint8_t*) v135;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v137 = __riscv_vle8_v_u8mf2(v136, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v138 = __riscv_vand_vx_u8mf2(v134, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v139 = __riscv_vsrl_vx_u8mf2(v134, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v140 = __riscv_vand_vx_u8mf2(v137, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v141 = __riscv_vsll_vx_u8mf2(v140, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v142 = __riscv_vand_vx_u8mf2(v137, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v143 = __riscv_vsll_vx_u8mf2(v142, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v144 = __riscv_vor_vv_u8mf2(v141, v138, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v145 = __riscv_vor_vv_u8mf2(v143, v139, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v146 = __riscv_vzext_vf2_u16m1(v144, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v147 = __riscv_vreinterpret_v_u16m1_i16m1(v146);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v148 = __riscv_vzext_vf2_u16m1(v145, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v149 = __riscv_vreinterpret_v_u16m1_i16m1(v148);
      const uint8_t* v150 = v19 + 104;
      const uint8_t* v151 = (const uint8_t*) v150;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v152 = __riscv_vle8_v_u8mf2(v151, 8);
      const uint8_t* v153 = v19 + 232;
      const uint8_t* v154 = (const uint8_t*) v153;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v155 = __riscv_vle8_v_u8mf2(v154, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v156 = __riscv_vand_vx_u8mf2(v152, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v157 = __riscv_vsrl_vx_u8mf2(v152, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v158 = __riscv_vand_vx_u8mf2(v155, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v159 = __riscv_vsll_vx_u8mf2(v158, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v160 = __riscv_vand_vx_u8mf2(v155, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v161 = __riscv_vsll_vx_u8mf2(v160, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v162 = __riscv_vor_vv_u8mf2(v159, v156, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v163 = __riscv_vor_vv_u8mf2(v161, v157, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v164 = __riscv_vzext_vf2_u16m1(v162, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v165 = __riscv_vreinterpret_v_u16m1_i16m1(v164);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v166 = __riscv_vzext_vf2_u16m1(v163, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v167 = __riscv_vreinterpret_v_u16m1_i16m1(v166);
      const uint8_t* v168 = v19 + 120;
      const uint8_t* v169 = (const uint8_t*) v168;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v170 = __riscv_vle8_v_u8mf2(v169, 8);
      const uint8_t* v171 = v19 + 248;
      const uint8_t* v172 = (const uint8_t*) v171;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v173 = __riscv_vle8_v_u8mf2(v172, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v174 = __riscv_vand_vx_u8mf2(v170, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v175 = __riscv_vsrl_vx_u8mf2(v170, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v176 = __riscv_vand_vx_u8mf2(v173, 0x03, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v177 = __riscv_vsll_vx_u8mf2(v176, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v178 = __riscv_vand_vx_u8mf2(v173, 0x0C, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v179 = __riscv_vsll_vx_u8mf2(v178, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v180 = __riscv_vor_vv_u8mf2(v177, v174, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v181 = __riscv_vor_vv_u8mf2(v179, v175, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v182 = __riscv_vzext_vf2_u16m1(v180, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v183 = __riscv_vreinterpret_v_u16m1_i16m1(v182);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v184 = __riscv_vzext_vf2_u16m1(v181, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v185 = __riscv_vreinterpret_v_u16m1_i16m1(v184);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v186 = v21 + 260;
      const int16_t* v187 = (const int16_t*) v186;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v188 = *(const int16_t *)(v187);
      const uint8_t* v189 = v21 + 262;
      const int16_t* v190 = (const int16_t*) v189;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v191 = *(const int16_t *)(v190);
      int32_t v192 = v188 + v191;
      vint32m2_t v193 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v194 = __riscv_vwmacc_vx_i32m2(v193, v192, v59, 8);
      v36 = v194;
      vint32m2_t v195 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v196 = __riscv_vwmacc_vx_i32m2(v195, v192, v131, 8);
      v40 = v196;
      const uint8_t* v197 = v21 + 264;
      const int16_t* v198 = (const int16_t*) v197;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v199 = *(const int16_t *)(v198);
      const uint8_t* v200 = v21 + 266;
      const int16_t* v201 = (const int16_t*) v200;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v202 = *(const int16_t *)(v201);
      int32_t v203 = v199 + v202;
      vint32m2_t v204 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v205 = __riscv_vwmacc_vx_i32m2(v204, v203, v77, 8);
      v36 = v205;
      vint32m2_t v206 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v207 = __riscv_vwmacc_vx_i32m2(v206, v203, v149, 8);
      v40 = v207;
      const uint8_t* v208 = v21 + 268;
      const int16_t* v209 = (const int16_t*) v208;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v210 = *(const int16_t *)(v209);
      const uint8_t* v211 = v21 + 270;
      const int16_t* v212 = (const int16_t*) v211;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v213 = *(const int16_t *)(v212);
      int32_t v214 = v210 + v213;
      vint32m2_t v215 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v216 = __riscv_vwmacc_vx_i32m2(v215, v214, v95, 8);
      v36 = v216;
      vint32m2_t v217 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v218 = __riscv_vwmacc_vx_i32m2(v217, v214, v167, 8);
      v40 = v218;
      const uint8_t* v219 = v21 + 272;
      const int16_t* v220 = (const int16_t*) v219;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v221 = *(const int16_t *)(v220);
      const uint8_t* v222 = v21 + 274;
      const int16_t* v223 = (const int16_t*) v222;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v224 = *(const int16_t *)(v223);
      int32_t v225 = v221 + v224;
      vint32m2_t v226 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v227 = __riscv_vwmacc_vx_i32m2(v226, v225, v113, 8);
      v36 = v227;
      vint32m2_t v228 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v229 = __riscv_vwmacc_vx_i32m2(v228, v225, v185, 8);
      v40 = v229;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v230 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v231 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v232 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v233 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v234 = v19 + 256;
      const uint8_t* v235 = (const uint8_t*) v234;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v236 = __riscv_vle8_v_u8mf2(v235, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v237 = __riscv_vand_vx_u8mf2(v236, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v238 = __riscv_vreinterpret_v_u8mf2_i8mf2(v237);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v239 = __riscv_vsrl_vx_u8mf2(v236, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v240 = __riscv_vreinterpret_v_u8mf2_i8mf2(v239);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v241 = v21 + 4;
      const int8_t* v242 = (const int8_t*) v241;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v243 = *(const int8_t *)(v242);
      const uint8_t* v244 = v21 + 36;
      const int8_t* v245 = (const int8_t*) v244;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v246 = *(const int8_t *)(v245);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v247 = __riscv_vwmacc_vx_i16m1(v230, v243, v238, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v248 = __riscv_vwmacc_vx_i16m1(v231, v246, v240, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v249 = v19 + 264;
      const uint8_t* v250 = (const uint8_t*) v249;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v251 = __riscv_vle8_v_u8mf2(v250, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v252 = __riscv_vand_vx_u8mf2(v251, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v253 = __riscv_vreinterpret_v_u8mf2_i8mf2(v252);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v254 = __riscv_vsrl_vx_u8mf2(v251, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v255 = __riscv_vreinterpret_v_u8mf2_i8mf2(v254);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v256 = v21 + 4;
      const int8_t* v257 = (const int8_t*) v256;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v258 = *(const int8_t *)(v257);
      const uint8_t* v259 = v21 + 36;
      const int8_t* v260 = (const int8_t*) v259;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v261 = *(const int8_t *)(v260);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v262 = __riscv_vwmacc_vx_i16m1(v232, v258, v253, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v263 = __riscv_vwmacc_vx_i16m1(v233, v261, v255, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v264 = v19 + 272;
      const uint8_t* v265 = (const uint8_t*) v264;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v266 = __riscv_vle8_v_u8mf2(v265, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v267 = __riscv_vand_vx_u8mf2(v266, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v268 = __riscv_vreinterpret_v_u8mf2_i8mf2(v267);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v269 = __riscv_vsrl_vx_u8mf2(v266, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v270 = __riscv_vreinterpret_v_u8mf2_i8mf2(v269);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v271 = v21 + 5;
      const int8_t* v272 = (const int8_t*) v271;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v273 = *(const int8_t *)(v272);
      const uint8_t* v274 = v21 + 37;
      const int8_t* v275 = (const int8_t*) v274;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v276 = *(const int8_t *)(v275);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v277 = __riscv_vwmacc_vx_i16m1(v247, v273, v268, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v278 = __riscv_vwmacc_vx_i16m1(v248, v276, v270, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v279 = v19 + 280;
      const uint8_t* v280 = (const uint8_t*) v279;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v281 = __riscv_vle8_v_u8mf2(v280, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v282 = __riscv_vand_vx_u8mf2(v281, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v282);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v284 = __riscv_vsrl_vx_u8mf2(v281, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v285 = __riscv_vreinterpret_v_u8mf2_i8mf2(v284);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v286 = v21 + 5;
      const int8_t* v287 = (const int8_t*) v286;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v288 = *(const int8_t *)(v287);
      const uint8_t* v289 = v21 + 37;
      const int8_t* v290 = (const int8_t*) v289;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v291 = *(const int8_t *)(v290);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v292 = __riscv_vwmacc_vx_i16m1(v262, v288, v283, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v293 = __riscv_vwmacc_vx_i16m1(v263, v291, v285, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v294 = v19 + 288;
      const uint8_t* v295 = (const uint8_t*) v294;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v296 = __riscv_vle8_v_u8mf2(v295, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v297 = __riscv_vand_vx_u8mf2(v296, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v298 = __riscv_vreinterpret_v_u8mf2_i8mf2(v297);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v299 = __riscv_vsrl_vx_u8mf2(v296, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v300 = __riscv_vreinterpret_v_u8mf2_i8mf2(v299);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v301 = v21 + 6;
      const int8_t* v302 = (const int8_t*) v301;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v303 = *(const int8_t *)(v302);
      const uint8_t* v304 = v21 + 38;
      const int8_t* v305 = (const int8_t*) v304;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v306 = *(const int8_t *)(v305);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v307 = __riscv_vwmacc_vx_i16m1(v277, v303, v298, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v308 = __riscv_vwmacc_vx_i16m1(v278, v306, v300, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v309 = v19 + 296;
      const uint8_t* v310 = (const uint8_t*) v309;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v311 = __riscv_vle8_v_u8mf2(v310, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v312 = __riscv_vand_vx_u8mf2(v311, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v312);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v314 = __riscv_vsrl_vx_u8mf2(v311, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v315 = __riscv_vreinterpret_v_u8mf2_i8mf2(v314);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v316 = v21 + 6;
      const int8_t* v317 = (const int8_t*) v316;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v318 = *(const int8_t *)(v317);
      const uint8_t* v319 = v21 + 38;
      const int8_t* v320 = (const int8_t*) v319;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v321 = *(const int8_t *)(v320);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v322 = __riscv_vwmacc_vx_i16m1(v292, v318, v313, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v323 = __riscv_vwmacc_vx_i16m1(v293, v321, v315, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v324 = v19 + 304;
      const uint8_t* v325 = (const uint8_t*) v324;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v326 = __riscv_vle8_v_u8mf2(v325, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v327 = __riscv_vand_vx_u8mf2(v326, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v328 = __riscv_vreinterpret_v_u8mf2_i8mf2(v327);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v329 = __riscv_vsrl_vx_u8mf2(v326, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v330 = __riscv_vreinterpret_v_u8mf2_i8mf2(v329);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v331 = v21 + 7;
      const int8_t* v332 = (const int8_t*) v331;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v333 = *(const int8_t *)(v332);
      const uint8_t* v334 = v21 + 39;
      const int8_t* v335 = (const int8_t*) v334;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v336 = *(const int8_t *)(v335);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v337 = __riscv_vwmacc_vx_i16m1(v307, v333, v328, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v338 = __riscv_vwmacc_vx_i16m1(v308, v336, v330, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v339 = v19 + 312;
      const uint8_t* v340 = (const uint8_t*) v339;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v341 = __riscv_vle8_v_u8mf2(v340, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v342 = __riscv_vand_vx_u8mf2(v341, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v343 = __riscv_vreinterpret_v_u8mf2_i8mf2(v342);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v344 = __riscv_vsrl_vx_u8mf2(v341, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v345 = __riscv_vreinterpret_v_u8mf2_i8mf2(v344);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v346 = v21 + 7;
      const int8_t* v347 = (const int8_t*) v346;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v348 = *(const int8_t *)(v347);
      const uint8_t* v349 = v21 + 39;
      const int8_t* v350 = (const int8_t*) v349;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v351 = *(const int8_t *)(v350);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v352 = __riscv_vwmacc_vx_i16m1(v322, v348, v343, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v353 = __riscv_vwmacc_vx_i16m1(v323, v351, v345, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v354 = v19 + 320;
      const uint8_t* v355 = (const uint8_t*) v354;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v356 = __riscv_vle8_v_u8mf2(v355, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v357 = __riscv_vand_vx_u8mf2(v356, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v358 = __riscv_vreinterpret_v_u8mf2_i8mf2(v357);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v359 = __riscv_vsrl_vx_u8mf2(v356, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v360 = __riscv_vreinterpret_v_u8mf2_i8mf2(v359);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v361 = v21 + 8;
      const int8_t* v362 = (const int8_t*) v361;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v363 = *(const int8_t *)(v362);
      const uint8_t* v364 = v21 + 40;
      const int8_t* v365 = (const int8_t*) v364;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v366 = *(const int8_t *)(v365);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v367 = __riscv_vwmacc_vx_i16m1(v337, v363, v358, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v368 = __riscv_vwmacc_vx_i16m1(v338, v366, v360, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v369 = v19 + 328;
      const uint8_t* v370 = (const uint8_t*) v369;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v371 = __riscv_vle8_v_u8mf2(v370, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v372 = __riscv_vand_vx_u8mf2(v371, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v373 = __riscv_vreinterpret_v_u8mf2_i8mf2(v372);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v374 = __riscv_vsrl_vx_u8mf2(v371, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v375 = __riscv_vreinterpret_v_u8mf2_i8mf2(v374);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v376 = v21 + 8;
      const int8_t* v377 = (const int8_t*) v376;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v378 = *(const int8_t *)(v377);
      const uint8_t* v379 = v21 + 40;
      const int8_t* v380 = (const int8_t*) v379;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v381 = *(const int8_t *)(v380);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v382 = __riscv_vwmacc_vx_i16m1(v352, v378, v373, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v383 = __riscv_vwmacc_vx_i16m1(v353, v381, v375, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v384 = v19 + 336;
      const uint8_t* v385 = (const uint8_t*) v384;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v386 = __riscv_vle8_v_u8mf2(v385, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v387 = __riscv_vand_vx_u8mf2(v386, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v387);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v389 = __riscv_vsrl_vx_u8mf2(v386, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v390 = __riscv_vreinterpret_v_u8mf2_i8mf2(v389);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v391 = v21 + 9;
      const int8_t* v392 = (const int8_t*) v391;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v393 = *(const int8_t *)(v392);
      const uint8_t* v394 = v21 + 41;
      const int8_t* v395 = (const int8_t*) v394;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v396 = *(const int8_t *)(v395);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v397 = __riscv_vwmacc_vx_i16m1(v367, v393, v388, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v398 = __riscv_vwmacc_vx_i16m1(v368, v396, v390, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v399 = v19 + 344;
      const uint8_t* v400 = (const uint8_t*) v399;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v401 = __riscv_vle8_v_u8mf2(v400, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v402 = __riscv_vand_vx_u8mf2(v401, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v403 = __riscv_vreinterpret_v_u8mf2_i8mf2(v402);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v404 = __riscv_vsrl_vx_u8mf2(v401, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v405 = __riscv_vreinterpret_v_u8mf2_i8mf2(v404);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v406 = v21 + 9;
      const int8_t* v407 = (const int8_t*) v406;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v408 = *(const int8_t *)(v407);
      const uint8_t* v409 = v21 + 41;
      const int8_t* v410 = (const int8_t*) v409;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v411 = *(const int8_t *)(v410);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v412 = __riscv_vwmacc_vx_i16m1(v382, v408, v403, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v413 = __riscv_vwmacc_vx_i16m1(v383, v411, v405, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v414 = v19 + 352;
      const uint8_t* v415 = (const uint8_t*) v414;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v416 = __riscv_vle8_v_u8mf2(v415, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v417 = __riscv_vand_vx_u8mf2(v416, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v417);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v419 = __riscv_vsrl_vx_u8mf2(v416, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v419);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v421 = v21 + 10;
      const int8_t* v422 = (const int8_t*) v421;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v423 = *(const int8_t *)(v422);
      const uint8_t* v424 = v21 + 42;
      const int8_t* v425 = (const int8_t*) v424;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v426 = *(const int8_t *)(v425);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v427 = __riscv_vwmacc_vx_i16m1(v397, v423, v418, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v428 = __riscv_vwmacc_vx_i16m1(v398, v426, v420, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v429 = v19 + 360;
      const uint8_t* v430 = (const uint8_t*) v429;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v431 = __riscv_vle8_v_u8mf2(v430, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v432 = __riscv_vand_vx_u8mf2(v431, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v433 = __riscv_vreinterpret_v_u8mf2_i8mf2(v432);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v434 = __riscv_vsrl_vx_u8mf2(v431, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v435 = __riscv_vreinterpret_v_u8mf2_i8mf2(v434);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v436 = v21 + 10;
      const int8_t* v437 = (const int8_t*) v436;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v438 = *(const int8_t *)(v437);
      const uint8_t* v439 = v21 + 42;
      const int8_t* v440 = (const int8_t*) v439;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v441 = *(const int8_t *)(v440);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v442 = __riscv_vwmacc_vx_i16m1(v412, v438, v433, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v443 = __riscv_vwmacc_vx_i16m1(v413, v441, v435, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v444 = v19 + 368;
      const uint8_t* v445 = (const uint8_t*) v444;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v446 = __riscv_vle8_v_u8mf2(v445, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v447 = __riscv_vand_vx_u8mf2(v446, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v448 = __riscv_vreinterpret_v_u8mf2_i8mf2(v447);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v449 = __riscv_vsrl_vx_u8mf2(v446, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v450 = __riscv_vreinterpret_v_u8mf2_i8mf2(v449);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v451 = v21 + 11;
      const int8_t* v452 = (const int8_t*) v451;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v453 = *(const int8_t *)(v452);
      const uint8_t* v454 = v21 + 43;
      const int8_t* v455 = (const int8_t*) v454;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v456 = *(const int8_t *)(v455);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v457 = __riscv_vwmacc_vx_i16m1(v427, v453, v448, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v458 = __riscv_vwmacc_vx_i16m1(v428, v456, v450, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v459 = v19 + 376;
      const uint8_t* v460 = (const uint8_t*) v459;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v461 = __riscv_vle8_v_u8mf2(v460, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v462 = __riscv_vand_vx_u8mf2(v461, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v463 = __riscv_vreinterpret_v_u8mf2_i8mf2(v462);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v464 = __riscv_vsrl_vx_u8mf2(v461, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v465 = __riscv_vreinterpret_v_u8mf2_i8mf2(v464);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v466 = v21 + 11;
      const int8_t* v467 = (const int8_t*) v466;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v468 = *(const int8_t *)(v467);
      const uint8_t* v469 = v21 + 43;
      const int8_t* v470 = (const int8_t*) v469;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v471 = *(const int8_t *)(v470);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v472 = __riscv_vwmacc_vx_i16m1(v442, v468, v463, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v473 = __riscv_vwmacc_vx_i16m1(v443, v471, v465, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v474 = v19 + 384;
      const uint8_t* v475 = (const uint8_t*) v474;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v476 = __riscv_vle8_v_u8mf2(v475, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v477 = __riscv_vand_vx_u8mf2(v476, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v478 = __riscv_vreinterpret_v_u8mf2_i8mf2(v477);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v479 = __riscv_vsrl_vx_u8mf2(v476, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v480 = __riscv_vreinterpret_v_u8mf2_i8mf2(v479);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v481 = v21 + 12;
      const int8_t* v482 = (const int8_t*) v481;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v483 = *(const int8_t *)(v482);
      const uint8_t* v484 = v21 + 44;
      const int8_t* v485 = (const int8_t*) v484;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v486 = *(const int8_t *)(v485);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v487 = __riscv_vwmacc_vx_i16m1(v457, v483, v478, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v488 = __riscv_vwmacc_vx_i16m1(v458, v486, v480, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v489 = v19 + 392;
      const uint8_t* v490 = (const uint8_t*) v489;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v491 = __riscv_vle8_v_u8mf2(v490, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v492 = __riscv_vand_vx_u8mf2(v491, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v493 = __riscv_vreinterpret_v_u8mf2_i8mf2(v492);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v494 = __riscv_vsrl_vx_u8mf2(v491, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v495 = __riscv_vreinterpret_v_u8mf2_i8mf2(v494);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v496 = v21 + 12;
      const int8_t* v497 = (const int8_t*) v496;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v498 = *(const int8_t *)(v497);
      const uint8_t* v499 = v21 + 44;
      const int8_t* v500 = (const int8_t*) v499;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v501 = *(const int8_t *)(v500);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v502 = __riscv_vwmacc_vx_i16m1(v472, v498, v493, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v503 = __riscv_vwmacc_vx_i16m1(v473, v501, v495, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v504 = v19 + 400;
      const uint8_t* v505 = (const uint8_t*) v504;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v506 = __riscv_vle8_v_u8mf2(v505, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v507 = __riscv_vand_vx_u8mf2(v506, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v508 = __riscv_vreinterpret_v_u8mf2_i8mf2(v507);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v509 = __riscv_vsrl_vx_u8mf2(v506, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v509);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v511 = v21 + 13;
      const int8_t* v512 = (const int8_t*) v511;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v513 = *(const int8_t *)(v512);
      const uint8_t* v514 = v21 + 45;
      const int8_t* v515 = (const int8_t*) v514;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v516 = *(const int8_t *)(v515);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v517 = __riscv_vwmacc_vx_i16m1(v487, v513, v508, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v518 = __riscv_vwmacc_vx_i16m1(v488, v516, v510, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v519 = v19 + 408;
      const uint8_t* v520 = (const uint8_t*) v519;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v521 = __riscv_vle8_v_u8mf2(v520, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v522 = __riscv_vand_vx_u8mf2(v521, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v523 = __riscv_vreinterpret_v_u8mf2_i8mf2(v522);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v524 = __riscv_vsrl_vx_u8mf2(v521, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v525 = __riscv_vreinterpret_v_u8mf2_i8mf2(v524);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v526 = v21 + 13;
      const int8_t* v527 = (const int8_t*) v526;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v528 = *(const int8_t *)(v527);
      const uint8_t* v529 = v21 + 45;
      const int8_t* v530 = (const int8_t*) v529;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v531 = *(const int8_t *)(v530);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v532 = __riscv_vwmacc_vx_i16m1(v502, v528, v523, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v533 = __riscv_vwmacc_vx_i16m1(v503, v531, v525, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v534 = v19 + 416;
      const uint8_t* v535 = (const uint8_t*) v534;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v536 = __riscv_vle8_v_u8mf2(v535, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v537 = __riscv_vand_vx_u8mf2(v536, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v538 = __riscv_vreinterpret_v_u8mf2_i8mf2(v537);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v539 = __riscv_vsrl_vx_u8mf2(v536, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v540 = __riscv_vreinterpret_v_u8mf2_i8mf2(v539);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v541 = v21 + 14;
      const int8_t* v542 = (const int8_t*) v541;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v543 = *(const int8_t *)(v542);
      const uint8_t* v544 = v21 + 46;
      const int8_t* v545 = (const int8_t*) v544;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v546 = *(const int8_t *)(v545);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v547 = __riscv_vwmacc_vx_i16m1(v517, v543, v538, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v548 = __riscv_vwmacc_vx_i16m1(v518, v546, v540, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v549 = v19 + 424;
      const uint8_t* v550 = (const uint8_t*) v549;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v551 = __riscv_vle8_v_u8mf2(v550, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v552 = __riscv_vand_vx_u8mf2(v551, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v553 = __riscv_vreinterpret_v_u8mf2_i8mf2(v552);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v554 = __riscv_vsrl_vx_u8mf2(v551, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v554);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v556 = v21 + 14;
      const int8_t* v557 = (const int8_t*) v556;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v558 = *(const int8_t *)(v557);
      const uint8_t* v559 = v21 + 46;
      const int8_t* v560 = (const int8_t*) v559;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v561 = *(const int8_t *)(v560);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v562 = __riscv_vwmacc_vx_i16m1(v532, v558, v553, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v563 = __riscv_vwmacc_vx_i16m1(v533, v561, v555, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v564 = v19 + 432;
      const uint8_t* v565 = (const uint8_t*) v564;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v566 = __riscv_vle8_v_u8mf2(v565, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v567 = __riscv_vand_vx_u8mf2(v566, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v568 = __riscv_vreinterpret_v_u8mf2_i8mf2(v567);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v569 = __riscv_vsrl_vx_u8mf2(v566, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v570 = __riscv_vreinterpret_v_u8mf2_i8mf2(v569);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v571 = v21 + 15;
      const int8_t* v572 = (const int8_t*) v571;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v573 = *(const int8_t *)(v572);
      const uint8_t* v574 = v21 + 47;
      const int8_t* v575 = (const int8_t*) v574;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v576 = *(const int8_t *)(v575);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v577 = __riscv_vwmacc_vx_i16m1(v547, v573, v568, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v578 = __riscv_vwmacc_vx_i16m1(v548, v576, v570, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v579 = v19 + 440;
      const uint8_t* v580 = (const uint8_t*) v579;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v581 = __riscv_vle8_v_u8mf2(v580, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v582 = __riscv_vand_vx_u8mf2(v581, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v583 = __riscv_vreinterpret_v_u8mf2_i8mf2(v582);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v584 = __riscv_vsrl_vx_u8mf2(v581, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v585 = __riscv_vreinterpret_v_u8mf2_i8mf2(v584);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v586 = v21 + 15;
      const int8_t* v587 = (const int8_t*) v586;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v588 = *(const int8_t *)(v587);
      const uint8_t* v589 = v21 + 47;
      const int8_t* v590 = (const int8_t*) v589;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v591 = *(const int8_t *)(v590);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v592 = __riscv_vwmacc_vx_i16m1(v562, v588, v583, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v593 = __riscv_vwmacc_vx_i16m1(v563, v591, v585, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v594 = v19 + 448;
      const uint8_t* v595 = (const uint8_t*) v594;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v596 = __riscv_vle8_v_u8mf2(v595, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v597 = __riscv_vand_vx_u8mf2(v596, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v598 = __riscv_vreinterpret_v_u8mf2_i8mf2(v597);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v599 = __riscv_vsrl_vx_u8mf2(v596, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v600 = __riscv_vreinterpret_v_u8mf2_i8mf2(v599);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v601 = v21 + 16;
      const int8_t* v602 = (const int8_t*) v601;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v603 = *(const int8_t *)(v602);
      const uint8_t* v604 = v21 + 48;
      const int8_t* v605 = (const int8_t*) v604;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v606 = *(const int8_t *)(v605);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v607 = __riscv_vwmacc_vx_i16m1(v577, v603, v598, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v608 = __riscv_vwmacc_vx_i16m1(v578, v606, v600, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v609 = v19 + 456;
      const uint8_t* v610 = (const uint8_t*) v609;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v611 = __riscv_vle8_v_u8mf2(v610, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v612 = __riscv_vand_vx_u8mf2(v611, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v613 = __riscv_vreinterpret_v_u8mf2_i8mf2(v612);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v614 = __riscv_vsrl_vx_u8mf2(v611, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v615 = __riscv_vreinterpret_v_u8mf2_i8mf2(v614);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v616 = v21 + 16;
      const int8_t* v617 = (const int8_t*) v616;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v618 = *(const int8_t *)(v617);
      const uint8_t* v619 = v21 + 48;
      const int8_t* v620 = (const int8_t*) v619;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v621 = *(const int8_t *)(v620);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v622 = __riscv_vwmacc_vx_i16m1(v592, v618, v613, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v623 = __riscv_vwmacc_vx_i16m1(v593, v621, v615, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v624 = v19 + 464;
      const uint8_t* v625 = (const uint8_t*) v624;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v626 = __riscv_vle8_v_u8mf2(v625, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v627 = __riscv_vand_vx_u8mf2(v626, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v628 = __riscv_vreinterpret_v_u8mf2_i8mf2(v627);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v629 = __riscv_vsrl_vx_u8mf2(v626, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v630 = __riscv_vreinterpret_v_u8mf2_i8mf2(v629);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v631 = v21 + 17;
      const int8_t* v632 = (const int8_t*) v631;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v633 = *(const int8_t *)(v632);
      const uint8_t* v634 = v21 + 49;
      const int8_t* v635 = (const int8_t*) v634;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v636 = *(const int8_t *)(v635);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v637 = __riscv_vwmacc_vx_i16m1(v607, v633, v628, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v638 = __riscv_vwmacc_vx_i16m1(v608, v636, v630, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v639 = v19 + 472;
      const uint8_t* v640 = (const uint8_t*) v639;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v641 = __riscv_vle8_v_u8mf2(v640, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v642 = __riscv_vand_vx_u8mf2(v641, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v643 = __riscv_vreinterpret_v_u8mf2_i8mf2(v642);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v644 = __riscv_vsrl_vx_u8mf2(v641, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v645 = __riscv_vreinterpret_v_u8mf2_i8mf2(v644);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v646 = v21 + 17;
      const int8_t* v647 = (const int8_t*) v646;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v648 = *(const int8_t *)(v647);
      const uint8_t* v649 = v21 + 49;
      const int8_t* v650 = (const int8_t*) v649;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v651 = *(const int8_t *)(v650);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v652 = __riscv_vwmacc_vx_i16m1(v622, v648, v643, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v653 = __riscv_vwmacc_vx_i16m1(v623, v651, v645, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v654 = v19 + 480;
      const uint8_t* v655 = (const uint8_t*) v654;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v656 = __riscv_vle8_v_u8mf2(v655, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v657 = __riscv_vand_vx_u8mf2(v656, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v658 = __riscv_vreinterpret_v_u8mf2_i8mf2(v657);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v659 = __riscv_vsrl_vx_u8mf2(v656, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v660 = __riscv_vreinterpret_v_u8mf2_i8mf2(v659);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v661 = v21 + 18;
      const int8_t* v662 = (const int8_t*) v661;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v663 = *(const int8_t *)(v662);
      const uint8_t* v664 = v21 + 50;
      const int8_t* v665 = (const int8_t*) v664;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v666 = *(const int8_t *)(v665);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v667 = __riscv_vwmacc_vx_i16m1(v637, v663, v658, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v668 = __riscv_vwmacc_vx_i16m1(v638, v666, v660, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v669 = v19 + 488;
      const uint8_t* v670 = (const uint8_t*) v669;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v671 = __riscv_vle8_v_u8mf2(v670, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v672 = __riscv_vand_vx_u8mf2(v671, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v672);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v674 = __riscv_vsrl_vx_u8mf2(v671, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v675 = __riscv_vreinterpret_v_u8mf2_i8mf2(v674);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v676 = v21 + 18;
      const int8_t* v677 = (const int8_t*) v676;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v678 = *(const int8_t *)(v677);
      const uint8_t* v679 = v21 + 50;
      const int8_t* v680 = (const int8_t*) v679;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v681 = *(const int8_t *)(v680);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v682 = __riscv_vwmacc_vx_i16m1(v652, v678, v673, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v683 = __riscv_vwmacc_vx_i16m1(v653, v681, v675, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v684 = v19 + 496;
      const uint8_t* v685 = (const uint8_t*) v684;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v686 = __riscv_vle8_v_u8mf2(v685, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v687 = __riscv_vand_vx_u8mf2(v686, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v688 = __riscv_vreinterpret_v_u8mf2_i8mf2(v687);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v689 = __riscv_vsrl_vx_u8mf2(v686, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v690 = __riscv_vreinterpret_v_u8mf2_i8mf2(v689);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v691 = v21 + 19;
      const int8_t* v692 = (const int8_t*) v691;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v693 = *(const int8_t *)(v692);
      const uint8_t* v694 = v21 + 51;
      const int8_t* v695 = (const int8_t*) v694;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v696 = *(const int8_t *)(v695);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v697 = __riscv_vwmacc_vx_i16m1(v667, v693, v688, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v698 = __riscv_vwmacc_vx_i16m1(v668, v696, v690, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v699 = v19 + 504;
      const uint8_t* v700 = (const uint8_t*) v699;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v701 = __riscv_vle8_v_u8mf2(v700, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v702 = __riscv_vand_vx_u8mf2(v701, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v703 = __riscv_vreinterpret_v_u8mf2_i8mf2(v702);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v704 = __riscv_vsrl_vx_u8mf2(v701, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v705 = __riscv_vreinterpret_v_u8mf2_i8mf2(v704);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v706 = v21 + 19;
      const int8_t* v707 = (const int8_t*) v706;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v708 = *(const int8_t *)(v707);
      const uint8_t* v709 = v21 + 51;
      const int8_t* v710 = (const int8_t*) v709;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v711 = *(const int8_t *)(v710);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v712 = __riscv_vwmacc_vx_i16m1(v682, v708, v703, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v713 = __riscv_vwmacc_vx_i16m1(v683, v711, v705, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v714 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v715 = __riscv_vwmacc_vv_i32m2(v714, v57, v697, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v716 = __riscv_vwmacc_vv_i32m2(v715, v75, v698, 8);
      v34 = v716;
      vint32m2_t v717 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v718 = __riscv_vwmacc_vv_i32m2(v717, v129, v712, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v719 = __riscv_vwmacc_vv_i32m2(v718, v147, v713, 8);
      v38 = v719;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v720 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v721 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v722 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v723 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v724 = v19 + 512;
      const uint8_t* v725 = (const uint8_t*) v724;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v726 = __riscv_vle8_v_u8mf2(v725, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v727 = __riscv_vand_vx_u8mf2(v726, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v728 = __riscv_vreinterpret_v_u8mf2_i8mf2(v727);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v729 = __riscv_vsrl_vx_u8mf2(v726, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v730 = __riscv_vreinterpret_v_u8mf2_i8mf2(v729);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v731 = v21 + 20;
      const int8_t* v732 = (const int8_t*) v731;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v733 = *(const int8_t *)(v732);
      const uint8_t* v734 = v21 + 52;
      const int8_t* v735 = (const int8_t*) v734;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v736 = *(const int8_t *)(v735);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v737 = __riscv_vwmacc_vx_i16m1(v720, v733, v728, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v738 = __riscv_vwmacc_vx_i16m1(v721, v736, v730, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v739 = v19 + 520;
      const uint8_t* v740 = (const uint8_t*) v739;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v741 = __riscv_vle8_v_u8mf2(v740, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v742 = __riscv_vand_vx_u8mf2(v741, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v743 = __riscv_vreinterpret_v_u8mf2_i8mf2(v742);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v744 = __riscv_vsrl_vx_u8mf2(v741, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v745 = __riscv_vreinterpret_v_u8mf2_i8mf2(v744);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v746 = v21 + 20;
      const int8_t* v747 = (const int8_t*) v746;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v748 = *(const int8_t *)(v747);
      const uint8_t* v749 = v21 + 52;
      const int8_t* v750 = (const int8_t*) v749;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v751 = *(const int8_t *)(v750);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v752 = __riscv_vwmacc_vx_i16m1(v722, v748, v743, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v753 = __riscv_vwmacc_vx_i16m1(v723, v751, v745, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v754 = v19 + 528;
      const uint8_t* v755 = (const uint8_t*) v754;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v756 = __riscv_vle8_v_u8mf2(v755, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v757 = __riscv_vand_vx_u8mf2(v756, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v758 = __riscv_vreinterpret_v_u8mf2_i8mf2(v757);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v759 = __riscv_vsrl_vx_u8mf2(v756, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v760 = __riscv_vreinterpret_v_u8mf2_i8mf2(v759);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v761 = v21 + 21;
      const int8_t* v762 = (const int8_t*) v761;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v763 = *(const int8_t *)(v762);
      const uint8_t* v764 = v21 + 53;
      const int8_t* v765 = (const int8_t*) v764;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v766 = *(const int8_t *)(v765);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v767 = __riscv_vwmacc_vx_i16m1(v737, v763, v758, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v768 = __riscv_vwmacc_vx_i16m1(v738, v766, v760, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v769 = v19 + 536;
      const uint8_t* v770 = (const uint8_t*) v769;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v771 = __riscv_vle8_v_u8mf2(v770, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v772 = __riscv_vand_vx_u8mf2(v771, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v773 = __riscv_vreinterpret_v_u8mf2_i8mf2(v772);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v774 = __riscv_vsrl_vx_u8mf2(v771, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v775 = __riscv_vreinterpret_v_u8mf2_i8mf2(v774);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v776 = v21 + 21;
      const int8_t* v777 = (const int8_t*) v776;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v778 = *(const int8_t *)(v777);
      const uint8_t* v779 = v21 + 53;
      const int8_t* v780 = (const int8_t*) v779;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v781 = *(const int8_t *)(v780);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v782 = __riscv_vwmacc_vx_i16m1(v752, v778, v773, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v783 = __riscv_vwmacc_vx_i16m1(v753, v781, v775, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v784 = v19 + 544;
      const uint8_t* v785 = (const uint8_t*) v784;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v786 = __riscv_vle8_v_u8mf2(v785, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v787 = __riscv_vand_vx_u8mf2(v786, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v788 = __riscv_vreinterpret_v_u8mf2_i8mf2(v787);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v789 = __riscv_vsrl_vx_u8mf2(v786, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v790 = __riscv_vreinterpret_v_u8mf2_i8mf2(v789);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v791 = v21 + 22;
      const int8_t* v792 = (const int8_t*) v791;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v793 = *(const int8_t *)(v792);
      const uint8_t* v794 = v21 + 54;
      const int8_t* v795 = (const int8_t*) v794;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v796 = *(const int8_t *)(v795);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v797 = __riscv_vwmacc_vx_i16m1(v767, v793, v788, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v798 = __riscv_vwmacc_vx_i16m1(v768, v796, v790, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v799 = v19 + 552;
      const uint8_t* v800 = (const uint8_t*) v799;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v801 = __riscv_vle8_v_u8mf2(v800, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v802 = __riscv_vand_vx_u8mf2(v801, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v803 = __riscv_vreinterpret_v_u8mf2_i8mf2(v802);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v804 = __riscv_vsrl_vx_u8mf2(v801, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v805 = __riscv_vreinterpret_v_u8mf2_i8mf2(v804);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v806 = v21 + 22;
      const int8_t* v807 = (const int8_t*) v806;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v808 = *(const int8_t *)(v807);
      const uint8_t* v809 = v21 + 54;
      const int8_t* v810 = (const int8_t*) v809;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v811 = *(const int8_t *)(v810);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v812 = __riscv_vwmacc_vx_i16m1(v782, v808, v803, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v813 = __riscv_vwmacc_vx_i16m1(v783, v811, v805, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v814 = v19 + 560;
      const uint8_t* v815 = (const uint8_t*) v814;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v816 = __riscv_vle8_v_u8mf2(v815, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v817 = __riscv_vand_vx_u8mf2(v816, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v818 = __riscv_vreinterpret_v_u8mf2_i8mf2(v817);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v819 = __riscv_vsrl_vx_u8mf2(v816, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v820 = __riscv_vreinterpret_v_u8mf2_i8mf2(v819);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v821 = v21 + 23;
      const int8_t* v822 = (const int8_t*) v821;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v823 = *(const int8_t *)(v822);
      const uint8_t* v824 = v21 + 55;
      const int8_t* v825 = (const int8_t*) v824;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v826 = *(const int8_t *)(v825);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v827 = __riscv_vwmacc_vx_i16m1(v797, v823, v818, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v828 = __riscv_vwmacc_vx_i16m1(v798, v826, v820, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v829 = v19 + 568;
      const uint8_t* v830 = (const uint8_t*) v829;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v831 = __riscv_vle8_v_u8mf2(v830, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v832 = __riscv_vand_vx_u8mf2(v831, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v833 = __riscv_vreinterpret_v_u8mf2_i8mf2(v832);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v834 = __riscv_vsrl_vx_u8mf2(v831, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v835 = __riscv_vreinterpret_v_u8mf2_i8mf2(v834);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v836 = v21 + 23;
      const int8_t* v837 = (const int8_t*) v836;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v838 = *(const int8_t *)(v837);
      const uint8_t* v839 = v21 + 55;
      const int8_t* v840 = (const int8_t*) v839;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v841 = *(const int8_t *)(v840);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v842 = __riscv_vwmacc_vx_i16m1(v812, v838, v833, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v843 = __riscv_vwmacc_vx_i16m1(v813, v841, v835, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v844 = v19 + 576;
      const uint8_t* v845 = (const uint8_t*) v844;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v846 = __riscv_vle8_v_u8mf2(v845, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v847 = __riscv_vand_vx_u8mf2(v846, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v848 = __riscv_vreinterpret_v_u8mf2_i8mf2(v847);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v849 = __riscv_vsrl_vx_u8mf2(v846, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v850 = __riscv_vreinterpret_v_u8mf2_i8mf2(v849);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v851 = v21 + 24;
      const int8_t* v852 = (const int8_t*) v851;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v853 = *(const int8_t *)(v852);
      const uint8_t* v854 = v21 + 56;
      const int8_t* v855 = (const int8_t*) v854;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v856 = *(const int8_t *)(v855);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v857 = __riscv_vwmacc_vx_i16m1(v827, v853, v848, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v858 = __riscv_vwmacc_vx_i16m1(v828, v856, v850, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v859 = v19 + 584;
      const uint8_t* v860 = (const uint8_t*) v859;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v861 = __riscv_vle8_v_u8mf2(v860, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v862 = __riscv_vand_vx_u8mf2(v861, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v863 = __riscv_vreinterpret_v_u8mf2_i8mf2(v862);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v864 = __riscv_vsrl_vx_u8mf2(v861, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v865 = __riscv_vreinterpret_v_u8mf2_i8mf2(v864);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v866 = v21 + 24;
      const int8_t* v867 = (const int8_t*) v866;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v868 = *(const int8_t *)(v867);
      const uint8_t* v869 = v21 + 56;
      const int8_t* v870 = (const int8_t*) v869;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v871 = *(const int8_t *)(v870);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v872 = __riscv_vwmacc_vx_i16m1(v842, v868, v863, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v873 = __riscv_vwmacc_vx_i16m1(v843, v871, v865, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v874 = v19 + 592;
      const uint8_t* v875 = (const uint8_t*) v874;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v876 = __riscv_vle8_v_u8mf2(v875, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v877 = __riscv_vand_vx_u8mf2(v876, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v878 = __riscv_vreinterpret_v_u8mf2_i8mf2(v877);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v879 = __riscv_vsrl_vx_u8mf2(v876, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v880 = __riscv_vreinterpret_v_u8mf2_i8mf2(v879);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v881 = v21 + 25;
      const int8_t* v882 = (const int8_t*) v881;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v883 = *(const int8_t *)(v882);
      const uint8_t* v884 = v21 + 57;
      const int8_t* v885 = (const int8_t*) v884;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v886 = *(const int8_t *)(v885);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v887 = __riscv_vwmacc_vx_i16m1(v857, v883, v878, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v888 = __riscv_vwmacc_vx_i16m1(v858, v886, v880, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v889 = v19 + 600;
      const uint8_t* v890 = (const uint8_t*) v889;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v891 = __riscv_vle8_v_u8mf2(v890, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v892 = __riscv_vand_vx_u8mf2(v891, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v892);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v894 = __riscv_vsrl_vx_u8mf2(v891, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v895 = __riscv_vreinterpret_v_u8mf2_i8mf2(v894);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v896 = v21 + 25;
      const int8_t* v897 = (const int8_t*) v896;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v898 = *(const int8_t *)(v897);
      const uint8_t* v899 = v21 + 57;
      const int8_t* v900 = (const int8_t*) v899;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v901 = *(const int8_t *)(v900);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v902 = __riscv_vwmacc_vx_i16m1(v872, v898, v893, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v903 = __riscv_vwmacc_vx_i16m1(v873, v901, v895, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v904 = v19 + 608;
      const uint8_t* v905 = (const uint8_t*) v904;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v906 = __riscv_vle8_v_u8mf2(v905, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v907 = __riscv_vand_vx_u8mf2(v906, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v908 = __riscv_vreinterpret_v_u8mf2_i8mf2(v907);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v909 = __riscv_vsrl_vx_u8mf2(v906, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v910 = __riscv_vreinterpret_v_u8mf2_i8mf2(v909);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v911 = v21 + 26;
      const int8_t* v912 = (const int8_t*) v911;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v913 = *(const int8_t *)(v912);
      const uint8_t* v914 = v21 + 58;
      const int8_t* v915 = (const int8_t*) v914;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v916 = *(const int8_t *)(v915);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v917 = __riscv_vwmacc_vx_i16m1(v887, v913, v908, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v918 = __riscv_vwmacc_vx_i16m1(v888, v916, v910, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v919 = v19 + 616;
      const uint8_t* v920 = (const uint8_t*) v919;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v921 = __riscv_vle8_v_u8mf2(v920, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v922 = __riscv_vand_vx_u8mf2(v921, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v923 = __riscv_vreinterpret_v_u8mf2_i8mf2(v922);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v924 = __riscv_vsrl_vx_u8mf2(v921, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v925 = __riscv_vreinterpret_v_u8mf2_i8mf2(v924);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v926 = v21 + 26;
      const int8_t* v927 = (const int8_t*) v926;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v928 = *(const int8_t *)(v927);
      const uint8_t* v929 = v21 + 58;
      const int8_t* v930 = (const int8_t*) v929;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v931 = *(const int8_t *)(v930);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v932 = __riscv_vwmacc_vx_i16m1(v902, v928, v923, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v933 = __riscv_vwmacc_vx_i16m1(v903, v931, v925, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v934 = v19 + 624;
      const uint8_t* v935 = (const uint8_t*) v934;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v936 = __riscv_vle8_v_u8mf2(v935, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v937 = __riscv_vand_vx_u8mf2(v936, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v938 = __riscv_vreinterpret_v_u8mf2_i8mf2(v937);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v939 = __riscv_vsrl_vx_u8mf2(v936, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v940 = __riscv_vreinterpret_v_u8mf2_i8mf2(v939);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v941 = v21 + 27;
      const int8_t* v942 = (const int8_t*) v941;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v943 = *(const int8_t *)(v942);
      const uint8_t* v944 = v21 + 59;
      const int8_t* v945 = (const int8_t*) v944;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v946 = *(const int8_t *)(v945);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v947 = __riscv_vwmacc_vx_i16m1(v917, v943, v938, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v948 = __riscv_vwmacc_vx_i16m1(v918, v946, v940, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v949 = v19 + 632;
      const uint8_t* v950 = (const uint8_t*) v949;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v951 = __riscv_vle8_v_u8mf2(v950, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v952 = __riscv_vand_vx_u8mf2(v951, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v953 = __riscv_vreinterpret_v_u8mf2_i8mf2(v952);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v954 = __riscv_vsrl_vx_u8mf2(v951, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v955 = __riscv_vreinterpret_v_u8mf2_i8mf2(v954);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v956 = v21 + 27;
      const int8_t* v957 = (const int8_t*) v956;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v958 = *(const int8_t *)(v957);
      const uint8_t* v959 = v21 + 59;
      const int8_t* v960 = (const int8_t*) v959;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v961 = *(const int8_t *)(v960);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v962 = __riscv_vwmacc_vx_i16m1(v932, v958, v953, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v963 = __riscv_vwmacc_vx_i16m1(v933, v961, v955, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v964 = v19 + 640;
      const uint8_t* v965 = (const uint8_t*) v964;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v966 = __riscv_vle8_v_u8mf2(v965, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v967 = __riscv_vand_vx_u8mf2(v966, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v968 = __riscv_vreinterpret_v_u8mf2_i8mf2(v967);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v969 = __riscv_vsrl_vx_u8mf2(v966, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v970 = __riscv_vreinterpret_v_u8mf2_i8mf2(v969);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v971 = v21 + 28;
      const int8_t* v972 = (const int8_t*) v971;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v973 = *(const int8_t *)(v972);
      const uint8_t* v974 = v21 + 60;
      const int8_t* v975 = (const int8_t*) v974;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v976 = *(const int8_t *)(v975);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v977 = __riscv_vwmacc_vx_i16m1(v947, v973, v968, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v978 = __riscv_vwmacc_vx_i16m1(v948, v976, v970, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v979 = v19 + 648;
      const uint8_t* v980 = (const uint8_t*) v979;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v981 = __riscv_vle8_v_u8mf2(v980, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v982 = __riscv_vand_vx_u8mf2(v981, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v983 = __riscv_vreinterpret_v_u8mf2_i8mf2(v982);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v984 = __riscv_vsrl_vx_u8mf2(v981, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v984);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v986 = v21 + 28;
      const int8_t* v987 = (const int8_t*) v986;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v988 = *(const int8_t *)(v987);
      const uint8_t* v989 = v21 + 60;
      const int8_t* v990 = (const int8_t*) v989;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v991 = *(const int8_t *)(v990);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v992 = __riscv_vwmacc_vx_i16m1(v962, v988, v983, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v993 = __riscv_vwmacc_vx_i16m1(v963, v991, v985, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v994 = v19 + 656;
      const uint8_t* v995 = (const uint8_t*) v994;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v996 = __riscv_vle8_v_u8mf2(v995, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v997 = __riscv_vand_vx_u8mf2(v996, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v998 = __riscv_vreinterpret_v_u8mf2_i8mf2(v997);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v999 = __riscv_vsrl_vx_u8mf2(v996, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1000 = __riscv_vreinterpret_v_u8mf2_i8mf2(v999);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1001 = v21 + 29;
      const int8_t* v1002 = (const int8_t*) v1001;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1003 = *(const int8_t *)(v1002);
      const uint8_t* v1004 = v21 + 61;
      const int8_t* v1005 = (const int8_t*) v1004;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1006 = *(const int8_t *)(v1005);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1007 = __riscv_vwmacc_vx_i16m1(v977, v1003, v998, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1008 = __riscv_vwmacc_vx_i16m1(v978, v1006, v1000, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1009 = v19 + 664;
      const uint8_t* v1010 = (const uint8_t*) v1009;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1011 = __riscv_vle8_v_u8mf2(v1010, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1012 = __riscv_vand_vx_u8mf2(v1011, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1013 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1012);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1014 = __riscv_vsrl_vx_u8mf2(v1011, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1015 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1014);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1016 = v21 + 29;
      const int8_t* v1017 = (const int8_t*) v1016;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1018 = *(const int8_t *)(v1017);
      const uint8_t* v1019 = v21 + 61;
      const int8_t* v1020 = (const int8_t*) v1019;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1021 = *(const int8_t *)(v1020);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1022 = __riscv_vwmacc_vx_i16m1(v992, v1018, v1013, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1023 = __riscv_vwmacc_vx_i16m1(v993, v1021, v1015, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1024 = v19 + 672;
      const uint8_t* v1025 = (const uint8_t*) v1024;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1026 = __riscv_vle8_v_u8mf2(v1025, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1027 = __riscv_vand_vx_u8mf2(v1026, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1028 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1027);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1029 = __riscv_vsrl_vx_u8mf2(v1026, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1030 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1029);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1031 = v21 + 30;
      const int8_t* v1032 = (const int8_t*) v1031;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1033 = *(const int8_t *)(v1032);
      const uint8_t* v1034 = v21 + 62;
      const int8_t* v1035 = (const int8_t*) v1034;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1036 = *(const int8_t *)(v1035);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1037 = __riscv_vwmacc_vx_i16m1(v1007, v1033, v1028, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1038 = __riscv_vwmacc_vx_i16m1(v1008, v1036, v1030, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1039 = v19 + 680;
      const uint8_t* v1040 = (const uint8_t*) v1039;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1041 = __riscv_vle8_v_u8mf2(v1040, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1042 = __riscv_vand_vx_u8mf2(v1041, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1043 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1042);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1044 = __riscv_vsrl_vx_u8mf2(v1041, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1045 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1044);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1046 = v21 + 30;
      const int8_t* v1047 = (const int8_t*) v1046;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1048 = *(const int8_t *)(v1047);
      const uint8_t* v1049 = v21 + 62;
      const int8_t* v1050 = (const int8_t*) v1049;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1051 = *(const int8_t *)(v1050);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1052 = __riscv_vwmacc_vx_i16m1(v1022, v1048, v1043, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1053 = __riscv_vwmacc_vx_i16m1(v1023, v1051, v1045, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1054 = v19 + 688;
      const uint8_t* v1055 = (const uint8_t*) v1054;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1056 = __riscv_vle8_v_u8mf2(v1055, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1057 = __riscv_vand_vx_u8mf2(v1056, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1058 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1057);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1059 = __riscv_vsrl_vx_u8mf2(v1056, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1060 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1059);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1061 = v21 + 31;
      const int8_t* v1062 = (const int8_t*) v1061;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1063 = *(const int8_t *)(v1062);
      const uint8_t* v1064 = v21 + 63;
      const int8_t* v1065 = (const int8_t*) v1064;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1066 = *(const int8_t *)(v1065);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1067 = __riscv_vwmacc_vx_i16m1(v1037, v1063, v1058, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1068 = __riscv_vwmacc_vx_i16m1(v1038, v1066, v1060, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1069 = v19 + 696;
      const uint8_t* v1070 = (const uint8_t*) v1069;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1071 = __riscv_vle8_v_u8mf2(v1070, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1072 = __riscv_vand_vx_u8mf2(v1071, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1072);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1074 = __riscv_vsrl_vx_u8mf2(v1071, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1075 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1074);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1076 = v21 + 31;
      const int8_t* v1077 = (const int8_t*) v1076;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1078 = *(const int8_t *)(v1077);
      const uint8_t* v1079 = v21 + 63;
      const int8_t* v1080 = (const int8_t*) v1079;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1081 = *(const int8_t *)(v1080);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1082 = __riscv_vwmacc_vx_i16m1(v1052, v1078, v1073, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1083 = __riscv_vwmacc_vx_i16m1(v1053, v1081, v1075, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1084 = v19 + 704;
      const uint8_t* v1085 = (const uint8_t*) v1084;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1086 = __riscv_vle8_v_u8mf2(v1085, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1087 = __riscv_vand_vx_u8mf2(v1086, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1088 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1087);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1089 = __riscv_vsrl_vx_u8mf2(v1086, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1090 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1089);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1091 = v21 + 32;
      const int8_t* v1092 = (const int8_t*) v1091;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1093 = *(const int8_t *)(v1092);
      const uint8_t* v1094 = v21 + 64;
      const int8_t* v1095 = (const int8_t*) v1094;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1096 = *(const int8_t *)(v1095);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1097 = __riscv_vwmacc_vx_i16m1(v1067, v1093, v1088, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1098 = __riscv_vwmacc_vx_i16m1(v1068, v1096, v1090, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1099 = v19 + 712;
      const uint8_t* v1100 = (const uint8_t*) v1099;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1101 = __riscv_vle8_v_u8mf2(v1100, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1102 = __riscv_vand_vx_u8mf2(v1101, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1103 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1102);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1104 = __riscv_vsrl_vx_u8mf2(v1101, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1105 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1104);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1106 = v21 + 32;
      const int8_t* v1107 = (const int8_t*) v1106;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1108 = *(const int8_t *)(v1107);
      const uint8_t* v1109 = v21 + 64;
      const int8_t* v1110 = (const int8_t*) v1109;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1111 = *(const int8_t *)(v1110);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1112 = __riscv_vwmacc_vx_i16m1(v1082, v1108, v1103, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1113 = __riscv_vwmacc_vx_i16m1(v1083, v1111, v1105, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1114 = v19 + 720;
      const uint8_t* v1115 = (const uint8_t*) v1114;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1116 = __riscv_vle8_v_u8mf2(v1115, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1117 = __riscv_vand_vx_u8mf2(v1116, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1118 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1117);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1119 = __riscv_vsrl_vx_u8mf2(v1116, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1119);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1121 = v21 + 33;
      const int8_t* v1122 = (const int8_t*) v1121;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1123 = *(const int8_t *)(v1122);
      const uint8_t* v1124 = v21 + 65;
      const int8_t* v1125 = (const int8_t*) v1124;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1126 = *(const int8_t *)(v1125);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1127 = __riscv_vwmacc_vx_i16m1(v1097, v1123, v1118, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1128 = __riscv_vwmacc_vx_i16m1(v1098, v1126, v1120, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1129 = v19 + 728;
      const uint8_t* v1130 = (const uint8_t*) v1129;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1131 = __riscv_vle8_v_u8mf2(v1130, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1132 = __riscv_vand_vx_u8mf2(v1131, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1132);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1134 = __riscv_vsrl_vx_u8mf2(v1131, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1135 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1134);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1136 = v21 + 33;
      const int8_t* v1137 = (const int8_t*) v1136;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1138 = *(const int8_t *)(v1137);
      const uint8_t* v1139 = v21 + 65;
      const int8_t* v1140 = (const int8_t*) v1139;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1141 = *(const int8_t *)(v1140);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1142 = __riscv_vwmacc_vx_i16m1(v1112, v1138, v1133, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1143 = __riscv_vwmacc_vx_i16m1(v1113, v1141, v1135, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1144 = v19 + 736;
      const uint8_t* v1145 = (const uint8_t*) v1144;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1146 = __riscv_vle8_v_u8mf2(v1145, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1147 = __riscv_vand_vx_u8mf2(v1146, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1148 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1147);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1149 = __riscv_vsrl_vx_u8mf2(v1146, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1150 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1149);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1151 = v21 + 34;
      const int8_t* v1152 = (const int8_t*) v1151;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1153 = *(const int8_t *)(v1152);
      const uint8_t* v1154 = v21 + 66;
      const int8_t* v1155 = (const int8_t*) v1154;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1156 = *(const int8_t *)(v1155);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1157 = __riscv_vwmacc_vx_i16m1(v1127, v1153, v1148, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1158 = __riscv_vwmacc_vx_i16m1(v1128, v1156, v1150, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1159 = v19 + 744;
      const uint8_t* v1160 = (const uint8_t*) v1159;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1161 = __riscv_vle8_v_u8mf2(v1160, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1162 = __riscv_vand_vx_u8mf2(v1161, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1163 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1162);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1164 = __riscv_vsrl_vx_u8mf2(v1161, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1164);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1166 = v21 + 34;
      const int8_t* v1167 = (const int8_t*) v1166;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1168 = *(const int8_t *)(v1167);
      const uint8_t* v1169 = v21 + 66;
      const int8_t* v1170 = (const int8_t*) v1169;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1171 = *(const int8_t *)(v1170);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1172 = __riscv_vwmacc_vx_i16m1(v1142, v1168, v1163, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1173 = __riscv_vwmacc_vx_i16m1(v1143, v1171, v1165, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1174 = v19 + 752;
      const uint8_t* v1175 = (const uint8_t*) v1174;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1176 = __riscv_vle8_v_u8mf2(v1175, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1177 = __riscv_vand_vx_u8mf2(v1176, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1178 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1177);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1179 = __riscv_vsrl_vx_u8mf2(v1176, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1180 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1179);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1181 = v21 + 35;
      const int8_t* v1182 = (const int8_t*) v1181;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1183 = *(const int8_t *)(v1182);
      const uint8_t* v1184 = v21 + 67;
      const int8_t* v1185 = (const int8_t*) v1184;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1186 = *(const int8_t *)(v1185);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1187 = __riscv_vwmacc_vx_i16m1(v1157, v1183, v1178, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1188 = __riscv_vwmacc_vx_i16m1(v1158, v1186, v1180, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1189 = v19 + 760;
      const uint8_t* v1190 = (const uint8_t*) v1189;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1191 = __riscv_vle8_v_u8mf2(v1190, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1192 = __riscv_vand_vx_u8mf2(v1191, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1192);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1194 = __riscv_vsrl_vx_u8mf2(v1191, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1195 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1194);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1196 = v21 + 35;
      const int8_t* v1197 = (const int8_t*) v1196;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1198 = *(const int8_t *)(v1197);
      const uint8_t* v1199 = v21 + 67;
      const int8_t* v1200 = (const int8_t*) v1199;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1201 = *(const int8_t *)(v1200);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1202 = __riscv_vwmacc_vx_i16m1(v1172, v1198, v1193, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1203 = __riscv_vwmacc_vx_i16m1(v1173, v1201, v1195, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1204 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1205 = __riscv_vwmacc_vv_i32m2(v1204, v57, v1187, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1206 = __riscv_vwmacc_vv_i32m2(v1205, v75, v1188, 8);
      v34 = v1206;
      vint32m2_t v1207 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1208 = __riscv_vwmacc_vv_i32m2(v1207, v129, v1202, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1209 = __riscv_vwmacc_vv_i32m2(v1208, v147, v1203, 8);
      v38 = v1209;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1210 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1211 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1212 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1213 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1214 = v19 + 768;
      const uint8_t* v1215 = (const uint8_t*) v1214;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1216 = __riscv_vle8_v_u8mf2(v1215, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1217 = __riscv_vand_vx_u8mf2(v1216, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1218 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1217);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1219 = __riscv_vsrl_vx_u8mf2(v1216, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1220 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1219);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1221 = v21 + 68;
      const int8_t* v1222 = (const int8_t*) v1221;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1223 = *(const int8_t *)(v1222);
      const uint8_t* v1224 = v21 + 100;
      const int8_t* v1225 = (const int8_t*) v1224;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1226 = *(const int8_t *)(v1225);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1227 = __riscv_vwmacc_vx_i16m1(v1210, v1223, v1218, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1228 = __riscv_vwmacc_vx_i16m1(v1211, v1226, v1220, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1229 = v19 + 776;
      const uint8_t* v1230 = (const uint8_t*) v1229;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1231 = __riscv_vle8_v_u8mf2(v1230, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1232 = __riscv_vand_vx_u8mf2(v1231, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1233 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1232);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1234 = __riscv_vsrl_vx_u8mf2(v1231, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1235 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1234);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1236 = v21 + 68;
      const int8_t* v1237 = (const int8_t*) v1236;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1238 = *(const int8_t *)(v1237);
      const uint8_t* v1239 = v21 + 100;
      const int8_t* v1240 = (const int8_t*) v1239;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1241 = *(const int8_t *)(v1240);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1242 = __riscv_vwmacc_vx_i16m1(v1212, v1238, v1233, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1243 = __riscv_vwmacc_vx_i16m1(v1213, v1241, v1235, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1244 = v19 + 784;
      const uint8_t* v1245 = (const uint8_t*) v1244;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1246 = __riscv_vle8_v_u8mf2(v1245, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1247 = __riscv_vand_vx_u8mf2(v1246, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1248 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1247);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1249 = __riscv_vsrl_vx_u8mf2(v1246, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1249);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1251 = v21 + 69;
      const int8_t* v1252 = (const int8_t*) v1251;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1253 = *(const int8_t *)(v1252);
      const uint8_t* v1254 = v21 + 101;
      const int8_t* v1255 = (const int8_t*) v1254;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1256 = *(const int8_t *)(v1255);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1257 = __riscv_vwmacc_vx_i16m1(v1227, v1253, v1248, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1258 = __riscv_vwmacc_vx_i16m1(v1228, v1256, v1250, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1259 = v19 + 792;
      const uint8_t* v1260 = (const uint8_t*) v1259;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1261 = __riscv_vle8_v_u8mf2(v1260, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1262 = __riscv_vand_vx_u8mf2(v1261, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1263 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1262);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1264 = __riscv_vsrl_vx_u8mf2(v1261, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1265 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1264);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1266 = v21 + 69;
      const int8_t* v1267 = (const int8_t*) v1266;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1268 = *(const int8_t *)(v1267);
      const uint8_t* v1269 = v21 + 101;
      const int8_t* v1270 = (const int8_t*) v1269;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1271 = *(const int8_t *)(v1270);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1272 = __riscv_vwmacc_vx_i16m1(v1242, v1268, v1263, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1273 = __riscv_vwmacc_vx_i16m1(v1243, v1271, v1265, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1274 = v19 + 800;
      const uint8_t* v1275 = (const uint8_t*) v1274;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1276 = __riscv_vle8_v_u8mf2(v1275, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1277 = __riscv_vand_vx_u8mf2(v1276, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1278 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1277);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1279 = __riscv_vsrl_vx_u8mf2(v1276, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1280 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1279);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1281 = v21 + 70;
      const int8_t* v1282 = (const int8_t*) v1281;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1283 = *(const int8_t *)(v1282);
      const uint8_t* v1284 = v21 + 102;
      const int8_t* v1285 = (const int8_t*) v1284;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1286 = *(const int8_t *)(v1285);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1287 = __riscv_vwmacc_vx_i16m1(v1257, v1283, v1278, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1288 = __riscv_vwmacc_vx_i16m1(v1258, v1286, v1280, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1289 = v19 + 808;
      const uint8_t* v1290 = (const uint8_t*) v1289;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1291 = __riscv_vle8_v_u8mf2(v1290, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1292 = __riscv_vand_vx_u8mf2(v1291, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1293 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1292);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1294 = __riscv_vsrl_vx_u8mf2(v1291, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1295 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1294);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1296 = v21 + 70;
      const int8_t* v1297 = (const int8_t*) v1296;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1298 = *(const int8_t *)(v1297);
      const uint8_t* v1299 = v21 + 102;
      const int8_t* v1300 = (const int8_t*) v1299;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1301 = *(const int8_t *)(v1300);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1302 = __riscv_vwmacc_vx_i16m1(v1272, v1298, v1293, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1303 = __riscv_vwmacc_vx_i16m1(v1273, v1301, v1295, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1304 = v19 + 816;
      const uint8_t* v1305 = (const uint8_t*) v1304;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1306 = __riscv_vle8_v_u8mf2(v1305, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1307 = __riscv_vand_vx_u8mf2(v1306, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1308 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1307);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1309 = __riscv_vsrl_vx_u8mf2(v1306, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1310 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1309);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1311 = v21 + 71;
      const int8_t* v1312 = (const int8_t*) v1311;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1313 = *(const int8_t *)(v1312);
      const uint8_t* v1314 = v21 + 103;
      const int8_t* v1315 = (const int8_t*) v1314;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1316 = *(const int8_t *)(v1315);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1317 = __riscv_vwmacc_vx_i16m1(v1287, v1313, v1308, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1318 = __riscv_vwmacc_vx_i16m1(v1288, v1316, v1310, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1319 = v19 + 824;
      const uint8_t* v1320 = (const uint8_t*) v1319;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1321 = __riscv_vle8_v_u8mf2(v1320, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1322 = __riscv_vand_vx_u8mf2(v1321, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1323 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1322);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1324 = __riscv_vsrl_vx_u8mf2(v1321, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1325 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1324);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1326 = v21 + 71;
      const int8_t* v1327 = (const int8_t*) v1326;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1328 = *(const int8_t *)(v1327);
      const uint8_t* v1329 = v21 + 103;
      const int8_t* v1330 = (const int8_t*) v1329;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1331 = *(const int8_t *)(v1330);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1332 = __riscv_vwmacc_vx_i16m1(v1302, v1328, v1323, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1333 = __riscv_vwmacc_vx_i16m1(v1303, v1331, v1325, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1334 = v19 + 832;
      const uint8_t* v1335 = (const uint8_t*) v1334;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1336 = __riscv_vle8_v_u8mf2(v1335, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1337 = __riscv_vand_vx_u8mf2(v1336, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1338 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1337);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1339 = __riscv_vsrl_vx_u8mf2(v1336, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1340 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1339);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1341 = v21 + 72;
      const int8_t* v1342 = (const int8_t*) v1341;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1343 = *(const int8_t *)(v1342);
      const uint8_t* v1344 = v21 + 104;
      const int8_t* v1345 = (const int8_t*) v1344;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1346 = *(const int8_t *)(v1345);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1347 = __riscv_vwmacc_vx_i16m1(v1317, v1343, v1338, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1348 = __riscv_vwmacc_vx_i16m1(v1318, v1346, v1340, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1349 = v19 + 840;
      const uint8_t* v1350 = (const uint8_t*) v1349;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1351 = __riscv_vle8_v_u8mf2(v1350, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1352 = __riscv_vand_vx_u8mf2(v1351, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1353 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1352);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1354 = __riscv_vsrl_vx_u8mf2(v1351, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1355 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1354);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1356 = v21 + 72;
      const int8_t* v1357 = (const int8_t*) v1356;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1358 = *(const int8_t *)(v1357);
      const uint8_t* v1359 = v21 + 104;
      const int8_t* v1360 = (const int8_t*) v1359;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1361 = *(const int8_t *)(v1360);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1362 = __riscv_vwmacc_vx_i16m1(v1332, v1358, v1353, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1363 = __riscv_vwmacc_vx_i16m1(v1333, v1361, v1355, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1364 = v19 + 848;
      const uint8_t* v1365 = (const uint8_t*) v1364;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1366 = __riscv_vle8_v_u8mf2(v1365, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1367 = __riscv_vand_vx_u8mf2(v1366, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1368 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1367);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1369 = __riscv_vsrl_vx_u8mf2(v1366, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1370 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1369);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1371 = v21 + 73;
      const int8_t* v1372 = (const int8_t*) v1371;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1373 = *(const int8_t *)(v1372);
      const uint8_t* v1374 = v21 + 105;
      const int8_t* v1375 = (const int8_t*) v1374;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1376 = *(const int8_t *)(v1375);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1377 = __riscv_vwmacc_vx_i16m1(v1347, v1373, v1368, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1378 = __riscv_vwmacc_vx_i16m1(v1348, v1376, v1370, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1379 = v19 + 856;
      const uint8_t* v1380 = (const uint8_t*) v1379;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1381 = __riscv_vle8_v_u8mf2(v1380, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1382 = __riscv_vand_vx_u8mf2(v1381, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1383 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1382);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1384 = __riscv_vsrl_vx_u8mf2(v1381, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1385 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1384);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1386 = v21 + 73;
      const int8_t* v1387 = (const int8_t*) v1386;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1388 = *(const int8_t *)(v1387);
      const uint8_t* v1389 = v21 + 105;
      const int8_t* v1390 = (const int8_t*) v1389;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1391 = *(const int8_t *)(v1390);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1392 = __riscv_vwmacc_vx_i16m1(v1362, v1388, v1383, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1393 = __riscv_vwmacc_vx_i16m1(v1363, v1391, v1385, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1394 = v19 + 864;
      const uint8_t* v1395 = (const uint8_t*) v1394;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1396 = __riscv_vle8_v_u8mf2(v1395, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1397 = __riscv_vand_vx_u8mf2(v1396, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1398 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1397);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1399 = __riscv_vsrl_vx_u8mf2(v1396, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1400 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1399);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1401 = v21 + 74;
      const int8_t* v1402 = (const int8_t*) v1401;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1403 = *(const int8_t *)(v1402);
      const uint8_t* v1404 = v21 + 106;
      const int8_t* v1405 = (const int8_t*) v1404;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1406 = *(const int8_t *)(v1405);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1407 = __riscv_vwmacc_vx_i16m1(v1377, v1403, v1398, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1408 = __riscv_vwmacc_vx_i16m1(v1378, v1406, v1400, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1409 = v19 + 872;
      const uint8_t* v1410 = (const uint8_t*) v1409;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1411 = __riscv_vle8_v_u8mf2(v1410, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1412 = __riscv_vand_vx_u8mf2(v1411, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1413 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1412);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1414 = __riscv_vsrl_vx_u8mf2(v1411, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1415 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1414);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1416 = v21 + 74;
      const int8_t* v1417 = (const int8_t*) v1416;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1418 = *(const int8_t *)(v1417);
      const uint8_t* v1419 = v21 + 106;
      const int8_t* v1420 = (const int8_t*) v1419;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1421 = *(const int8_t *)(v1420);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1422 = __riscv_vwmacc_vx_i16m1(v1392, v1418, v1413, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1423 = __riscv_vwmacc_vx_i16m1(v1393, v1421, v1415, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1424 = v19 + 880;
      const uint8_t* v1425 = (const uint8_t*) v1424;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1426 = __riscv_vle8_v_u8mf2(v1425, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1427 = __riscv_vand_vx_u8mf2(v1426, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1428 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1427);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1429 = __riscv_vsrl_vx_u8mf2(v1426, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1430 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1429);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1431 = v21 + 75;
      const int8_t* v1432 = (const int8_t*) v1431;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1433 = *(const int8_t *)(v1432);
      const uint8_t* v1434 = v21 + 107;
      const int8_t* v1435 = (const int8_t*) v1434;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1436 = *(const int8_t *)(v1435);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1437 = __riscv_vwmacc_vx_i16m1(v1407, v1433, v1428, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1438 = __riscv_vwmacc_vx_i16m1(v1408, v1436, v1430, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1439 = v19 + 888;
      const uint8_t* v1440 = (const uint8_t*) v1439;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1441 = __riscv_vle8_v_u8mf2(v1440, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1442 = __riscv_vand_vx_u8mf2(v1441, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1443 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1442);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1444 = __riscv_vsrl_vx_u8mf2(v1441, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1445 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1444);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1446 = v21 + 75;
      const int8_t* v1447 = (const int8_t*) v1446;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1448 = *(const int8_t *)(v1447);
      const uint8_t* v1449 = v21 + 107;
      const int8_t* v1450 = (const int8_t*) v1449;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1451 = *(const int8_t *)(v1450);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1452 = __riscv_vwmacc_vx_i16m1(v1422, v1448, v1443, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1453 = __riscv_vwmacc_vx_i16m1(v1423, v1451, v1445, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1454 = v19 + 896;
      const uint8_t* v1455 = (const uint8_t*) v1454;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1456 = __riscv_vle8_v_u8mf2(v1455, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1457 = __riscv_vand_vx_u8mf2(v1456, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1458 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1457);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1459 = __riscv_vsrl_vx_u8mf2(v1456, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1460 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1459);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1461 = v21 + 76;
      const int8_t* v1462 = (const int8_t*) v1461;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1463 = *(const int8_t *)(v1462);
      const uint8_t* v1464 = v21 + 108;
      const int8_t* v1465 = (const int8_t*) v1464;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1466 = *(const int8_t *)(v1465);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1467 = __riscv_vwmacc_vx_i16m1(v1437, v1463, v1458, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1468 = __riscv_vwmacc_vx_i16m1(v1438, v1466, v1460, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1469 = v19 + 904;
      const uint8_t* v1470 = (const uint8_t*) v1469;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1471 = __riscv_vle8_v_u8mf2(v1470, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1472 = __riscv_vand_vx_u8mf2(v1471, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1473 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1472);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1474 = __riscv_vsrl_vx_u8mf2(v1471, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1475 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1474);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1476 = v21 + 76;
      const int8_t* v1477 = (const int8_t*) v1476;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1478 = *(const int8_t *)(v1477);
      const uint8_t* v1479 = v21 + 108;
      const int8_t* v1480 = (const int8_t*) v1479;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1481 = *(const int8_t *)(v1480);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1482 = __riscv_vwmacc_vx_i16m1(v1452, v1478, v1473, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1483 = __riscv_vwmacc_vx_i16m1(v1453, v1481, v1475, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1484 = v19 + 912;
      const uint8_t* v1485 = (const uint8_t*) v1484;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1486 = __riscv_vle8_v_u8mf2(v1485, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1487 = __riscv_vand_vx_u8mf2(v1486, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1488 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1487);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1489 = __riscv_vsrl_vx_u8mf2(v1486, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1490 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1489);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1491 = v21 + 77;
      const int8_t* v1492 = (const int8_t*) v1491;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1493 = *(const int8_t *)(v1492);
      const uint8_t* v1494 = v21 + 109;
      const int8_t* v1495 = (const int8_t*) v1494;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1496 = *(const int8_t *)(v1495);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1497 = __riscv_vwmacc_vx_i16m1(v1467, v1493, v1488, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1498 = __riscv_vwmacc_vx_i16m1(v1468, v1496, v1490, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1499 = v19 + 920;
      const uint8_t* v1500 = (const uint8_t*) v1499;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1501 = __riscv_vle8_v_u8mf2(v1500, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1502 = __riscv_vand_vx_u8mf2(v1501, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1503 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1502);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1504 = __riscv_vsrl_vx_u8mf2(v1501, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1505 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1504);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1506 = v21 + 77;
      const int8_t* v1507 = (const int8_t*) v1506;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1508 = *(const int8_t *)(v1507);
      const uint8_t* v1509 = v21 + 109;
      const int8_t* v1510 = (const int8_t*) v1509;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1511 = *(const int8_t *)(v1510);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1512 = __riscv_vwmacc_vx_i16m1(v1482, v1508, v1503, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1513 = __riscv_vwmacc_vx_i16m1(v1483, v1511, v1505, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1514 = v19 + 928;
      const uint8_t* v1515 = (const uint8_t*) v1514;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1516 = __riscv_vle8_v_u8mf2(v1515, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1517 = __riscv_vand_vx_u8mf2(v1516, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1518 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1517);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1519 = __riscv_vsrl_vx_u8mf2(v1516, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1520 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1519);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1521 = v21 + 78;
      const int8_t* v1522 = (const int8_t*) v1521;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1523 = *(const int8_t *)(v1522);
      const uint8_t* v1524 = v21 + 110;
      const int8_t* v1525 = (const int8_t*) v1524;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1526 = *(const int8_t *)(v1525);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1527 = __riscv_vwmacc_vx_i16m1(v1497, v1523, v1518, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1528 = __riscv_vwmacc_vx_i16m1(v1498, v1526, v1520, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1529 = v19 + 936;
      const uint8_t* v1530 = (const uint8_t*) v1529;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1531 = __riscv_vle8_v_u8mf2(v1530, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1532 = __riscv_vand_vx_u8mf2(v1531, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1533 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1532);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1534 = __riscv_vsrl_vx_u8mf2(v1531, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1535 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1534);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1536 = v21 + 78;
      const int8_t* v1537 = (const int8_t*) v1536;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1538 = *(const int8_t *)(v1537);
      const uint8_t* v1539 = v21 + 110;
      const int8_t* v1540 = (const int8_t*) v1539;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1541 = *(const int8_t *)(v1540);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1542 = __riscv_vwmacc_vx_i16m1(v1512, v1538, v1533, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1543 = __riscv_vwmacc_vx_i16m1(v1513, v1541, v1535, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1544 = v19 + 944;
      const uint8_t* v1545 = (const uint8_t*) v1544;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1546 = __riscv_vle8_v_u8mf2(v1545, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1547 = __riscv_vand_vx_u8mf2(v1546, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1548 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1547);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1549 = __riscv_vsrl_vx_u8mf2(v1546, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1550 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1549);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1551 = v21 + 79;
      const int8_t* v1552 = (const int8_t*) v1551;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1553 = *(const int8_t *)(v1552);
      const uint8_t* v1554 = v21 + 111;
      const int8_t* v1555 = (const int8_t*) v1554;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1556 = *(const int8_t *)(v1555);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1557 = __riscv_vwmacc_vx_i16m1(v1527, v1553, v1548, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1558 = __riscv_vwmacc_vx_i16m1(v1528, v1556, v1550, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1559 = v19 + 952;
      const uint8_t* v1560 = (const uint8_t*) v1559;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1561 = __riscv_vle8_v_u8mf2(v1560, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1562 = __riscv_vand_vx_u8mf2(v1561, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1563 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1562);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1564 = __riscv_vsrl_vx_u8mf2(v1561, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1565 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1564);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1566 = v21 + 79;
      const int8_t* v1567 = (const int8_t*) v1566;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1568 = *(const int8_t *)(v1567);
      const uint8_t* v1569 = v21 + 111;
      const int8_t* v1570 = (const int8_t*) v1569;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1571 = *(const int8_t *)(v1570);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1572 = __riscv_vwmacc_vx_i16m1(v1542, v1568, v1563, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1573 = __riscv_vwmacc_vx_i16m1(v1543, v1571, v1565, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1574 = v19 + 960;
      const uint8_t* v1575 = (const uint8_t*) v1574;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1576 = __riscv_vle8_v_u8mf2(v1575, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1577 = __riscv_vand_vx_u8mf2(v1576, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1578 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1577);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1579 = __riscv_vsrl_vx_u8mf2(v1576, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1580 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1579);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1581 = v21 + 80;
      const int8_t* v1582 = (const int8_t*) v1581;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1583 = *(const int8_t *)(v1582);
      const uint8_t* v1584 = v21 + 112;
      const int8_t* v1585 = (const int8_t*) v1584;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1586 = *(const int8_t *)(v1585);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1587 = __riscv_vwmacc_vx_i16m1(v1557, v1583, v1578, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1588 = __riscv_vwmacc_vx_i16m1(v1558, v1586, v1580, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1589 = v19 + 968;
      const uint8_t* v1590 = (const uint8_t*) v1589;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1591 = __riscv_vle8_v_u8mf2(v1590, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1592 = __riscv_vand_vx_u8mf2(v1591, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1593 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1592);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1594 = __riscv_vsrl_vx_u8mf2(v1591, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1595 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1594);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1596 = v21 + 80;
      const int8_t* v1597 = (const int8_t*) v1596;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1598 = *(const int8_t *)(v1597);
      const uint8_t* v1599 = v21 + 112;
      const int8_t* v1600 = (const int8_t*) v1599;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1601 = *(const int8_t *)(v1600);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1602 = __riscv_vwmacc_vx_i16m1(v1572, v1598, v1593, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1603 = __riscv_vwmacc_vx_i16m1(v1573, v1601, v1595, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1604 = v19 + 976;
      const uint8_t* v1605 = (const uint8_t*) v1604;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1606 = __riscv_vle8_v_u8mf2(v1605, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1607 = __riscv_vand_vx_u8mf2(v1606, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1608 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1607);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1609 = __riscv_vsrl_vx_u8mf2(v1606, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1610 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1609);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1611 = v21 + 81;
      const int8_t* v1612 = (const int8_t*) v1611;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1613 = *(const int8_t *)(v1612);
      const uint8_t* v1614 = v21 + 113;
      const int8_t* v1615 = (const int8_t*) v1614;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1616 = *(const int8_t *)(v1615);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1617 = __riscv_vwmacc_vx_i16m1(v1587, v1613, v1608, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1618 = __riscv_vwmacc_vx_i16m1(v1588, v1616, v1610, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1619 = v19 + 984;
      const uint8_t* v1620 = (const uint8_t*) v1619;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1621 = __riscv_vle8_v_u8mf2(v1620, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1622 = __riscv_vand_vx_u8mf2(v1621, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1623 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1622);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1624 = __riscv_vsrl_vx_u8mf2(v1621, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1625 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1624);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1626 = v21 + 81;
      const int8_t* v1627 = (const int8_t*) v1626;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1628 = *(const int8_t *)(v1627);
      const uint8_t* v1629 = v21 + 113;
      const int8_t* v1630 = (const int8_t*) v1629;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1631 = *(const int8_t *)(v1630);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1632 = __riscv_vwmacc_vx_i16m1(v1602, v1628, v1623, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1633 = __riscv_vwmacc_vx_i16m1(v1603, v1631, v1625, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1634 = v19 + 992;
      const uint8_t* v1635 = (const uint8_t*) v1634;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1636 = __riscv_vle8_v_u8mf2(v1635, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1637 = __riscv_vand_vx_u8mf2(v1636, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1638 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1637);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1639 = __riscv_vsrl_vx_u8mf2(v1636, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1640 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1639);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1641 = v21 + 82;
      const int8_t* v1642 = (const int8_t*) v1641;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1643 = *(const int8_t *)(v1642);
      const uint8_t* v1644 = v21 + 114;
      const int8_t* v1645 = (const int8_t*) v1644;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1646 = *(const int8_t *)(v1645);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1647 = __riscv_vwmacc_vx_i16m1(v1617, v1643, v1638, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1648 = __riscv_vwmacc_vx_i16m1(v1618, v1646, v1640, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1649 = v19 + 1000;
      const uint8_t* v1650 = (const uint8_t*) v1649;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1651 = __riscv_vle8_v_u8mf2(v1650, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1652 = __riscv_vand_vx_u8mf2(v1651, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1653 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1652);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1654 = __riscv_vsrl_vx_u8mf2(v1651, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1655 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1654);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1656 = v21 + 82;
      const int8_t* v1657 = (const int8_t*) v1656;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1658 = *(const int8_t *)(v1657);
      const uint8_t* v1659 = v21 + 114;
      const int8_t* v1660 = (const int8_t*) v1659;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1661 = *(const int8_t *)(v1660);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1662 = __riscv_vwmacc_vx_i16m1(v1632, v1658, v1653, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1663 = __riscv_vwmacc_vx_i16m1(v1633, v1661, v1655, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1664 = v19 + 1008;
      const uint8_t* v1665 = (const uint8_t*) v1664;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1666 = __riscv_vle8_v_u8mf2(v1665, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1667 = __riscv_vand_vx_u8mf2(v1666, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1668 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1667);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1669 = __riscv_vsrl_vx_u8mf2(v1666, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1670 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1669);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1671 = v21 + 83;
      const int8_t* v1672 = (const int8_t*) v1671;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1673 = *(const int8_t *)(v1672);
      const uint8_t* v1674 = v21 + 115;
      const int8_t* v1675 = (const int8_t*) v1674;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1676 = *(const int8_t *)(v1675);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1677 = __riscv_vwmacc_vx_i16m1(v1647, v1673, v1668, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1678 = __riscv_vwmacc_vx_i16m1(v1648, v1676, v1670, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1679 = v19 + 1016;
      const uint8_t* v1680 = (const uint8_t*) v1679;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1681 = __riscv_vle8_v_u8mf2(v1680, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1682 = __riscv_vand_vx_u8mf2(v1681, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1683 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1682);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1684 = __riscv_vsrl_vx_u8mf2(v1681, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1685 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1684);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1686 = v21 + 83;
      const int8_t* v1687 = (const int8_t*) v1686;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1688 = *(const int8_t *)(v1687);
      const uint8_t* v1689 = v21 + 115;
      const int8_t* v1690 = (const int8_t*) v1689;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1691 = *(const int8_t *)(v1690);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1692 = __riscv_vwmacc_vx_i16m1(v1662, v1688, v1683, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1693 = __riscv_vwmacc_vx_i16m1(v1663, v1691, v1685, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1694 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1695 = __riscv_vwmacc_vv_i32m2(v1694, v93, v1677, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1696 = __riscv_vwmacc_vv_i32m2(v1695, v111, v1678, 8);
      v34 = v1696;
      vint32m2_t v1697 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1698 = __riscv_vwmacc_vv_i32m2(v1697, v165, v1692, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1699 = __riscv_vwmacc_vv_i32m2(v1698, v183, v1693, 8);
      v38 = v1699;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1700 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1701 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1702 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1703 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1704 = v19 + 1024;
      const uint8_t* v1705 = (const uint8_t*) v1704;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1706 = __riscv_vle8_v_u8mf2(v1705, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1707 = __riscv_vand_vx_u8mf2(v1706, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1708 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1707);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1709 = __riscv_vsrl_vx_u8mf2(v1706, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1710 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1709);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1711 = v21 + 84;
      const int8_t* v1712 = (const int8_t*) v1711;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1713 = *(const int8_t *)(v1712);
      const uint8_t* v1714 = v21 + 116;
      const int8_t* v1715 = (const int8_t*) v1714;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1716 = *(const int8_t *)(v1715);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1717 = __riscv_vwmacc_vx_i16m1(v1700, v1713, v1708, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1718 = __riscv_vwmacc_vx_i16m1(v1701, v1716, v1710, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1719 = v19 + 1032;
      const uint8_t* v1720 = (const uint8_t*) v1719;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1721 = __riscv_vle8_v_u8mf2(v1720, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1722 = __riscv_vand_vx_u8mf2(v1721, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1723 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1722);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1724 = __riscv_vsrl_vx_u8mf2(v1721, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1724);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1726 = v21 + 84;
      const int8_t* v1727 = (const int8_t*) v1726;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1728 = *(const int8_t *)(v1727);
      const uint8_t* v1729 = v21 + 116;
      const int8_t* v1730 = (const int8_t*) v1729;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1731 = *(const int8_t *)(v1730);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1732 = __riscv_vwmacc_vx_i16m1(v1702, v1728, v1723, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1733 = __riscv_vwmacc_vx_i16m1(v1703, v1731, v1725, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1734 = v19 + 1040;
      const uint8_t* v1735 = (const uint8_t*) v1734;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1736 = __riscv_vle8_v_u8mf2(v1735, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1737 = __riscv_vand_vx_u8mf2(v1736, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1738 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1737);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1739 = __riscv_vsrl_vx_u8mf2(v1736, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1740 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1739);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1741 = v21 + 85;
      const int8_t* v1742 = (const int8_t*) v1741;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1743 = *(const int8_t *)(v1742);
      const uint8_t* v1744 = v21 + 117;
      const int8_t* v1745 = (const int8_t*) v1744;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1746 = *(const int8_t *)(v1745);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1747 = __riscv_vwmacc_vx_i16m1(v1717, v1743, v1738, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1748 = __riscv_vwmacc_vx_i16m1(v1718, v1746, v1740, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1749 = v19 + 1048;
      const uint8_t* v1750 = (const uint8_t*) v1749;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1751 = __riscv_vle8_v_u8mf2(v1750, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1752 = __riscv_vand_vx_u8mf2(v1751, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1753 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1752);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1754 = __riscv_vsrl_vx_u8mf2(v1751, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1755 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1754);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1756 = v21 + 85;
      const int8_t* v1757 = (const int8_t*) v1756;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1758 = *(const int8_t *)(v1757);
      const uint8_t* v1759 = v21 + 117;
      const int8_t* v1760 = (const int8_t*) v1759;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1761 = *(const int8_t *)(v1760);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1762 = __riscv_vwmacc_vx_i16m1(v1732, v1758, v1753, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1763 = __riscv_vwmacc_vx_i16m1(v1733, v1761, v1755, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1764 = v19 + 1056;
      const uint8_t* v1765 = (const uint8_t*) v1764;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1766 = __riscv_vle8_v_u8mf2(v1765, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1767 = __riscv_vand_vx_u8mf2(v1766, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1768 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1767);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1769 = __riscv_vsrl_vx_u8mf2(v1766, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1770 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1769);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1771 = v21 + 86;
      const int8_t* v1772 = (const int8_t*) v1771;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1773 = *(const int8_t *)(v1772);
      const uint8_t* v1774 = v21 + 118;
      const int8_t* v1775 = (const int8_t*) v1774;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1776 = *(const int8_t *)(v1775);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1777 = __riscv_vwmacc_vx_i16m1(v1747, v1773, v1768, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1778 = __riscv_vwmacc_vx_i16m1(v1748, v1776, v1770, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1779 = v19 + 1064;
      const uint8_t* v1780 = (const uint8_t*) v1779;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1781 = __riscv_vle8_v_u8mf2(v1780, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1782 = __riscv_vand_vx_u8mf2(v1781, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1783 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1782);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1784 = __riscv_vsrl_vx_u8mf2(v1781, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1784);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1786 = v21 + 86;
      const int8_t* v1787 = (const int8_t*) v1786;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1788 = *(const int8_t *)(v1787);
      const uint8_t* v1789 = v21 + 118;
      const int8_t* v1790 = (const int8_t*) v1789;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1791 = *(const int8_t *)(v1790);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1792 = __riscv_vwmacc_vx_i16m1(v1762, v1788, v1783, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1793 = __riscv_vwmacc_vx_i16m1(v1763, v1791, v1785, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1794 = v19 + 1072;
      const uint8_t* v1795 = (const uint8_t*) v1794;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1796 = __riscv_vle8_v_u8mf2(v1795, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1797 = __riscv_vand_vx_u8mf2(v1796, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1798 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1797);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1799 = __riscv_vsrl_vx_u8mf2(v1796, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1800 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1799);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1801 = v21 + 87;
      const int8_t* v1802 = (const int8_t*) v1801;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1803 = *(const int8_t *)(v1802);
      const uint8_t* v1804 = v21 + 119;
      const int8_t* v1805 = (const int8_t*) v1804;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1806 = *(const int8_t *)(v1805);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1807 = __riscv_vwmacc_vx_i16m1(v1777, v1803, v1798, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1808 = __riscv_vwmacc_vx_i16m1(v1778, v1806, v1800, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1809 = v19 + 1080;
      const uint8_t* v1810 = (const uint8_t*) v1809;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1811 = __riscv_vle8_v_u8mf2(v1810, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1812 = __riscv_vand_vx_u8mf2(v1811, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1813 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1812);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1814 = __riscv_vsrl_vx_u8mf2(v1811, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1814);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1816 = v21 + 87;
      const int8_t* v1817 = (const int8_t*) v1816;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1818 = *(const int8_t *)(v1817);
      const uint8_t* v1819 = v21 + 119;
      const int8_t* v1820 = (const int8_t*) v1819;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1821 = *(const int8_t *)(v1820);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1822 = __riscv_vwmacc_vx_i16m1(v1792, v1818, v1813, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1823 = __riscv_vwmacc_vx_i16m1(v1793, v1821, v1815, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1824 = v19 + 1088;
      const uint8_t* v1825 = (const uint8_t*) v1824;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1826 = __riscv_vle8_v_u8mf2(v1825, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1827 = __riscv_vand_vx_u8mf2(v1826, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1828 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1827);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1829 = __riscv_vsrl_vx_u8mf2(v1826, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1830 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1829);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1831 = v21 + 88;
      const int8_t* v1832 = (const int8_t*) v1831;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1833 = *(const int8_t *)(v1832);
      const uint8_t* v1834 = v21 + 120;
      const int8_t* v1835 = (const int8_t*) v1834;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1836 = *(const int8_t *)(v1835);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1837 = __riscv_vwmacc_vx_i16m1(v1807, v1833, v1828, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1838 = __riscv_vwmacc_vx_i16m1(v1808, v1836, v1830, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1839 = v19 + 1096;
      const uint8_t* v1840 = (const uint8_t*) v1839;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1841 = __riscv_vle8_v_u8mf2(v1840, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1842 = __riscv_vand_vx_u8mf2(v1841, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1843 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1842);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1844 = __riscv_vsrl_vx_u8mf2(v1841, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1844);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1846 = v21 + 88;
      const int8_t* v1847 = (const int8_t*) v1846;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1848 = *(const int8_t *)(v1847);
      const uint8_t* v1849 = v21 + 120;
      const int8_t* v1850 = (const int8_t*) v1849;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1851 = *(const int8_t *)(v1850);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1852 = __riscv_vwmacc_vx_i16m1(v1822, v1848, v1843, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1853 = __riscv_vwmacc_vx_i16m1(v1823, v1851, v1845, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1854 = v19 + 1104;
      const uint8_t* v1855 = (const uint8_t*) v1854;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1856 = __riscv_vle8_v_u8mf2(v1855, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1857 = __riscv_vand_vx_u8mf2(v1856, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1858 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1857);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1859 = __riscv_vsrl_vx_u8mf2(v1856, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1859);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1861 = v21 + 89;
      const int8_t* v1862 = (const int8_t*) v1861;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1863 = *(const int8_t *)(v1862);
      const uint8_t* v1864 = v21 + 121;
      const int8_t* v1865 = (const int8_t*) v1864;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1866 = *(const int8_t *)(v1865);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1867 = __riscv_vwmacc_vx_i16m1(v1837, v1863, v1858, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1868 = __riscv_vwmacc_vx_i16m1(v1838, v1866, v1860, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1869 = v19 + 1112;
      const uint8_t* v1870 = (const uint8_t*) v1869;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1871 = __riscv_vle8_v_u8mf2(v1870, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1872 = __riscv_vand_vx_u8mf2(v1871, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1873 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1872);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1874 = __riscv_vsrl_vx_u8mf2(v1871, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1875 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1874);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1876 = v21 + 89;
      const int8_t* v1877 = (const int8_t*) v1876;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1878 = *(const int8_t *)(v1877);
      const uint8_t* v1879 = v21 + 121;
      const int8_t* v1880 = (const int8_t*) v1879;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1881 = *(const int8_t *)(v1880);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1882 = __riscv_vwmacc_vx_i16m1(v1852, v1878, v1873, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1883 = __riscv_vwmacc_vx_i16m1(v1853, v1881, v1875, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1884 = v19 + 1120;
      const uint8_t* v1885 = (const uint8_t*) v1884;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1886 = __riscv_vle8_v_u8mf2(v1885, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1887 = __riscv_vand_vx_u8mf2(v1886, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1888 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1887);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1889 = __riscv_vsrl_vx_u8mf2(v1886, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1890 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1889);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1891 = v21 + 90;
      const int8_t* v1892 = (const int8_t*) v1891;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1893 = *(const int8_t *)(v1892);
      const uint8_t* v1894 = v21 + 122;
      const int8_t* v1895 = (const int8_t*) v1894;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1896 = *(const int8_t *)(v1895);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1897 = __riscv_vwmacc_vx_i16m1(v1867, v1893, v1888, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1898 = __riscv_vwmacc_vx_i16m1(v1868, v1896, v1890, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1899 = v19 + 1128;
      const uint8_t* v1900 = (const uint8_t*) v1899;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1901 = __riscv_vle8_v_u8mf2(v1900, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1902 = __riscv_vand_vx_u8mf2(v1901, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1903 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1902);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1904 = __riscv_vsrl_vx_u8mf2(v1901, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1905 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1904);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1906 = v21 + 90;
      const int8_t* v1907 = (const int8_t*) v1906;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1908 = *(const int8_t *)(v1907);
      const uint8_t* v1909 = v21 + 122;
      const int8_t* v1910 = (const int8_t*) v1909;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1911 = *(const int8_t *)(v1910);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1912 = __riscv_vwmacc_vx_i16m1(v1882, v1908, v1903, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1913 = __riscv_vwmacc_vx_i16m1(v1883, v1911, v1905, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1914 = v19 + 1136;
      const uint8_t* v1915 = (const uint8_t*) v1914;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1916 = __riscv_vle8_v_u8mf2(v1915, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1917 = __riscv_vand_vx_u8mf2(v1916, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1918 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1917);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1919 = __riscv_vsrl_vx_u8mf2(v1916, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1920 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1919);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1921 = v21 + 91;
      const int8_t* v1922 = (const int8_t*) v1921;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1923 = *(const int8_t *)(v1922);
      const uint8_t* v1924 = v21 + 123;
      const int8_t* v1925 = (const int8_t*) v1924;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1926 = *(const int8_t *)(v1925);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1927 = __riscv_vwmacc_vx_i16m1(v1897, v1923, v1918, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1928 = __riscv_vwmacc_vx_i16m1(v1898, v1926, v1920, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1929 = v19 + 1144;
      const uint8_t* v1930 = (const uint8_t*) v1929;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1931 = __riscv_vle8_v_u8mf2(v1930, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1932 = __riscv_vand_vx_u8mf2(v1931, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1933 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1932);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1934 = __riscv_vsrl_vx_u8mf2(v1931, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1935 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1934);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1936 = v21 + 91;
      const int8_t* v1937 = (const int8_t*) v1936;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1938 = *(const int8_t *)(v1937);
      const uint8_t* v1939 = v21 + 123;
      const int8_t* v1940 = (const int8_t*) v1939;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1941 = *(const int8_t *)(v1940);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1942 = __riscv_vwmacc_vx_i16m1(v1912, v1938, v1933, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1943 = __riscv_vwmacc_vx_i16m1(v1913, v1941, v1935, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1944 = v19 + 1152;
      const uint8_t* v1945 = (const uint8_t*) v1944;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1946 = __riscv_vle8_v_u8mf2(v1945, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1947 = __riscv_vand_vx_u8mf2(v1946, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1948 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1947);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1949 = __riscv_vsrl_vx_u8mf2(v1946, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1950 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1949);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1951 = v21 + 92;
      const int8_t* v1952 = (const int8_t*) v1951;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1953 = *(const int8_t *)(v1952);
      const uint8_t* v1954 = v21 + 124;
      const int8_t* v1955 = (const int8_t*) v1954;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1956 = *(const int8_t *)(v1955);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1957 = __riscv_vwmacc_vx_i16m1(v1927, v1953, v1948, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1958 = __riscv_vwmacc_vx_i16m1(v1928, v1956, v1950, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1959 = v19 + 1160;
      const uint8_t* v1960 = (const uint8_t*) v1959;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1961 = __riscv_vle8_v_u8mf2(v1960, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1962 = __riscv_vand_vx_u8mf2(v1961, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1963 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1962);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1964 = __riscv_vsrl_vx_u8mf2(v1961, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1965 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1964);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1966 = v21 + 92;
      const int8_t* v1967 = (const int8_t*) v1966;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1968 = *(const int8_t *)(v1967);
      const uint8_t* v1969 = v21 + 124;
      const int8_t* v1970 = (const int8_t*) v1969;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1971 = *(const int8_t *)(v1970);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1972 = __riscv_vwmacc_vx_i16m1(v1942, v1968, v1963, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1973 = __riscv_vwmacc_vx_i16m1(v1943, v1971, v1965, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1974 = v19 + 1168;
      const uint8_t* v1975 = (const uint8_t*) v1974;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1976 = __riscv_vle8_v_u8mf2(v1975, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1977 = __riscv_vand_vx_u8mf2(v1976, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1978 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1977);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1979 = __riscv_vsrl_vx_u8mf2(v1976, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1980 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1979);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1981 = v21 + 93;
      const int8_t* v1982 = (const int8_t*) v1981;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1983 = *(const int8_t *)(v1982);
      const uint8_t* v1984 = v21 + 125;
      const int8_t* v1985 = (const int8_t*) v1984;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1986 = *(const int8_t *)(v1985);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1987 = __riscv_vwmacc_vx_i16m1(v1957, v1983, v1978, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1988 = __riscv_vwmacc_vx_i16m1(v1958, v1986, v1980, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1989 = v19 + 1176;
      const uint8_t* v1990 = (const uint8_t*) v1989;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1991 = __riscv_vle8_v_u8mf2(v1990, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1992 = __riscv_vand_vx_u8mf2(v1991, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1993 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1992);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1994 = __riscv_vsrl_vx_u8mf2(v1991, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1995 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1994);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1996 = v21 + 93;
      const int8_t* v1997 = (const int8_t*) v1996;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1998 = *(const int8_t *)(v1997);
      const uint8_t* v1999 = v21 + 125;
      const int8_t* v2000 = (const int8_t*) v1999;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2001 = *(const int8_t *)(v2000);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2002 = __riscv_vwmacc_vx_i16m1(v1972, v1998, v1993, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2003 = __riscv_vwmacc_vx_i16m1(v1973, v2001, v1995, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2004 = v19 + 1184;
      const uint8_t* v2005 = (const uint8_t*) v2004;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2006 = __riscv_vle8_v_u8mf2(v2005, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2007 = __riscv_vand_vx_u8mf2(v2006, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2008 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2007);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2009 = __riscv_vsrl_vx_u8mf2(v2006, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2010 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2009);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2011 = v21 + 94;
      const int8_t* v2012 = (const int8_t*) v2011;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2013 = *(const int8_t *)(v2012);
      const uint8_t* v2014 = v21 + 126;
      const int8_t* v2015 = (const int8_t*) v2014;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2016 = *(const int8_t *)(v2015);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2017 = __riscv_vwmacc_vx_i16m1(v1987, v2013, v2008, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2018 = __riscv_vwmacc_vx_i16m1(v1988, v2016, v2010, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2019 = v19 + 1192;
      const uint8_t* v2020 = (const uint8_t*) v2019;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2021 = __riscv_vle8_v_u8mf2(v2020, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2022 = __riscv_vand_vx_u8mf2(v2021, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2023 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2022);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2024 = __riscv_vsrl_vx_u8mf2(v2021, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2025 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2024);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2026 = v21 + 94;
      const int8_t* v2027 = (const int8_t*) v2026;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2028 = *(const int8_t *)(v2027);
      const uint8_t* v2029 = v21 + 126;
      const int8_t* v2030 = (const int8_t*) v2029;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2031 = *(const int8_t *)(v2030);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2032 = __riscv_vwmacc_vx_i16m1(v2002, v2028, v2023, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2033 = __riscv_vwmacc_vx_i16m1(v2003, v2031, v2025, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2034 = v19 + 1200;
      const uint8_t* v2035 = (const uint8_t*) v2034;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2036 = __riscv_vle8_v_u8mf2(v2035, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2037 = __riscv_vand_vx_u8mf2(v2036, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2038 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2037);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2039 = __riscv_vsrl_vx_u8mf2(v2036, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2040 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2039);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2041 = v21 + 95;
      const int8_t* v2042 = (const int8_t*) v2041;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2043 = *(const int8_t *)(v2042);
      const uint8_t* v2044 = v21 + 127;
      const int8_t* v2045 = (const int8_t*) v2044;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2046 = *(const int8_t *)(v2045);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2047 = __riscv_vwmacc_vx_i16m1(v2017, v2043, v2038, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2048 = __riscv_vwmacc_vx_i16m1(v2018, v2046, v2040, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2049 = v19 + 1208;
      const uint8_t* v2050 = (const uint8_t*) v2049;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2051 = __riscv_vle8_v_u8mf2(v2050, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2052 = __riscv_vand_vx_u8mf2(v2051, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2053 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2052);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2054 = __riscv_vsrl_vx_u8mf2(v2051, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2055 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2054);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2056 = v21 + 95;
      const int8_t* v2057 = (const int8_t*) v2056;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2058 = *(const int8_t *)(v2057);
      const uint8_t* v2059 = v21 + 127;
      const int8_t* v2060 = (const int8_t*) v2059;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2061 = *(const int8_t *)(v2060);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2062 = __riscv_vwmacc_vx_i16m1(v2032, v2058, v2053, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2063 = __riscv_vwmacc_vx_i16m1(v2033, v2061, v2055, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2064 = v19 + 1216;
      const uint8_t* v2065 = (const uint8_t*) v2064;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2066 = __riscv_vle8_v_u8mf2(v2065, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2067 = __riscv_vand_vx_u8mf2(v2066, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2068 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2067);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2069 = __riscv_vsrl_vx_u8mf2(v2066, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2070 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2069);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2071 = v21 + 96;
      const int8_t* v2072 = (const int8_t*) v2071;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2073 = *(const int8_t *)(v2072);
      const uint8_t* v2074 = v21 + 128;
      const int8_t* v2075 = (const int8_t*) v2074;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2076 = *(const int8_t *)(v2075);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2077 = __riscv_vwmacc_vx_i16m1(v2047, v2073, v2068, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2078 = __riscv_vwmacc_vx_i16m1(v2048, v2076, v2070, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2079 = v19 + 1224;
      const uint8_t* v2080 = (const uint8_t*) v2079;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2081 = __riscv_vle8_v_u8mf2(v2080, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2082 = __riscv_vand_vx_u8mf2(v2081, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2083 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2082);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2084 = __riscv_vsrl_vx_u8mf2(v2081, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2085 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2084);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2086 = v21 + 96;
      const int8_t* v2087 = (const int8_t*) v2086;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2088 = *(const int8_t *)(v2087);
      const uint8_t* v2089 = v21 + 128;
      const int8_t* v2090 = (const int8_t*) v2089;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2091 = *(const int8_t *)(v2090);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2092 = __riscv_vwmacc_vx_i16m1(v2062, v2088, v2083, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2093 = __riscv_vwmacc_vx_i16m1(v2063, v2091, v2085, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2094 = v19 + 1232;
      const uint8_t* v2095 = (const uint8_t*) v2094;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2096 = __riscv_vle8_v_u8mf2(v2095, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2097 = __riscv_vand_vx_u8mf2(v2096, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2098 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2097);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2099 = __riscv_vsrl_vx_u8mf2(v2096, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2100 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2099);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2101 = v21 + 97;
      const int8_t* v2102 = (const int8_t*) v2101;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2103 = *(const int8_t *)(v2102);
      const uint8_t* v2104 = v21 + 129;
      const int8_t* v2105 = (const int8_t*) v2104;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2106 = *(const int8_t *)(v2105);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2107 = __riscv_vwmacc_vx_i16m1(v2077, v2103, v2098, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2108 = __riscv_vwmacc_vx_i16m1(v2078, v2106, v2100, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2109 = v19 + 1240;
      const uint8_t* v2110 = (const uint8_t*) v2109;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2111 = __riscv_vle8_v_u8mf2(v2110, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2112 = __riscv_vand_vx_u8mf2(v2111, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2113 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2112);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2114 = __riscv_vsrl_vx_u8mf2(v2111, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2115 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2114);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2116 = v21 + 97;
      const int8_t* v2117 = (const int8_t*) v2116;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2118 = *(const int8_t *)(v2117);
      const uint8_t* v2119 = v21 + 129;
      const int8_t* v2120 = (const int8_t*) v2119;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2121 = *(const int8_t *)(v2120);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2122 = __riscv_vwmacc_vx_i16m1(v2092, v2118, v2113, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2123 = __riscv_vwmacc_vx_i16m1(v2093, v2121, v2115, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2124 = v19 + 1248;
      const uint8_t* v2125 = (const uint8_t*) v2124;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2126 = __riscv_vle8_v_u8mf2(v2125, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2127 = __riscv_vand_vx_u8mf2(v2126, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2128 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2127);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2129 = __riscv_vsrl_vx_u8mf2(v2126, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2130 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2129);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2131 = v21 + 98;
      const int8_t* v2132 = (const int8_t*) v2131;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2133 = *(const int8_t *)(v2132);
      const uint8_t* v2134 = v21 + 130;
      const int8_t* v2135 = (const int8_t*) v2134;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2136 = *(const int8_t *)(v2135);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2137 = __riscv_vwmacc_vx_i16m1(v2107, v2133, v2128, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2138 = __riscv_vwmacc_vx_i16m1(v2108, v2136, v2130, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2139 = v19 + 1256;
      const uint8_t* v2140 = (const uint8_t*) v2139;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2141 = __riscv_vle8_v_u8mf2(v2140, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2142 = __riscv_vand_vx_u8mf2(v2141, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2143 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2142);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2144 = __riscv_vsrl_vx_u8mf2(v2141, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2145 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2144);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2146 = v21 + 98;
      const int8_t* v2147 = (const int8_t*) v2146;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2148 = *(const int8_t *)(v2147);
      const uint8_t* v2149 = v21 + 130;
      const int8_t* v2150 = (const int8_t*) v2149;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2151 = *(const int8_t *)(v2150);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2152 = __riscv_vwmacc_vx_i16m1(v2122, v2148, v2143, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2153 = __riscv_vwmacc_vx_i16m1(v2123, v2151, v2145, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2154 = v19 + 1264;
      const uint8_t* v2155 = (const uint8_t*) v2154;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2156 = __riscv_vle8_v_u8mf2(v2155, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2157 = __riscv_vand_vx_u8mf2(v2156, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2158 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2157);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2159 = __riscv_vsrl_vx_u8mf2(v2156, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2160 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2159);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2161 = v21 + 99;
      const int8_t* v2162 = (const int8_t*) v2161;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2163 = *(const int8_t *)(v2162);
      const uint8_t* v2164 = v21 + 131;
      const int8_t* v2165 = (const int8_t*) v2164;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2166 = *(const int8_t *)(v2165);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2167 = __riscv_vwmacc_vx_i16m1(v2137, v2163, v2158, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2168 = __riscv_vwmacc_vx_i16m1(v2138, v2166, v2160, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2169 = v19 + 1272;
      const uint8_t* v2170 = (const uint8_t*) v2169;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2171 = __riscv_vle8_v_u8mf2(v2170, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2172 = __riscv_vand_vx_u8mf2(v2171, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2173 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2172);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2174 = __riscv_vsrl_vx_u8mf2(v2171, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2175 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2174);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2176 = v21 + 99;
      const int8_t* v2177 = (const int8_t*) v2176;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2178 = *(const int8_t *)(v2177);
      const uint8_t* v2179 = v21 + 131;
      const int8_t* v2180 = (const int8_t*) v2179;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2181 = *(const int8_t *)(v2180);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2182 = __riscv_vwmacc_vx_i16m1(v2152, v2178, v2173, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2183 = __riscv_vwmacc_vx_i16m1(v2153, v2181, v2175, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v2184 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2185 = __riscv_vwmacc_vv_i32m2(v2184, v93, v2167, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2186 = __riscv_vwmacc_vv_i32m2(v2185, v111, v2168, 8);
      v34 = v2186;
      vint32m2_t v2187 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2188 = __riscv_vwmacc_vv_i32m2(v2187, v165, v2182, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2189 = __riscv_vwmacc_vv_i32m2(v2188, v183, v2183, 8);
      v38 = v2189;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v2190 = v19 + 128;
      const uint8_t* v2191 = (const uint8_t*) v2190;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2192 = __riscv_vle8_v_u8mf2(v2191, 8);
      const uint8_t* v2193 = v19 + 192;
      const uint8_t* v2194 = (const uint8_t*) v2193;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2195 = __riscv_vle8_v_u8mf2(v2194, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2196 = __riscv_vand_vx_u8mf2(v2192, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2197 = __riscv_vsrl_vx_u8mf2(v2192, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2198 = __riscv_vand_vx_u8mf2(v2195, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2199 = __riscv_vand_vx_u8mf2(v2195, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2200 = __riscv_vsrl_vx_u8mf2(v2199, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2201 = __riscv_vor_vv_u8mf2(v2198, v2196, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2202 = __riscv_vor_vv_u8mf2(v2200, v2197, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2203 = __riscv_vzext_vf2_u16m1(v2201, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2204 = __riscv_vreinterpret_v_u16m1_i16m1(v2203);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2205 = __riscv_vzext_vf2_u16m1(v2202, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2206 = __riscv_vreinterpret_v_u16m1_i16m1(v2205);
      const uint8_t* v2207 = v19 + 144;
      const uint8_t* v2208 = (const uint8_t*) v2207;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2209 = __riscv_vle8_v_u8mf2(v2208, 8);
      const uint8_t* v2210 = v19 + 208;
      const uint8_t* v2211 = (const uint8_t*) v2210;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2212 = __riscv_vle8_v_u8mf2(v2211, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2213 = __riscv_vand_vx_u8mf2(v2209, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2214 = __riscv_vsrl_vx_u8mf2(v2209, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2215 = __riscv_vand_vx_u8mf2(v2212, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2216 = __riscv_vand_vx_u8mf2(v2212, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2217 = __riscv_vsrl_vx_u8mf2(v2216, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2218 = __riscv_vor_vv_u8mf2(v2215, v2213, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2219 = __riscv_vor_vv_u8mf2(v2217, v2214, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2220 = __riscv_vzext_vf2_u16m1(v2218, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2221 = __riscv_vreinterpret_v_u16m1_i16m1(v2220);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2222 = __riscv_vzext_vf2_u16m1(v2219, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2223 = __riscv_vreinterpret_v_u16m1_i16m1(v2222);
      const uint8_t* v2224 = v19 + 160;
      const uint8_t* v2225 = (const uint8_t*) v2224;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2226 = __riscv_vle8_v_u8mf2(v2225, 8);
      const uint8_t* v2227 = v19 + 224;
      const uint8_t* v2228 = (const uint8_t*) v2227;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2229 = __riscv_vle8_v_u8mf2(v2228, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2230 = __riscv_vand_vx_u8mf2(v2226, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2231 = __riscv_vsrl_vx_u8mf2(v2226, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2232 = __riscv_vand_vx_u8mf2(v2229, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2233 = __riscv_vand_vx_u8mf2(v2229, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2234 = __riscv_vsrl_vx_u8mf2(v2233, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2235 = __riscv_vor_vv_u8mf2(v2232, v2230, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2236 = __riscv_vor_vv_u8mf2(v2234, v2231, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2237 = __riscv_vzext_vf2_u16m1(v2235, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2238 = __riscv_vreinterpret_v_u16m1_i16m1(v2237);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2239 = __riscv_vzext_vf2_u16m1(v2236, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2240 = __riscv_vreinterpret_v_u16m1_i16m1(v2239);
      const uint8_t* v2241 = v19 + 176;
      const uint8_t* v2242 = (const uint8_t*) v2241;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2243 = __riscv_vle8_v_u8mf2(v2242, 8);
      const uint8_t* v2244 = v19 + 240;
      const uint8_t* v2245 = (const uint8_t*) v2244;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2246 = __riscv_vle8_v_u8mf2(v2245, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2247 = __riscv_vand_vx_u8mf2(v2243, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2248 = __riscv_vsrl_vx_u8mf2(v2243, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2249 = __riscv_vand_vx_u8mf2(v2246, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2250 = __riscv_vand_vx_u8mf2(v2246, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2251 = __riscv_vsrl_vx_u8mf2(v2250, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2252 = __riscv_vor_vv_u8mf2(v2249, v2247, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2253 = __riscv_vor_vv_u8mf2(v2251, v2248, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2254 = __riscv_vzext_vf2_u16m1(v2252, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2255 = __riscv_vreinterpret_v_u16m1_i16m1(v2254);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2256 = __riscv_vzext_vf2_u16m1(v2253, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2257 = __riscv_vreinterpret_v_u16m1_i16m1(v2256);
      const uint8_t* v2258 = v19 + 136;
      const uint8_t* v2259 = (const uint8_t*) v2258;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2260 = __riscv_vle8_v_u8mf2(v2259, 8);
      const uint8_t* v2261 = v19 + 200;
      const uint8_t* v2262 = (const uint8_t*) v2261;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2263 = __riscv_vle8_v_u8mf2(v2262, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2264 = __riscv_vand_vx_u8mf2(v2260, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2265 = __riscv_vsrl_vx_u8mf2(v2260, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2266 = __riscv_vand_vx_u8mf2(v2263, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2267 = __riscv_vand_vx_u8mf2(v2263, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2268 = __riscv_vsrl_vx_u8mf2(v2267, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2269 = __riscv_vor_vv_u8mf2(v2266, v2264, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2270 = __riscv_vor_vv_u8mf2(v2268, v2265, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2271 = __riscv_vzext_vf2_u16m1(v2269, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2272 = __riscv_vreinterpret_v_u16m1_i16m1(v2271);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2273 = __riscv_vzext_vf2_u16m1(v2270, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2274 = __riscv_vreinterpret_v_u16m1_i16m1(v2273);
      const uint8_t* v2275 = v19 + 152;
      const uint8_t* v2276 = (const uint8_t*) v2275;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2277 = __riscv_vle8_v_u8mf2(v2276, 8);
      const uint8_t* v2278 = v19 + 216;
      const uint8_t* v2279 = (const uint8_t*) v2278;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2280 = __riscv_vle8_v_u8mf2(v2279, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2281 = __riscv_vand_vx_u8mf2(v2277, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2282 = __riscv_vsrl_vx_u8mf2(v2277, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2283 = __riscv_vand_vx_u8mf2(v2280, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2284 = __riscv_vand_vx_u8mf2(v2280, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2285 = __riscv_vsrl_vx_u8mf2(v2284, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2286 = __riscv_vor_vv_u8mf2(v2283, v2281, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2287 = __riscv_vor_vv_u8mf2(v2285, v2282, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2288 = __riscv_vzext_vf2_u16m1(v2286, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2289 = __riscv_vreinterpret_v_u16m1_i16m1(v2288);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2290 = __riscv_vzext_vf2_u16m1(v2287, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2291 = __riscv_vreinterpret_v_u16m1_i16m1(v2290);
      const uint8_t* v2292 = v19 + 168;
      const uint8_t* v2293 = (const uint8_t*) v2292;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2294 = __riscv_vle8_v_u8mf2(v2293, 8);
      const uint8_t* v2295 = v19 + 232;
      const uint8_t* v2296 = (const uint8_t*) v2295;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2297 = __riscv_vle8_v_u8mf2(v2296, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2298 = __riscv_vand_vx_u8mf2(v2294, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2299 = __riscv_vsrl_vx_u8mf2(v2294, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2300 = __riscv_vand_vx_u8mf2(v2297, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2301 = __riscv_vand_vx_u8mf2(v2297, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2302 = __riscv_vsrl_vx_u8mf2(v2301, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2303 = __riscv_vor_vv_u8mf2(v2300, v2298, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2304 = __riscv_vor_vv_u8mf2(v2302, v2299, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2305 = __riscv_vzext_vf2_u16m1(v2303, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2306 = __riscv_vreinterpret_v_u16m1_i16m1(v2305);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2307 = __riscv_vzext_vf2_u16m1(v2304, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2308 = __riscv_vreinterpret_v_u16m1_i16m1(v2307);
      const uint8_t* v2309 = v19 + 184;
      const uint8_t* v2310 = (const uint8_t*) v2309;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2311 = __riscv_vle8_v_u8mf2(v2310, 8);
      const uint8_t* v2312 = v19 + 248;
      const uint8_t* v2313 = (const uint8_t*) v2312;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2314 = __riscv_vle8_v_u8mf2(v2313, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2315 = __riscv_vand_vx_u8mf2(v2311, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2316 = __riscv_vsrl_vx_u8mf2(v2311, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2317 = __riscv_vand_vx_u8mf2(v2314, 0x30, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2318 = __riscv_vand_vx_u8mf2(v2314, 0xC0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2319 = __riscv_vsrl_vx_u8mf2(v2318, 2, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2320 = __riscv_vor_vv_u8mf2(v2317, v2315, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v2321 = __riscv_vor_vv_u8mf2(v2319, v2316, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2322 = __riscv_vzext_vf2_u16m1(v2320, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2323 = __riscv_vreinterpret_v_u16m1_i16m1(v2322);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v2324 = __riscv_vzext_vf2_u16m1(v2321, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v2325 = __riscv_vreinterpret_v_u16m1_i16m1(v2324);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v2326 = v21 + 276;
      const int16_t* v2327 = (const int16_t*) v2326;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2328 = *(const int16_t *)(v2327);
      const uint8_t* v2329 = v21 + 278;
      const int16_t* v2330 = (const int16_t*) v2329;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2331 = *(const int16_t *)(v2330);
      int32_t v2332 = v2328 + v2331;
      vint32m2_t v2333 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2334 = __riscv_vwmacc_vx_i32m2(v2333, v2332, v2206, 8);
      v36 = v2334;
      vint32m2_t v2335 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2336 = __riscv_vwmacc_vx_i32m2(v2335, v2332, v2274, 8);
      v40 = v2336;
      const uint8_t* v2337 = v21 + 280;
      const int16_t* v2338 = (const int16_t*) v2337;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2339 = *(const int16_t *)(v2338);
      const uint8_t* v2340 = v21 + 282;
      const int16_t* v2341 = (const int16_t*) v2340;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2342 = *(const int16_t *)(v2341);
      int32_t v2343 = v2339 + v2342;
      vint32m2_t v2344 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2345 = __riscv_vwmacc_vx_i32m2(v2344, v2343, v2223, 8);
      v36 = v2345;
      vint32m2_t v2346 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2347 = __riscv_vwmacc_vx_i32m2(v2346, v2343, v2291, 8);
      v40 = v2347;
      const uint8_t* v2348 = v21 + 284;
      const int16_t* v2349 = (const int16_t*) v2348;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2350 = *(const int16_t *)(v2349);
      const uint8_t* v2351 = v21 + 286;
      const int16_t* v2352 = (const int16_t*) v2351;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2353 = *(const int16_t *)(v2352);
      int32_t v2354 = v2350 + v2353;
      vint32m2_t v2355 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2356 = __riscv_vwmacc_vx_i32m2(v2355, v2354, v2240, 8);
      v36 = v2356;
      vint32m2_t v2357 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2358 = __riscv_vwmacc_vx_i32m2(v2357, v2354, v2308, 8);
      v40 = v2358;
      const uint8_t* v2359 = v21 + 288;
      const int16_t* v2360 = (const int16_t*) v2359;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2361 = *(const int16_t *)(v2360);
      const uint8_t* v2362 = v21 + 290;
      const int16_t* v2363 = (const int16_t*) v2362;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v2364 = *(const int16_t *)(v2363);
      int32_t v2365 = v2361 + v2364;
      vint32m2_t v2366 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2367 = __riscv_vwmacc_vx_i32m2(v2366, v2365, v2257, 8);
      v36 = v2367;
      vint32m2_t v2368 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v2369 = __riscv_vwmacc_vx_i32m2(v2368, v2365, v2325, 8);
      v40 = v2369;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2370 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2371 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2372 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2373 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2374 = v19 + 1280;
      const uint8_t* v2375 = (const uint8_t*) v2374;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2376 = __riscv_vle8_v_u8mf2(v2375, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2377 = __riscv_vand_vx_u8mf2(v2376, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2378 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2377);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2379 = __riscv_vsrl_vx_u8mf2(v2376, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2380 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2379);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2381 = v21 + 132;
      const int8_t* v2382 = (const int8_t*) v2381;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2383 = *(const int8_t *)(v2382);
      const uint8_t* v2384 = v21 + 164;
      const int8_t* v2385 = (const int8_t*) v2384;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2386 = *(const int8_t *)(v2385);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2387 = __riscv_vwmacc_vx_i16m1(v2370, v2383, v2378, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2388 = __riscv_vwmacc_vx_i16m1(v2371, v2386, v2380, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2389 = v19 + 1288;
      const uint8_t* v2390 = (const uint8_t*) v2389;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2391 = __riscv_vle8_v_u8mf2(v2390, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2392 = __riscv_vand_vx_u8mf2(v2391, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2393 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2392);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2394 = __riscv_vsrl_vx_u8mf2(v2391, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2395 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2394);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2396 = v21 + 132;
      const int8_t* v2397 = (const int8_t*) v2396;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2398 = *(const int8_t *)(v2397);
      const uint8_t* v2399 = v21 + 164;
      const int8_t* v2400 = (const int8_t*) v2399;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2401 = *(const int8_t *)(v2400);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2402 = __riscv_vwmacc_vx_i16m1(v2372, v2398, v2393, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2403 = __riscv_vwmacc_vx_i16m1(v2373, v2401, v2395, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2404 = v19 + 1296;
      const uint8_t* v2405 = (const uint8_t*) v2404;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2406 = __riscv_vle8_v_u8mf2(v2405, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2407 = __riscv_vand_vx_u8mf2(v2406, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2408 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2407);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2409 = __riscv_vsrl_vx_u8mf2(v2406, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2410 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2409);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2411 = v21 + 133;
      const int8_t* v2412 = (const int8_t*) v2411;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2413 = *(const int8_t *)(v2412);
      const uint8_t* v2414 = v21 + 165;
      const int8_t* v2415 = (const int8_t*) v2414;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2416 = *(const int8_t *)(v2415);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2417 = __riscv_vwmacc_vx_i16m1(v2387, v2413, v2408, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2418 = __riscv_vwmacc_vx_i16m1(v2388, v2416, v2410, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2419 = v19 + 1304;
      const uint8_t* v2420 = (const uint8_t*) v2419;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2421 = __riscv_vle8_v_u8mf2(v2420, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2422 = __riscv_vand_vx_u8mf2(v2421, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2423 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2422);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2424 = __riscv_vsrl_vx_u8mf2(v2421, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2425 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2424);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2426 = v21 + 133;
      const int8_t* v2427 = (const int8_t*) v2426;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2428 = *(const int8_t *)(v2427);
      const uint8_t* v2429 = v21 + 165;
      const int8_t* v2430 = (const int8_t*) v2429;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2431 = *(const int8_t *)(v2430);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2432 = __riscv_vwmacc_vx_i16m1(v2402, v2428, v2423, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2433 = __riscv_vwmacc_vx_i16m1(v2403, v2431, v2425, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2434 = v19 + 1312;
      const uint8_t* v2435 = (const uint8_t*) v2434;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2436 = __riscv_vle8_v_u8mf2(v2435, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2437 = __riscv_vand_vx_u8mf2(v2436, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2438 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2437);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2439 = __riscv_vsrl_vx_u8mf2(v2436, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2440 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2439);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2441 = v21 + 134;
      const int8_t* v2442 = (const int8_t*) v2441;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2443 = *(const int8_t *)(v2442);
      const uint8_t* v2444 = v21 + 166;
      const int8_t* v2445 = (const int8_t*) v2444;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2446 = *(const int8_t *)(v2445);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2447 = __riscv_vwmacc_vx_i16m1(v2417, v2443, v2438, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2448 = __riscv_vwmacc_vx_i16m1(v2418, v2446, v2440, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2449 = v19 + 1320;
      const uint8_t* v2450 = (const uint8_t*) v2449;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2451 = __riscv_vle8_v_u8mf2(v2450, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2452 = __riscv_vand_vx_u8mf2(v2451, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2453 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2452);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2454 = __riscv_vsrl_vx_u8mf2(v2451, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2455 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2454);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2456 = v21 + 134;
      const int8_t* v2457 = (const int8_t*) v2456;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2458 = *(const int8_t *)(v2457);
      const uint8_t* v2459 = v21 + 166;
      const int8_t* v2460 = (const int8_t*) v2459;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2461 = *(const int8_t *)(v2460);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2462 = __riscv_vwmacc_vx_i16m1(v2432, v2458, v2453, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2463 = __riscv_vwmacc_vx_i16m1(v2433, v2461, v2455, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2464 = v19 + 1328;
      const uint8_t* v2465 = (const uint8_t*) v2464;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2466 = __riscv_vle8_v_u8mf2(v2465, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2467 = __riscv_vand_vx_u8mf2(v2466, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2468 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2467);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2469 = __riscv_vsrl_vx_u8mf2(v2466, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2470 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2469);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2471 = v21 + 135;
      const int8_t* v2472 = (const int8_t*) v2471;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2473 = *(const int8_t *)(v2472);
      const uint8_t* v2474 = v21 + 167;
      const int8_t* v2475 = (const int8_t*) v2474;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2476 = *(const int8_t *)(v2475);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2477 = __riscv_vwmacc_vx_i16m1(v2447, v2473, v2468, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2478 = __riscv_vwmacc_vx_i16m1(v2448, v2476, v2470, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2479 = v19 + 1336;
      const uint8_t* v2480 = (const uint8_t*) v2479;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2481 = __riscv_vle8_v_u8mf2(v2480, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2482 = __riscv_vand_vx_u8mf2(v2481, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2483 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2482);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2484 = __riscv_vsrl_vx_u8mf2(v2481, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2485 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2484);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2486 = v21 + 135;
      const int8_t* v2487 = (const int8_t*) v2486;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2488 = *(const int8_t *)(v2487);
      const uint8_t* v2489 = v21 + 167;
      const int8_t* v2490 = (const int8_t*) v2489;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2491 = *(const int8_t *)(v2490);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2492 = __riscv_vwmacc_vx_i16m1(v2462, v2488, v2483, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2493 = __riscv_vwmacc_vx_i16m1(v2463, v2491, v2485, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2494 = v19 + 1344;
      const uint8_t* v2495 = (const uint8_t*) v2494;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2496 = __riscv_vle8_v_u8mf2(v2495, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2497 = __riscv_vand_vx_u8mf2(v2496, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2498 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2497);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2499 = __riscv_vsrl_vx_u8mf2(v2496, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2500 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2499);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2501 = v21 + 136;
      const int8_t* v2502 = (const int8_t*) v2501;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2503 = *(const int8_t *)(v2502);
      const uint8_t* v2504 = v21 + 168;
      const int8_t* v2505 = (const int8_t*) v2504;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2506 = *(const int8_t *)(v2505);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2507 = __riscv_vwmacc_vx_i16m1(v2477, v2503, v2498, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2508 = __riscv_vwmacc_vx_i16m1(v2478, v2506, v2500, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2509 = v19 + 1352;
      const uint8_t* v2510 = (const uint8_t*) v2509;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2511 = __riscv_vle8_v_u8mf2(v2510, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2512 = __riscv_vand_vx_u8mf2(v2511, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2513 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2512);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2514 = __riscv_vsrl_vx_u8mf2(v2511, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2515 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2514);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2516 = v21 + 136;
      const int8_t* v2517 = (const int8_t*) v2516;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2518 = *(const int8_t *)(v2517);
      const uint8_t* v2519 = v21 + 168;
      const int8_t* v2520 = (const int8_t*) v2519;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2521 = *(const int8_t *)(v2520);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2522 = __riscv_vwmacc_vx_i16m1(v2492, v2518, v2513, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2523 = __riscv_vwmacc_vx_i16m1(v2493, v2521, v2515, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2524 = v19 + 1360;
      const uint8_t* v2525 = (const uint8_t*) v2524;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2526 = __riscv_vle8_v_u8mf2(v2525, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2527 = __riscv_vand_vx_u8mf2(v2526, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2528 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2527);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2529 = __riscv_vsrl_vx_u8mf2(v2526, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2530 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2529);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2531 = v21 + 137;
      const int8_t* v2532 = (const int8_t*) v2531;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2533 = *(const int8_t *)(v2532);
      const uint8_t* v2534 = v21 + 169;
      const int8_t* v2535 = (const int8_t*) v2534;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2536 = *(const int8_t *)(v2535);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2537 = __riscv_vwmacc_vx_i16m1(v2507, v2533, v2528, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2538 = __riscv_vwmacc_vx_i16m1(v2508, v2536, v2530, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2539 = v19 + 1368;
      const uint8_t* v2540 = (const uint8_t*) v2539;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2541 = __riscv_vle8_v_u8mf2(v2540, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2542 = __riscv_vand_vx_u8mf2(v2541, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2543 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2542);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2544 = __riscv_vsrl_vx_u8mf2(v2541, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2545 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2544);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2546 = v21 + 137;
      const int8_t* v2547 = (const int8_t*) v2546;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2548 = *(const int8_t *)(v2547);
      const uint8_t* v2549 = v21 + 169;
      const int8_t* v2550 = (const int8_t*) v2549;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2551 = *(const int8_t *)(v2550);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2552 = __riscv_vwmacc_vx_i16m1(v2522, v2548, v2543, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2553 = __riscv_vwmacc_vx_i16m1(v2523, v2551, v2545, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2554 = v19 + 1376;
      const uint8_t* v2555 = (const uint8_t*) v2554;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2556 = __riscv_vle8_v_u8mf2(v2555, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2557 = __riscv_vand_vx_u8mf2(v2556, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2558 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2557);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2559 = __riscv_vsrl_vx_u8mf2(v2556, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2560 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2559);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2561 = v21 + 138;
      const int8_t* v2562 = (const int8_t*) v2561;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2563 = *(const int8_t *)(v2562);
      const uint8_t* v2564 = v21 + 170;
      const int8_t* v2565 = (const int8_t*) v2564;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2566 = *(const int8_t *)(v2565);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2567 = __riscv_vwmacc_vx_i16m1(v2537, v2563, v2558, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2568 = __riscv_vwmacc_vx_i16m1(v2538, v2566, v2560, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2569 = v19 + 1384;
      const uint8_t* v2570 = (const uint8_t*) v2569;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2571 = __riscv_vle8_v_u8mf2(v2570, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2572 = __riscv_vand_vx_u8mf2(v2571, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2573 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2572);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2574 = __riscv_vsrl_vx_u8mf2(v2571, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2575 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2574);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2576 = v21 + 138;
      const int8_t* v2577 = (const int8_t*) v2576;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2578 = *(const int8_t *)(v2577);
      const uint8_t* v2579 = v21 + 170;
      const int8_t* v2580 = (const int8_t*) v2579;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2581 = *(const int8_t *)(v2580);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2582 = __riscv_vwmacc_vx_i16m1(v2552, v2578, v2573, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2583 = __riscv_vwmacc_vx_i16m1(v2553, v2581, v2575, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2584 = v19 + 1392;
      const uint8_t* v2585 = (const uint8_t*) v2584;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2586 = __riscv_vle8_v_u8mf2(v2585, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2587 = __riscv_vand_vx_u8mf2(v2586, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2588 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2587);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2589 = __riscv_vsrl_vx_u8mf2(v2586, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2590 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2589);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2591 = v21 + 139;
      const int8_t* v2592 = (const int8_t*) v2591;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2593 = *(const int8_t *)(v2592);
      const uint8_t* v2594 = v21 + 171;
      const int8_t* v2595 = (const int8_t*) v2594;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2596 = *(const int8_t *)(v2595);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2597 = __riscv_vwmacc_vx_i16m1(v2567, v2593, v2588, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2598 = __riscv_vwmacc_vx_i16m1(v2568, v2596, v2590, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2599 = v19 + 1400;
      const uint8_t* v2600 = (const uint8_t*) v2599;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2601 = __riscv_vle8_v_u8mf2(v2600, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2602 = __riscv_vand_vx_u8mf2(v2601, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2603 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2602);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2604 = __riscv_vsrl_vx_u8mf2(v2601, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2605 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2604);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2606 = v21 + 139;
      const int8_t* v2607 = (const int8_t*) v2606;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2608 = *(const int8_t *)(v2607);
      const uint8_t* v2609 = v21 + 171;
      const int8_t* v2610 = (const int8_t*) v2609;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2611 = *(const int8_t *)(v2610);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2612 = __riscv_vwmacc_vx_i16m1(v2582, v2608, v2603, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2613 = __riscv_vwmacc_vx_i16m1(v2583, v2611, v2605, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2614 = v19 + 1408;
      const uint8_t* v2615 = (const uint8_t*) v2614;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2616 = __riscv_vle8_v_u8mf2(v2615, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2617 = __riscv_vand_vx_u8mf2(v2616, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2618 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2617);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2619 = __riscv_vsrl_vx_u8mf2(v2616, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2620 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2619);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2621 = v21 + 140;
      const int8_t* v2622 = (const int8_t*) v2621;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2623 = *(const int8_t *)(v2622);
      const uint8_t* v2624 = v21 + 172;
      const int8_t* v2625 = (const int8_t*) v2624;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2626 = *(const int8_t *)(v2625);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2627 = __riscv_vwmacc_vx_i16m1(v2597, v2623, v2618, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2628 = __riscv_vwmacc_vx_i16m1(v2598, v2626, v2620, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2629 = v19 + 1416;
      const uint8_t* v2630 = (const uint8_t*) v2629;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2631 = __riscv_vle8_v_u8mf2(v2630, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2632 = __riscv_vand_vx_u8mf2(v2631, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2633 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2632);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2634 = __riscv_vsrl_vx_u8mf2(v2631, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2635 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2634);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2636 = v21 + 140;
      const int8_t* v2637 = (const int8_t*) v2636;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2638 = *(const int8_t *)(v2637);
      const uint8_t* v2639 = v21 + 172;
      const int8_t* v2640 = (const int8_t*) v2639;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2641 = *(const int8_t *)(v2640);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2642 = __riscv_vwmacc_vx_i16m1(v2612, v2638, v2633, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2643 = __riscv_vwmacc_vx_i16m1(v2613, v2641, v2635, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2644 = v19 + 1424;
      const uint8_t* v2645 = (const uint8_t*) v2644;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2646 = __riscv_vle8_v_u8mf2(v2645, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2647 = __riscv_vand_vx_u8mf2(v2646, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2648 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2647);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2649 = __riscv_vsrl_vx_u8mf2(v2646, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2650 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2649);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2651 = v21 + 141;
      const int8_t* v2652 = (const int8_t*) v2651;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2653 = *(const int8_t *)(v2652);
      const uint8_t* v2654 = v21 + 173;
      const int8_t* v2655 = (const int8_t*) v2654;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2656 = *(const int8_t *)(v2655);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2657 = __riscv_vwmacc_vx_i16m1(v2627, v2653, v2648, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2658 = __riscv_vwmacc_vx_i16m1(v2628, v2656, v2650, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2659 = v19 + 1432;
      const uint8_t* v2660 = (const uint8_t*) v2659;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2661 = __riscv_vle8_v_u8mf2(v2660, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2662 = __riscv_vand_vx_u8mf2(v2661, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2663 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2662);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2664 = __riscv_vsrl_vx_u8mf2(v2661, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2665 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2664);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2666 = v21 + 141;
      const int8_t* v2667 = (const int8_t*) v2666;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2668 = *(const int8_t *)(v2667);
      const uint8_t* v2669 = v21 + 173;
      const int8_t* v2670 = (const int8_t*) v2669;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2671 = *(const int8_t *)(v2670);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2672 = __riscv_vwmacc_vx_i16m1(v2642, v2668, v2663, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2673 = __riscv_vwmacc_vx_i16m1(v2643, v2671, v2665, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2674 = v19 + 1440;
      const uint8_t* v2675 = (const uint8_t*) v2674;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2676 = __riscv_vle8_v_u8mf2(v2675, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2677 = __riscv_vand_vx_u8mf2(v2676, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2678 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2677);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2679 = __riscv_vsrl_vx_u8mf2(v2676, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2680 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2679);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2681 = v21 + 142;
      const int8_t* v2682 = (const int8_t*) v2681;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2683 = *(const int8_t *)(v2682);
      const uint8_t* v2684 = v21 + 174;
      const int8_t* v2685 = (const int8_t*) v2684;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2686 = *(const int8_t *)(v2685);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2687 = __riscv_vwmacc_vx_i16m1(v2657, v2683, v2678, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2688 = __riscv_vwmacc_vx_i16m1(v2658, v2686, v2680, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2689 = v19 + 1448;
      const uint8_t* v2690 = (const uint8_t*) v2689;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2691 = __riscv_vle8_v_u8mf2(v2690, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2692 = __riscv_vand_vx_u8mf2(v2691, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2693 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2692);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2694 = __riscv_vsrl_vx_u8mf2(v2691, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2695 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2694);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2696 = v21 + 142;
      const int8_t* v2697 = (const int8_t*) v2696;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2698 = *(const int8_t *)(v2697);
      const uint8_t* v2699 = v21 + 174;
      const int8_t* v2700 = (const int8_t*) v2699;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2701 = *(const int8_t *)(v2700);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2702 = __riscv_vwmacc_vx_i16m1(v2672, v2698, v2693, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2703 = __riscv_vwmacc_vx_i16m1(v2673, v2701, v2695, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2704 = v19 + 1456;
      const uint8_t* v2705 = (const uint8_t*) v2704;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2706 = __riscv_vle8_v_u8mf2(v2705, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2707 = __riscv_vand_vx_u8mf2(v2706, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2708 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2707);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2709 = __riscv_vsrl_vx_u8mf2(v2706, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2710 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2709);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2711 = v21 + 143;
      const int8_t* v2712 = (const int8_t*) v2711;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2713 = *(const int8_t *)(v2712);
      const uint8_t* v2714 = v21 + 175;
      const int8_t* v2715 = (const int8_t*) v2714;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2716 = *(const int8_t *)(v2715);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2717 = __riscv_vwmacc_vx_i16m1(v2687, v2713, v2708, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2718 = __riscv_vwmacc_vx_i16m1(v2688, v2716, v2710, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2719 = v19 + 1464;
      const uint8_t* v2720 = (const uint8_t*) v2719;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2721 = __riscv_vle8_v_u8mf2(v2720, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2722 = __riscv_vand_vx_u8mf2(v2721, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2723 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2722);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2724 = __riscv_vsrl_vx_u8mf2(v2721, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2724);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2726 = v21 + 143;
      const int8_t* v2727 = (const int8_t*) v2726;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2728 = *(const int8_t *)(v2727);
      const uint8_t* v2729 = v21 + 175;
      const int8_t* v2730 = (const int8_t*) v2729;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2731 = *(const int8_t *)(v2730);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2732 = __riscv_vwmacc_vx_i16m1(v2702, v2728, v2723, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2733 = __riscv_vwmacc_vx_i16m1(v2703, v2731, v2725, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2734 = v19 + 1472;
      const uint8_t* v2735 = (const uint8_t*) v2734;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2736 = __riscv_vle8_v_u8mf2(v2735, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2737 = __riscv_vand_vx_u8mf2(v2736, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2738 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2737);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2739 = __riscv_vsrl_vx_u8mf2(v2736, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2740 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2739);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2741 = v21 + 144;
      const int8_t* v2742 = (const int8_t*) v2741;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2743 = *(const int8_t *)(v2742);
      const uint8_t* v2744 = v21 + 176;
      const int8_t* v2745 = (const int8_t*) v2744;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2746 = *(const int8_t *)(v2745);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2747 = __riscv_vwmacc_vx_i16m1(v2717, v2743, v2738, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2748 = __riscv_vwmacc_vx_i16m1(v2718, v2746, v2740, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2749 = v19 + 1480;
      const uint8_t* v2750 = (const uint8_t*) v2749;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2751 = __riscv_vle8_v_u8mf2(v2750, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2752 = __riscv_vand_vx_u8mf2(v2751, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2753 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2752);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2754 = __riscv_vsrl_vx_u8mf2(v2751, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2755 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2754);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2756 = v21 + 144;
      const int8_t* v2757 = (const int8_t*) v2756;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2758 = *(const int8_t *)(v2757);
      const uint8_t* v2759 = v21 + 176;
      const int8_t* v2760 = (const int8_t*) v2759;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2761 = *(const int8_t *)(v2760);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2762 = __riscv_vwmacc_vx_i16m1(v2732, v2758, v2753, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2763 = __riscv_vwmacc_vx_i16m1(v2733, v2761, v2755, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2764 = v19 + 1488;
      const uint8_t* v2765 = (const uint8_t*) v2764;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2766 = __riscv_vle8_v_u8mf2(v2765, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2767 = __riscv_vand_vx_u8mf2(v2766, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2768 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2767);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2769 = __riscv_vsrl_vx_u8mf2(v2766, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2770 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2769);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2771 = v21 + 145;
      const int8_t* v2772 = (const int8_t*) v2771;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2773 = *(const int8_t *)(v2772);
      const uint8_t* v2774 = v21 + 177;
      const int8_t* v2775 = (const int8_t*) v2774;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2776 = *(const int8_t *)(v2775);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2777 = __riscv_vwmacc_vx_i16m1(v2747, v2773, v2768, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2778 = __riscv_vwmacc_vx_i16m1(v2748, v2776, v2770, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2779 = v19 + 1496;
      const uint8_t* v2780 = (const uint8_t*) v2779;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2781 = __riscv_vle8_v_u8mf2(v2780, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2782 = __riscv_vand_vx_u8mf2(v2781, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2783 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2782);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2784 = __riscv_vsrl_vx_u8mf2(v2781, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2784);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2786 = v21 + 145;
      const int8_t* v2787 = (const int8_t*) v2786;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2788 = *(const int8_t *)(v2787);
      const uint8_t* v2789 = v21 + 177;
      const int8_t* v2790 = (const int8_t*) v2789;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2791 = *(const int8_t *)(v2790);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2792 = __riscv_vwmacc_vx_i16m1(v2762, v2788, v2783, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2793 = __riscv_vwmacc_vx_i16m1(v2763, v2791, v2785, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2794 = v19 + 1504;
      const uint8_t* v2795 = (const uint8_t*) v2794;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2796 = __riscv_vle8_v_u8mf2(v2795, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2797 = __riscv_vand_vx_u8mf2(v2796, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2798 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2797);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2799 = __riscv_vsrl_vx_u8mf2(v2796, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2800 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2799);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2801 = v21 + 146;
      const int8_t* v2802 = (const int8_t*) v2801;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2803 = *(const int8_t *)(v2802);
      const uint8_t* v2804 = v21 + 178;
      const int8_t* v2805 = (const int8_t*) v2804;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2806 = *(const int8_t *)(v2805);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2807 = __riscv_vwmacc_vx_i16m1(v2777, v2803, v2798, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2808 = __riscv_vwmacc_vx_i16m1(v2778, v2806, v2800, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2809 = v19 + 1512;
      const uint8_t* v2810 = (const uint8_t*) v2809;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2811 = __riscv_vle8_v_u8mf2(v2810, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2812 = __riscv_vand_vx_u8mf2(v2811, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2813 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2812);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2814 = __riscv_vsrl_vx_u8mf2(v2811, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2814);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2816 = v21 + 146;
      const int8_t* v2817 = (const int8_t*) v2816;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2818 = *(const int8_t *)(v2817);
      const uint8_t* v2819 = v21 + 178;
      const int8_t* v2820 = (const int8_t*) v2819;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2821 = *(const int8_t *)(v2820);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2822 = __riscv_vwmacc_vx_i16m1(v2792, v2818, v2813, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2823 = __riscv_vwmacc_vx_i16m1(v2793, v2821, v2815, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2824 = v19 + 1520;
      const uint8_t* v2825 = (const uint8_t*) v2824;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2826 = __riscv_vle8_v_u8mf2(v2825, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2827 = __riscv_vand_vx_u8mf2(v2826, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2828 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2827);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2829 = __riscv_vsrl_vx_u8mf2(v2826, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2830 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2829);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2831 = v21 + 147;
      const int8_t* v2832 = (const int8_t*) v2831;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2833 = *(const int8_t *)(v2832);
      const uint8_t* v2834 = v21 + 179;
      const int8_t* v2835 = (const int8_t*) v2834;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2836 = *(const int8_t *)(v2835);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2837 = __riscv_vwmacc_vx_i16m1(v2807, v2833, v2828, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2838 = __riscv_vwmacc_vx_i16m1(v2808, v2836, v2830, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2839 = v19 + 1528;
      const uint8_t* v2840 = (const uint8_t*) v2839;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2841 = __riscv_vle8_v_u8mf2(v2840, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2842 = __riscv_vand_vx_u8mf2(v2841, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2843 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2842);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2844 = __riscv_vsrl_vx_u8mf2(v2841, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2844);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2846 = v21 + 147;
      const int8_t* v2847 = (const int8_t*) v2846;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2848 = *(const int8_t *)(v2847);
      const uint8_t* v2849 = v21 + 179;
      const int8_t* v2850 = (const int8_t*) v2849;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2851 = *(const int8_t *)(v2850);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2852 = __riscv_vwmacc_vx_i16m1(v2822, v2848, v2843, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2853 = __riscv_vwmacc_vx_i16m1(v2823, v2851, v2845, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v2854 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2855 = __riscv_vwmacc_vv_i32m2(v2854, v2204, v2837, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2856 = __riscv_vwmacc_vv_i32m2(v2855, v2221, v2838, 8);
      v34 = v2856;
      vint32m2_t v2857 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2858 = __riscv_vwmacc_vv_i32m2(v2857, v2272, v2852, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2859 = __riscv_vwmacc_vv_i32m2(v2858, v2289, v2853, 8);
      v38 = v2859;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2860 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2861 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2862 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v2863 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2864 = v19 + 1536;
      const uint8_t* v2865 = (const uint8_t*) v2864;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2866 = __riscv_vle8_v_u8mf2(v2865, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2867 = __riscv_vand_vx_u8mf2(v2866, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2868 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2867);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2869 = __riscv_vsrl_vx_u8mf2(v2866, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2870 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2869);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2871 = v21 + 148;
      const int8_t* v2872 = (const int8_t*) v2871;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2873 = *(const int8_t *)(v2872);
      const uint8_t* v2874 = v21 + 180;
      const int8_t* v2875 = (const int8_t*) v2874;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2876 = *(const int8_t *)(v2875);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2877 = __riscv_vwmacc_vx_i16m1(v2860, v2873, v2868, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2878 = __riscv_vwmacc_vx_i16m1(v2861, v2876, v2870, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2879 = v19 + 1544;
      const uint8_t* v2880 = (const uint8_t*) v2879;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2881 = __riscv_vle8_v_u8mf2(v2880, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2882 = __riscv_vand_vx_u8mf2(v2881, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2883 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2882);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2884 = __riscv_vsrl_vx_u8mf2(v2881, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2885 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2884);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2886 = v21 + 148;
      const int8_t* v2887 = (const int8_t*) v2886;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2888 = *(const int8_t *)(v2887);
      const uint8_t* v2889 = v21 + 180;
      const int8_t* v2890 = (const int8_t*) v2889;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2891 = *(const int8_t *)(v2890);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2892 = __riscv_vwmacc_vx_i16m1(v2862, v2888, v2883, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2893 = __riscv_vwmacc_vx_i16m1(v2863, v2891, v2885, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2894 = v19 + 1552;
      const uint8_t* v2895 = (const uint8_t*) v2894;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2896 = __riscv_vle8_v_u8mf2(v2895, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2897 = __riscv_vand_vx_u8mf2(v2896, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2898 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2897);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2899 = __riscv_vsrl_vx_u8mf2(v2896, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2900 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2899);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2901 = v21 + 149;
      const int8_t* v2902 = (const int8_t*) v2901;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2903 = *(const int8_t *)(v2902);
      const uint8_t* v2904 = v21 + 181;
      const int8_t* v2905 = (const int8_t*) v2904;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2906 = *(const int8_t *)(v2905);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2907 = __riscv_vwmacc_vx_i16m1(v2877, v2903, v2898, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2908 = __riscv_vwmacc_vx_i16m1(v2878, v2906, v2900, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2909 = v19 + 1560;
      const uint8_t* v2910 = (const uint8_t*) v2909;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2911 = __riscv_vle8_v_u8mf2(v2910, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2912 = __riscv_vand_vx_u8mf2(v2911, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2913 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2912);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2914 = __riscv_vsrl_vx_u8mf2(v2911, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2915 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2914);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2916 = v21 + 149;
      const int8_t* v2917 = (const int8_t*) v2916;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2918 = *(const int8_t *)(v2917);
      const uint8_t* v2919 = v21 + 181;
      const int8_t* v2920 = (const int8_t*) v2919;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2921 = *(const int8_t *)(v2920);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2922 = __riscv_vwmacc_vx_i16m1(v2892, v2918, v2913, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2923 = __riscv_vwmacc_vx_i16m1(v2893, v2921, v2915, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2924 = v19 + 1568;
      const uint8_t* v2925 = (const uint8_t*) v2924;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2926 = __riscv_vle8_v_u8mf2(v2925, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2927 = __riscv_vand_vx_u8mf2(v2926, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2928 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2927);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2929 = __riscv_vsrl_vx_u8mf2(v2926, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2930 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2929);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2931 = v21 + 150;
      const int8_t* v2932 = (const int8_t*) v2931;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2933 = *(const int8_t *)(v2932);
      const uint8_t* v2934 = v21 + 182;
      const int8_t* v2935 = (const int8_t*) v2934;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2936 = *(const int8_t *)(v2935);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2937 = __riscv_vwmacc_vx_i16m1(v2907, v2933, v2928, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2938 = __riscv_vwmacc_vx_i16m1(v2908, v2936, v2930, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2939 = v19 + 1576;
      const uint8_t* v2940 = (const uint8_t*) v2939;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2941 = __riscv_vle8_v_u8mf2(v2940, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2942 = __riscv_vand_vx_u8mf2(v2941, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2943 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2942);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2944 = __riscv_vsrl_vx_u8mf2(v2941, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2945 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2944);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2946 = v21 + 150;
      const int8_t* v2947 = (const int8_t*) v2946;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2948 = *(const int8_t *)(v2947);
      const uint8_t* v2949 = v21 + 182;
      const int8_t* v2950 = (const int8_t*) v2949;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2951 = *(const int8_t *)(v2950);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2952 = __riscv_vwmacc_vx_i16m1(v2922, v2948, v2943, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2953 = __riscv_vwmacc_vx_i16m1(v2923, v2951, v2945, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2954 = v19 + 1584;
      const uint8_t* v2955 = (const uint8_t*) v2954;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2956 = __riscv_vle8_v_u8mf2(v2955, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2957 = __riscv_vand_vx_u8mf2(v2956, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2958 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2957);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2959 = __riscv_vsrl_vx_u8mf2(v2956, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2960 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2959);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2961 = v21 + 151;
      const int8_t* v2962 = (const int8_t*) v2961;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2963 = *(const int8_t *)(v2962);
      const uint8_t* v2964 = v21 + 183;
      const int8_t* v2965 = (const int8_t*) v2964;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2966 = *(const int8_t *)(v2965);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2967 = __riscv_vwmacc_vx_i16m1(v2937, v2963, v2958, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2968 = __riscv_vwmacc_vx_i16m1(v2938, v2966, v2960, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2969 = v19 + 1592;
      const uint8_t* v2970 = (const uint8_t*) v2969;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2971 = __riscv_vle8_v_u8mf2(v2970, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2972 = __riscv_vand_vx_u8mf2(v2971, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2973 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2972);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2974 = __riscv_vsrl_vx_u8mf2(v2971, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2975 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2974);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2976 = v21 + 151;
      const int8_t* v2977 = (const int8_t*) v2976;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2978 = *(const int8_t *)(v2977);
      const uint8_t* v2979 = v21 + 183;
      const int8_t* v2980 = (const int8_t*) v2979;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2981 = *(const int8_t *)(v2980);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2982 = __riscv_vwmacc_vx_i16m1(v2952, v2978, v2973, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2983 = __riscv_vwmacc_vx_i16m1(v2953, v2981, v2975, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2984 = v19 + 1600;
      const uint8_t* v2985 = (const uint8_t*) v2984;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2986 = __riscv_vle8_v_u8mf2(v2985, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2987 = __riscv_vand_vx_u8mf2(v2986, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2988 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2987);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2989 = __riscv_vsrl_vx_u8mf2(v2986, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2990 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2989);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2991 = v21 + 152;
      const int8_t* v2992 = (const int8_t*) v2991;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2993 = *(const int8_t *)(v2992);
      const uint8_t* v2994 = v21 + 184;
      const int8_t* v2995 = (const int8_t*) v2994;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2996 = *(const int8_t *)(v2995);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2997 = __riscv_vwmacc_vx_i16m1(v2967, v2993, v2988, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2998 = __riscv_vwmacc_vx_i16m1(v2968, v2996, v2990, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2999 = v19 + 1608;
      const uint8_t* v3000 = (const uint8_t*) v2999;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3001 = __riscv_vle8_v_u8mf2(v3000, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3002 = __riscv_vand_vx_u8mf2(v3001, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3003 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3002);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3004 = __riscv_vsrl_vx_u8mf2(v3001, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3005 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3004);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3006 = v21 + 152;
      const int8_t* v3007 = (const int8_t*) v3006;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3008 = *(const int8_t *)(v3007);
      const uint8_t* v3009 = v21 + 184;
      const int8_t* v3010 = (const int8_t*) v3009;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3011 = *(const int8_t *)(v3010);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3012 = __riscv_vwmacc_vx_i16m1(v2982, v3008, v3003, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3013 = __riscv_vwmacc_vx_i16m1(v2983, v3011, v3005, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3014 = v19 + 1616;
      const uint8_t* v3015 = (const uint8_t*) v3014;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3016 = __riscv_vle8_v_u8mf2(v3015, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3017 = __riscv_vand_vx_u8mf2(v3016, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3018 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3017);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3019 = __riscv_vsrl_vx_u8mf2(v3016, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3020 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3019);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3021 = v21 + 153;
      const int8_t* v3022 = (const int8_t*) v3021;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3023 = *(const int8_t *)(v3022);
      const uint8_t* v3024 = v21 + 185;
      const int8_t* v3025 = (const int8_t*) v3024;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3026 = *(const int8_t *)(v3025);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3027 = __riscv_vwmacc_vx_i16m1(v2997, v3023, v3018, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3028 = __riscv_vwmacc_vx_i16m1(v2998, v3026, v3020, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3029 = v19 + 1624;
      const uint8_t* v3030 = (const uint8_t*) v3029;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3031 = __riscv_vle8_v_u8mf2(v3030, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3032 = __riscv_vand_vx_u8mf2(v3031, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3033 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3032);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3034 = __riscv_vsrl_vx_u8mf2(v3031, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3035 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3034);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3036 = v21 + 153;
      const int8_t* v3037 = (const int8_t*) v3036;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3038 = *(const int8_t *)(v3037);
      const uint8_t* v3039 = v21 + 185;
      const int8_t* v3040 = (const int8_t*) v3039;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3041 = *(const int8_t *)(v3040);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3042 = __riscv_vwmacc_vx_i16m1(v3012, v3038, v3033, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3043 = __riscv_vwmacc_vx_i16m1(v3013, v3041, v3035, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3044 = v19 + 1632;
      const uint8_t* v3045 = (const uint8_t*) v3044;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3046 = __riscv_vle8_v_u8mf2(v3045, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3047 = __riscv_vand_vx_u8mf2(v3046, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3048 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3047);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3049 = __riscv_vsrl_vx_u8mf2(v3046, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3050 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3049);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3051 = v21 + 154;
      const int8_t* v3052 = (const int8_t*) v3051;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3053 = *(const int8_t *)(v3052);
      const uint8_t* v3054 = v21 + 186;
      const int8_t* v3055 = (const int8_t*) v3054;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3056 = *(const int8_t *)(v3055);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3057 = __riscv_vwmacc_vx_i16m1(v3027, v3053, v3048, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3058 = __riscv_vwmacc_vx_i16m1(v3028, v3056, v3050, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3059 = v19 + 1640;
      const uint8_t* v3060 = (const uint8_t*) v3059;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3061 = __riscv_vle8_v_u8mf2(v3060, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3062 = __riscv_vand_vx_u8mf2(v3061, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3063 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3062);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3064 = __riscv_vsrl_vx_u8mf2(v3061, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3065 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3064);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3066 = v21 + 154;
      const int8_t* v3067 = (const int8_t*) v3066;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3068 = *(const int8_t *)(v3067);
      const uint8_t* v3069 = v21 + 186;
      const int8_t* v3070 = (const int8_t*) v3069;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3071 = *(const int8_t *)(v3070);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3072 = __riscv_vwmacc_vx_i16m1(v3042, v3068, v3063, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3073 = __riscv_vwmacc_vx_i16m1(v3043, v3071, v3065, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3074 = v19 + 1648;
      const uint8_t* v3075 = (const uint8_t*) v3074;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3076 = __riscv_vle8_v_u8mf2(v3075, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3077 = __riscv_vand_vx_u8mf2(v3076, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3078 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3077);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3079 = __riscv_vsrl_vx_u8mf2(v3076, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3080 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3079);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3081 = v21 + 155;
      const int8_t* v3082 = (const int8_t*) v3081;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3083 = *(const int8_t *)(v3082);
      const uint8_t* v3084 = v21 + 187;
      const int8_t* v3085 = (const int8_t*) v3084;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3086 = *(const int8_t *)(v3085);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3087 = __riscv_vwmacc_vx_i16m1(v3057, v3083, v3078, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3088 = __riscv_vwmacc_vx_i16m1(v3058, v3086, v3080, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3089 = v19 + 1656;
      const uint8_t* v3090 = (const uint8_t*) v3089;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3091 = __riscv_vle8_v_u8mf2(v3090, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3092 = __riscv_vand_vx_u8mf2(v3091, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3093 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3092);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3094 = __riscv_vsrl_vx_u8mf2(v3091, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3095 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3094);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3096 = v21 + 155;
      const int8_t* v3097 = (const int8_t*) v3096;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3098 = *(const int8_t *)(v3097);
      const uint8_t* v3099 = v21 + 187;
      const int8_t* v3100 = (const int8_t*) v3099;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3101 = *(const int8_t *)(v3100);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3102 = __riscv_vwmacc_vx_i16m1(v3072, v3098, v3093, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3103 = __riscv_vwmacc_vx_i16m1(v3073, v3101, v3095, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3104 = v19 + 1664;
      const uint8_t* v3105 = (const uint8_t*) v3104;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3106 = __riscv_vle8_v_u8mf2(v3105, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3107 = __riscv_vand_vx_u8mf2(v3106, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3108 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3107);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3109 = __riscv_vsrl_vx_u8mf2(v3106, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3110 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3109);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3111 = v21 + 156;
      const int8_t* v3112 = (const int8_t*) v3111;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3113 = *(const int8_t *)(v3112);
      const uint8_t* v3114 = v21 + 188;
      const int8_t* v3115 = (const int8_t*) v3114;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3116 = *(const int8_t *)(v3115);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3117 = __riscv_vwmacc_vx_i16m1(v3087, v3113, v3108, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3118 = __riscv_vwmacc_vx_i16m1(v3088, v3116, v3110, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3119 = v19 + 1672;
      const uint8_t* v3120 = (const uint8_t*) v3119;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3121 = __riscv_vle8_v_u8mf2(v3120, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3122 = __riscv_vand_vx_u8mf2(v3121, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3123 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3122);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3124 = __riscv_vsrl_vx_u8mf2(v3121, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3125 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3124);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3126 = v21 + 156;
      const int8_t* v3127 = (const int8_t*) v3126;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3128 = *(const int8_t *)(v3127);
      const uint8_t* v3129 = v21 + 188;
      const int8_t* v3130 = (const int8_t*) v3129;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3131 = *(const int8_t *)(v3130);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3132 = __riscv_vwmacc_vx_i16m1(v3102, v3128, v3123, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3133 = __riscv_vwmacc_vx_i16m1(v3103, v3131, v3125, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3134 = v19 + 1680;
      const uint8_t* v3135 = (const uint8_t*) v3134;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3136 = __riscv_vle8_v_u8mf2(v3135, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3137 = __riscv_vand_vx_u8mf2(v3136, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3138 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3137);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3139 = __riscv_vsrl_vx_u8mf2(v3136, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3140 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3139);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3141 = v21 + 157;
      const int8_t* v3142 = (const int8_t*) v3141;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3143 = *(const int8_t *)(v3142);
      const uint8_t* v3144 = v21 + 189;
      const int8_t* v3145 = (const int8_t*) v3144;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3146 = *(const int8_t *)(v3145);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3147 = __riscv_vwmacc_vx_i16m1(v3117, v3143, v3138, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3148 = __riscv_vwmacc_vx_i16m1(v3118, v3146, v3140, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3149 = v19 + 1688;
      const uint8_t* v3150 = (const uint8_t*) v3149;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3151 = __riscv_vle8_v_u8mf2(v3150, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3152 = __riscv_vand_vx_u8mf2(v3151, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3153 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3152);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3154 = __riscv_vsrl_vx_u8mf2(v3151, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3155 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3154);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3156 = v21 + 157;
      const int8_t* v3157 = (const int8_t*) v3156;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3158 = *(const int8_t *)(v3157);
      const uint8_t* v3159 = v21 + 189;
      const int8_t* v3160 = (const int8_t*) v3159;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3161 = *(const int8_t *)(v3160);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3162 = __riscv_vwmacc_vx_i16m1(v3132, v3158, v3153, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3163 = __riscv_vwmacc_vx_i16m1(v3133, v3161, v3155, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3164 = v19 + 1696;
      const uint8_t* v3165 = (const uint8_t*) v3164;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3166 = __riscv_vle8_v_u8mf2(v3165, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3167 = __riscv_vand_vx_u8mf2(v3166, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3168 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3167);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3169 = __riscv_vsrl_vx_u8mf2(v3166, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3170 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3169);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3171 = v21 + 158;
      const int8_t* v3172 = (const int8_t*) v3171;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3173 = *(const int8_t *)(v3172);
      const uint8_t* v3174 = v21 + 190;
      const int8_t* v3175 = (const int8_t*) v3174;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3176 = *(const int8_t *)(v3175);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3177 = __riscv_vwmacc_vx_i16m1(v3147, v3173, v3168, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3178 = __riscv_vwmacc_vx_i16m1(v3148, v3176, v3170, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3179 = v19 + 1704;
      const uint8_t* v3180 = (const uint8_t*) v3179;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3181 = __riscv_vle8_v_u8mf2(v3180, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3182 = __riscv_vand_vx_u8mf2(v3181, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3183 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3182);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3184 = __riscv_vsrl_vx_u8mf2(v3181, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3185 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3184);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3186 = v21 + 158;
      const int8_t* v3187 = (const int8_t*) v3186;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3188 = *(const int8_t *)(v3187);
      const uint8_t* v3189 = v21 + 190;
      const int8_t* v3190 = (const int8_t*) v3189;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3191 = *(const int8_t *)(v3190);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3192 = __riscv_vwmacc_vx_i16m1(v3162, v3188, v3183, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3193 = __riscv_vwmacc_vx_i16m1(v3163, v3191, v3185, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3194 = v19 + 1712;
      const uint8_t* v3195 = (const uint8_t*) v3194;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3196 = __riscv_vle8_v_u8mf2(v3195, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3197 = __riscv_vand_vx_u8mf2(v3196, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3198 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3197);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3199 = __riscv_vsrl_vx_u8mf2(v3196, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3200 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3199);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3201 = v21 + 159;
      const int8_t* v3202 = (const int8_t*) v3201;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3203 = *(const int8_t *)(v3202);
      const uint8_t* v3204 = v21 + 191;
      const int8_t* v3205 = (const int8_t*) v3204;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3206 = *(const int8_t *)(v3205);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3207 = __riscv_vwmacc_vx_i16m1(v3177, v3203, v3198, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3208 = __riscv_vwmacc_vx_i16m1(v3178, v3206, v3200, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3209 = v19 + 1720;
      const uint8_t* v3210 = (const uint8_t*) v3209;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3211 = __riscv_vle8_v_u8mf2(v3210, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3212 = __riscv_vand_vx_u8mf2(v3211, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3213 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3212);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3214 = __riscv_vsrl_vx_u8mf2(v3211, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3215 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3214);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3216 = v21 + 159;
      const int8_t* v3217 = (const int8_t*) v3216;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3218 = *(const int8_t *)(v3217);
      const uint8_t* v3219 = v21 + 191;
      const int8_t* v3220 = (const int8_t*) v3219;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3221 = *(const int8_t *)(v3220);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3222 = __riscv_vwmacc_vx_i16m1(v3192, v3218, v3213, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3223 = __riscv_vwmacc_vx_i16m1(v3193, v3221, v3215, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3224 = v19 + 1728;
      const uint8_t* v3225 = (const uint8_t*) v3224;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3226 = __riscv_vle8_v_u8mf2(v3225, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3227 = __riscv_vand_vx_u8mf2(v3226, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3228 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3227);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3229 = __riscv_vsrl_vx_u8mf2(v3226, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3230 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3229);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3231 = v21 + 160;
      const int8_t* v3232 = (const int8_t*) v3231;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3233 = *(const int8_t *)(v3232);
      const uint8_t* v3234 = v21 + 192;
      const int8_t* v3235 = (const int8_t*) v3234;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3236 = *(const int8_t *)(v3235);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3237 = __riscv_vwmacc_vx_i16m1(v3207, v3233, v3228, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3238 = __riscv_vwmacc_vx_i16m1(v3208, v3236, v3230, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3239 = v19 + 1736;
      const uint8_t* v3240 = (const uint8_t*) v3239;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3241 = __riscv_vle8_v_u8mf2(v3240, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3242 = __riscv_vand_vx_u8mf2(v3241, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3243 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3242);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3244 = __riscv_vsrl_vx_u8mf2(v3241, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3245 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3244);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3246 = v21 + 160;
      const int8_t* v3247 = (const int8_t*) v3246;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3248 = *(const int8_t *)(v3247);
      const uint8_t* v3249 = v21 + 192;
      const int8_t* v3250 = (const int8_t*) v3249;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3251 = *(const int8_t *)(v3250);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3252 = __riscv_vwmacc_vx_i16m1(v3222, v3248, v3243, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3253 = __riscv_vwmacc_vx_i16m1(v3223, v3251, v3245, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3254 = v19 + 1744;
      const uint8_t* v3255 = (const uint8_t*) v3254;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3256 = __riscv_vle8_v_u8mf2(v3255, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3257 = __riscv_vand_vx_u8mf2(v3256, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3258 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3257);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3259 = __riscv_vsrl_vx_u8mf2(v3256, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3260 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3259);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3261 = v21 + 161;
      const int8_t* v3262 = (const int8_t*) v3261;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3263 = *(const int8_t *)(v3262);
      const uint8_t* v3264 = v21 + 193;
      const int8_t* v3265 = (const int8_t*) v3264;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3266 = *(const int8_t *)(v3265);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3267 = __riscv_vwmacc_vx_i16m1(v3237, v3263, v3258, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3268 = __riscv_vwmacc_vx_i16m1(v3238, v3266, v3260, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3269 = v19 + 1752;
      const uint8_t* v3270 = (const uint8_t*) v3269;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3271 = __riscv_vle8_v_u8mf2(v3270, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3272 = __riscv_vand_vx_u8mf2(v3271, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3273 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3272);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3274 = __riscv_vsrl_vx_u8mf2(v3271, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3275 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3274);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3276 = v21 + 161;
      const int8_t* v3277 = (const int8_t*) v3276;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3278 = *(const int8_t *)(v3277);
      const uint8_t* v3279 = v21 + 193;
      const int8_t* v3280 = (const int8_t*) v3279;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3281 = *(const int8_t *)(v3280);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3282 = __riscv_vwmacc_vx_i16m1(v3252, v3278, v3273, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3283 = __riscv_vwmacc_vx_i16m1(v3253, v3281, v3275, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3284 = v19 + 1760;
      const uint8_t* v3285 = (const uint8_t*) v3284;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3286 = __riscv_vle8_v_u8mf2(v3285, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3287 = __riscv_vand_vx_u8mf2(v3286, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3288 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3287);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3289 = __riscv_vsrl_vx_u8mf2(v3286, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3290 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3289);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3291 = v21 + 162;
      const int8_t* v3292 = (const int8_t*) v3291;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3293 = *(const int8_t *)(v3292);
      const uint8_t* v3294 = v21 + 194;
      const int8_t* v3295 = (const int8_t*) v3294;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3296 = *(const int8_t *)(v3295);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3297 = __riscv_vwmacc_vx_i16m1(v3267, v3293, v3288, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3298 = __riscv_vwmacc_vx_i16m1(v3268, v3296, v3290, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3299 = v19 + 1768;
      const uint8_t* v3300 = (const uint8_t*) v3299;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3301 = __riscv_vle8_v_u8mf2(v3300, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3302 = __riscv_vand_vx_u8mf2(v3301, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3303 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3302);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3304 = __riscv_vsrl_vx_u8mf2(v3301, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3305 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3304);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3306 = v21 + 162;
      const int8_t* v3307 = (const int8_t*) v3306;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3308 = *(const int8_t *)(v3307);
      const uint8_t* v3309 = v21 + 194;
      const int8_t* v3310 = (const int8_t*) v3309;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3311 = *(const int8_t *)(v3310);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3312 = __riscv_vwmacc_vx_i16m1(v3282, v3308, v3303, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3313 = __riscv_vwmacc_vx_i16m1(v3283, v3311, v3305, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3314 = v19 + 1776;
      const uint8_t* v3315 = (const uint8_t*) v3314;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3316 = __riscv_vle8_v_u8mf2(v3315, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3317 = __riscv_vand_vx_u8mf2(v3316, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3318 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3317);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3319 = __riscv_vsrl_vx_u8mf2(v3316, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3320 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3319);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3321 = v21 + 163;
      const int8_t* v3322 = (const int8_t*) v3321;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3323 = *(const int8_t *)(v3322);
      const uint8_t* v3324 = v21 + 195;
      const int8_t* v3325 = (const int8_t*) v3324;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3326 = *(const int8_t *)(v3325);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3327 = __riscv_vwmacc_vx_i16m1(v3297, v3323, v3318, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3328 = __riscv_vwmacc_vx_i16m1(v3298, v3326, v3320, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3329 = v19 + 1784;
      const uint8_t* v3330 = (const uint8_t*) v3329;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3331 = __riscv_vle8_v_u8mf2(v3330, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3332 = __riscv_vand_vx_u8mf2(v3331, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3333 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3332);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3334 = __riscv_vsrl_vx_u8mf2(v3331, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3335 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3334);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3336 = v21 + 163;
      const int8_t* v3337 = (const int8_t*) v3336;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3338 = *(const int8_t *)(v3337);
      const uint8_t* v3339 = v21 + 195;
      const int8_t* v3340 = (const int8_t*) v3339;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3341 = *(const int8_t *)(v3340);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3342 = __riscv_vwmacc_vx_i16m1(v3312, v3338, v3333, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3343 = __riscv_vwmacc_vx_i16m1(v3313, v3341, v3335, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v3344 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3345 = __riscv_vwmacc_vv_i32m2(v3344, v2204, v3327, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3346 = __riscv_vwmacc_vv_i32m2(v3345, v2221, v3328, 8);
      v34 = v3346;
      vint32m2_t v3347 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3348 = __riscv_vwmacc_vv_i32m2(v3347, v2272, v3342, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3349 = __riscv_vwmacc_vv_i32m2(v3348, v2289, v3343, 8);
      v38 = v3349;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3350 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3351 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3352 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3353 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3354 = v19 + 1792;
      const uint8_t* v3355 = (const uint8_t*) v3354;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3356 = __riscv_vle8_v_u8mf2(v3355, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3357 = __riscv_vand_vx_u8mf2(v3356, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3358 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3357);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3359 = __riscv_vsrl_vx_u8mf2(v3356, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3360 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3359);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3361 = v21 + 196;
      const int8_t* v3362 = (const int8_t*) v3361;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3363 = *(const int8_t *)(v3362);
      const uint8_t* v3364 = v21 + 228;
      const int8_t* v3365 = (const int8_t*) v3364;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3366 = *(const int8_t *)(v3365);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3367 = __riscv_vwmacc_vx_i16m1(v3350, v3363, v3358, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3368 = __riscv_vwmacc_vx_i16m1(v3351, v3366, v3360, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3369 = v19 + 1800;
      const uint8_t* v3370 = (const uint8_t*) v3369;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3371 = __riscv_vle8_v_u8mf2(v3370, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3372 = __riscv_vand_vx_u8mf2(v3371, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3373 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3372);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3374 = __riscv_vsrl_vx_u8mf2(v3371, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3375 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3374);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3376 = v21 + 196;
      const int8_t* v3377 = (const int8_t*) v3376;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3378 = *(const int8_t *)(v3377);
      const uint8_t* v3379 = v21 + 228;
      const int8_t* v3380 = (const int8_t*) v3379;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3381 = *(const int8_t *)(v3380);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3382 = __riscv_vwmacc_vx_i16m1(v3352, v3378, v3373, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3383 = __riscv_vwmacc_vx_i16m1(v3353, v3381, v3375, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3384 = v19 + 1808;
      const uint8_t* v3385 = (const uint8_t*) v3384;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3386 = __riscv_vle8_v_u8mf2(v3385, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3387 = __riscv_vand_vx_u8mf2(v3386, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3388 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3387);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3389 = __riscv_vsrl_vx_u8mf2(v3386, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3390 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3389);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3391 = v21 + 197;
      const int8_t* v3392 = (const int8_t*) v3391;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3393 = *(const int8_t *)(v3392);
      const uint8_t* v3394 = v21 + 229;
      const int8_t* v3395 = (const int8_t*) v3394;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3396 = *(const int8_t *)(v3395);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3397 = __riscv_vwmacc_vx_i16m1(v3367, v3393, v3388, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3398 = __riscv_vwmacc_vx_i16m1(v3368, v3396, v3390, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3399 = v19 + 1816;
      const uint8_t* v3400 = (const uint8_t*) v3399;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3401 = __riscv_vle8_v_u8mf2(v3400, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3402 = __riscv_vand_vx_u8mf2(v3401, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3403 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3402);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3404 = __riscv_vsrl_vx_u8mf2(v3401, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3405 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3404);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3406 = v21 + 197;
      const int8_t* v3407 = (const int8_t*) v3406;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3408 = *(const int8_t *)(v3407);
      const uint8_t* v3409 = v21 + 229;
      const int8_t* v3410 = (const int8_t*) v3409;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3411 = *(const int8_t *)(v3410);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3412 = __riscv_vwmacc_vx_i16m1(v3382, v3408, v3403, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3413 = __riscv_vwmacc_vx_i16m1(v3383, v3411, v3405, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3414 = v19 + 1824;
      const uint8_t* v3415 = (const uint8_t*) v3414;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3416 = __riscv_vle8_v_u8mf2(v3415, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3417 = __riscv_vand_vx_u8mf2(v3416, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3418 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3417);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3419 = __riscv_vsrl_vx_u8mf2(v3416, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3419);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3421 = v21 + 198;
      const int8_t* v3422 = (const int8_t*) v3421;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3423 = *(const int8_t *)(v3422);
      const uint8_t* v3424 = v21 + 230;
      const int8_t* v3425 = (const int8_t*) v3424;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3426 = *(const int8_t *)(v3425);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3427 = __riscv_vwmacc_vx_i16m1(v3397, v3423, v3418, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3428 = __riscv_vwmacc_vx_i16m1(v3398, v3426, v3420, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3429 = v19 + 1832;
      const uint8_t* v3430 = (const uint8_t*) v3429;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3431 = __riscv_vle8_v_u8mf2(v3430, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3432 = __riscv_vand_vx_u8mf2(v3431, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3433 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3432);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3434 = __riscv_vsrl_vx_u8mf2(v3431, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3435 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3434);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3436 = v21 + 198;
      const int8_t* v3437 = (const int8_t*) v3436;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3438 = *(const int8_t *)(v3437);
      const uint8_t* v3439 = v21 + 230;
      const int8_t* v3440 = (const int8_t*) v3439;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3441 = *(const int8_t *)(v3440);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3442 = __riscv_vwmacc_vx_i16m1(v3412, v3438, v3433, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3443 = __riscv_vwmacc_vx_i16m1(v3413, v3441, v3435, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3444 = v19 + 1840;
      const uint8_t* v3445 = (const uint8_t*) v3444;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3446 = __riscv_vle8_v_u8mf2(v3445, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3447 = __riscv_vand_vx_u8mf2(v3446, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3448 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3447);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3449 = __riscv_vsrl_vx_u8mf2(v3446, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3450 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3449);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3451 = v21 + 199;
      const int8_t* v3452 = (const int8_t*) v3451;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3453 = *(const int8_t *)(v3452);
      const uint8_t* v3454 = v21 + 231;
      const int8_t* v3455 = (const int8_t*) v3454;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3456 = *(const int8_t *)(v3455);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3457 = __riscv_vwmacc_vx_i16m1(v3427, v3453, v3448, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3458 = __riscv_vwmacc_vx_i16m1(v3428, v3456, v3450, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3459 = v19 + 1848;
      const uint8_t* v3460 = (const uint8_t*) v3459;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3461 = __riscv_vle8_v_u8mf2(v3460, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3462 = __riscv_vand_vx_u8mf2(v3461, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3463 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3462);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3464 = __riscv_vsrl_vx_u8mf2(v3461, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3465 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3464);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3466 = v21 + 199;
      const int8_t* v3467 = (const int8_t*) v3466;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3468 = *(const int8_t *)(v3467);
      const uint8_t* v3469 = v21 + 231;
      const int8_t* v3470 = (const int8_t*) v3469;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3471 = *(const int8_t *)(v3470);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3472 = __riscv_vwmacc_vx_i16m1(v3442, v3468, v3463, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3473 = __riscv_vwmacc_vx_i16m1(v3443, v3471, v3465, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3474 = v19 + 1856;
      const uint8_t* v3475 = (const uint8_t*) v3474;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3476 = __riscv_vle8_v_u8mf2(v3475, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3477 = __riscv_vand_vx_u8mf2(v3476, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3478 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3477);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3479 = __riscv_vsrl_vx_u8mf2(v3476, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3480 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3479);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3481 = v21 + 200;
      const int8_t* v3482 = (const int8_t*) v3481;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3483 = *(const int8_t *)(v3482);
      const uint8_t* v3484 = v21 + 232;
      const int8_t* v3485 = (const int8_t*) v3484;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3486 = *(const int8_t *)(v3485);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3487 = __riscv_vwmacc_vx_i16m1(v3457, v3483, v3478, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3488 = __riscv_vwmacc_vx_i16m1(v3458, v3486, v3480, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3489 = v19 + 1864;
      const uint8_t* v3490 = (const uint8_t*) v3489;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3491 = __riscv_vle8_v_u8mf2(v3490, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3492 = __riscv_vand_vx_u8mf2(v3491, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3493 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3492);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3494 = __riscv_vsrl_vx_u8mf2(v3491, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3495 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3494);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3496 = v21 + 200;
      const int8_t* v3497 = (const int8_t*) v3496;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3498 = *(const int8_t *)(v3497);
      const uint8_t* v3499 = v21 + 232;
      const int8_t* v3500 = (const int8_t*) v3499;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3501 = *(const int8_t *)(v3500);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3502 = __riscv_vwmacc_vx_i16m1(v3472, v3498, v3493, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3503 = __riscv_vwmacc_vx_i16m1(v3473, v3501, v3495, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3504 = v19 + 1872;
      const uint8_t* v3505 = (const uint8_t*) v3504;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3506 = __riscv_vle8_v_u8mf2(v3505, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3507 = __riscv_vand_vx_u8mf2(v3506, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3508 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3507);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3509 = __riscv_vsrl_vx_u8mf2(v3506, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3509);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3511 = v21 + 201;
      const int8_t* v3512 = (const int8_t*) v3511;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3513 = *(const int8_t *)(v3512);
      const uint8_t* v3514 = v21 + 233;
      const int8_t* v3515 = (const int8_t*) v3514;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3516 = *(const int8_t *)(v3515);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3517 = __riscv_vwmacc_vx_i16m1(v3487, v3513, v3508, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3518 = __riscv_vwmacc_vx_i16m1(v3488, v3516, v3510, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3519 = v19 + 1880;
      const uint8_t* v3520 = (const uint8_t*) v3519;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3521 = __riscv_vle8_v_u8mf2(v3520, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3522 = __riscv_vand_vx_u8mf2(v3521, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3523 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3522);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3524 = __riscv_vsrl_vx_u8mf2(v3521, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3525 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3524);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3526 = v21 + 201;
      const int8_t* v3527 = (const int8_t*) v3526;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3528 = *(const int8_t *)(v3527);
      const uint8_t* v3529 = v21 + 233;
      const int8_t* v3530 = (const int8_t*) v3529;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3531 = *(const int8_t *)(v3530);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3532 = __riscv_vwmacc_vx_i16m1(v3502, v3528, v3523, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3533 = __riscv_vwmacc_vx_i16m1(v3503, v3531, v3525, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3534 = v19 + 1888;
      const uint8_t* v3535 = (const uint8_t*) v3534;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3536 = __riscv_vle8_v_u8mf2(v3535, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3537 = __riscv_vand_vx_u8mf2(v3536, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3538 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3537);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3539 = __riscv_vsrl_vx_u8mf2(v3536, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3540 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3539);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3541 = v21 + 202;
      const int8_t* v3542 = (const int8_t*) v3541;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3543 = *(const int8_t *)(v3542);
      const uint8_t* v3544 = v21 + 234;
      const int8_t* v3545 = (const int8_t*) v3544;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3546 = *(const int8_t *)(v3545);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3547 = __riscv_vwmacc_vx_i16m1(v3517, v3543, v3538, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3548 = __riscv_vwmacc_vx_i16m1(v3518, v3546, v3540, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3549 = v19 + 1896;
      const uint8_t* v3550 = (const uint8_t*) v3549;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3551 = __riscv_vle8_v_u8mf2(v3550, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3552 = __riscv_vand_vx_u8mf2(v3551, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3553 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3552);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3554 = __riscv_vsrl_vx_u8mf2(v3551, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3554);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3556 = v21 + 202;
      const int8_t* v3557 = (const int8_t*) v3556;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3558 = *(const int8_t *)(v3557);
      const uint8_t* v3559 = v21 + 234;
      const int8_t* v3560 = (const int8_t*) v3559;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3561 = *(const int8_t *)(v3560);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3562 = __riscv_vwmacc_vx_i16m1(v3532, v3558, v3553, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3563 = __riscv_vwmacc_vx_i16m1(v3533, v3561, v3555, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3564 = v19 + 1904;
      const uint8_t* v3565 = (const uint8_t*) v3564;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3566 = __riscv_vle8_v_u8mf2(v3565, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3567 = __riscv_vand_vx_u8mf2(v3566, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3568 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3567);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3569 = __riscv_vsrl_vx_u8mf2(v3566, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3570 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3569);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3571 = v21 + 203;
      const int8_t* v3572 = (const int8_t*) v3571;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3573 = *(const int8_t *)(v3572);
      const uint8_t* v3574 = v21 + 235;
      const int8_t* v3575 = (const int8_t*) v3574;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3576 = *(const int8_t *)(v3575);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3577 = __riscv_vwmacc_vx_i16m1(v3547, v3573, v3568, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3578 = __riscv_vwmacc_vx_i16m1(v3548, v3576, v3570, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3579 = v19 + 1912;
      const uint8_t* v3580 = (const uint8_t*) v3579;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3581 = __riscv_vle8_v_u8mf2(v3580, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3582 = __riscv_vand_vx_u8mf2(v3581, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3583 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3582);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3584 = __riscv_vsrl_vx_u8mf2(v3581, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3585 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3584);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3586 = v21 + 203;
      const int8_t* v3587 = (const int8_t*) v3586;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3588 = *(const int8_t *)(v3587);
      const uint8_t* v3589 = v21 + 235;
      const int8_t* v3590 = (const int8_t*) v3589;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3591 = *(const int8_t *)(v3590);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3592 = __riscv_vwmacc_vx_i16m1(v3562, v3588, v3583, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3593 = __riscv_vwmacc_vx_i16m1(v3563, v3591, v3585, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3594 = v19 + 1920;
      const uint8_t* v3595 = (const uint8_t*) v3594;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3596 = __riscv_vle8_v_u8mf2(v3595, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3597 = __riscv_vand_vx_u8mf2(v3596, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3598 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3597);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3599 = __riscv_vsrl_vx_u8mf2(v3596, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3600 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3599);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3601 = v21 + 204;
      const int8_t* v3602 = (const int8_t*) v3601;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3603 = *(const int8_t *)(v3602);
      const uint8_t* v3604 = v21 + 236;
      const int8_t* v3605 = (const int8_t*) v3604;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3606 = *(const int8_t *)(v3605);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3607 = __riscv_vwmacc_vx_i16m1(v3577, v3603, v3598, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3608 = __riscv_vwmacc_vx_i16m1(v3578, v3606, v3600, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3609 = v19 + 1928;
      const uint8_t* v3610 = (const uint8_t*) v3609;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3611 = __riscv_vle8_v_u8mf2(v3610, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3612 = __riscv_vand_vx_u8mf2(v3611, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3613 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3612);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3614 = __riscv_vsrl_vx_u8mf2(v3611, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3615 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3614);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3616 = v21 + 204;
      const int8_t* v3617 = (const int8_t*) v3616;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3618 = *(const int8_t *)(v3617);
      const uint8_t* v3619 = v21 + 236;
      const int8_t* v3620 = (const int8_t*) v3619;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3621 = *(const int8_t *)(v3620);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3622 = __riscv_vwmacc_vx_i16m1(v3592, v3618, v3613, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3623 = __riscv_vwmacc_vx_i16m1(v3593, v3621, v3615, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3624 = v19 + 1936;
      const uint8_t* v3625 = (const uint8_t*) v3624;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3626 = __riscv_vle8_v_u8mf2(v3625, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3627 = __riscv_vand_vx_u8mf2(v3626, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3628 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3627);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3629 = __riscv_vsrl_vx_u8mf2(v3626, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3630 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3629);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3631 = v21 + 205;
      const int8_t* v3632 = (const int8_t*) v3631;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3633 = *(const int8_t *)(v3632);
      const uint8_t* v3634 = v21 + 237;
      const int8_t* v3635 = (const int8_t*) v3634;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3636 = *(const int8_t *)(v3635);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3637 = __riscv_vwmacc_vx_i16m1(v3607, v3633, v3628, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3638 = __riscv_vwmacc_vx_i16m1(v3608, v3636, v3630, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3639 = v19 + 1944;
      const uint8_t* v3640 = (const uint8_t*) v3639;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3641 = __riscv_vle8_v_u8mf2(v3640, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3642 = __riscv_vand_vx_u8mf2(v3641, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3643 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3642);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3644 = __riscv_vsrl_vx_u8mf2(v3641, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3645 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3644);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3646 = v21 + 205;
      const int8_t* v3647 = (const int8_t*) v3646;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3648 = *(const int8_t *)(v3647);
      const uint8_t* v3649 = v21 + 237;
      const int8_t* v3650 = (const int8_t*) v3649;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3651 = *(const int8_t *)(v3650);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3652 = __riscv_vwmacc_vx_i16m1(v3622, v3648, v3643, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3653 = __riscv_vwmacc_vx_i16m1(v3623, v3651, v3645, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3654 = v19 + 1952;
      const uint8_t* v3655 = (const uint8_t*) v3654;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3656 = __riscv_vle8_v_u8mf2(v3655, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3657 = __riscv_vand_vx_u8mf2(v3656, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3658 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3657);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3659 = __riscv_vsrl_vx_u8mf2(v3656, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3660 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3659);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3661 = v21 + 206;
      const int8_t* v3662 = (const int8_t*) v3661;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3663 = *(const int8_t *)(v3662);
      const uint8_t* v3664 = v21 + 238;
      const int8_t* v3665 = (const int8_t*) v3664;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3666 = *(const int8_t *)(v3665);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3667 = __riscv_vwmacc_vx_i16m1(v3637, v3663, v3658, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3668 = __riscv_vwmacc_vx_i16m1(v3638, v3666, v3660, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3669 = v19 + 1960;
      const uint8_t* v3670 = (const uint8_t*) v3669;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3671 = __riscv_vle8_v_u8mf2(v3670, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3672 = __riscv_vand_vx_u8mf2(v3671, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3673 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3672);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3674 = __riscv_vsrl_vx_u8mf2(v3671, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3675 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3674);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3676 = v21 + 206;
      const int8_t* v3677 = (const int8_t*) v3676;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3678 = *(const int8_t *)(v3677);
      const uint8_t* v3679 = v21 + 238;
      const int8_t* v3680 = (const int8_t*) v3679;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3681 = *(const int8_t *)(v3680);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3682 = __riscv_vwmacc_vx_i16m1(v3652, v3678, v3673, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3683 = __riscv_vwmacc_vx_i16m1(v3653, v3681, v3675, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3684 = v19 + 1968;
      const uint8_t* v3685 = (const uint8_t*) v3684;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3686 = __riscv_vle8_v_u8mf2(v3685, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3687 = __riscv_vand_vx_u8mf2(v3686, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3688 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3687);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3689 = __riscv_vsrl_vx_u8mf2(v3686, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3690 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3689);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3691 = v21 + 207;
      const int8_t* v3692 = (const int8_t*) v3691;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3693 = *(const int8_t *)(v3692);
      const uint8_t* v3694 = v21 + 239;
      const int8_t* v3695 = (const int8_t*) v3694;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3696 = *(const int8_t *)(v3695);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3697 = __riscv_vwmacc_vx_i16m1(v3667, v3693, v3688, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3698 = __riscv_vwmacc_vx_i16m1(v3668, v3696, v3690, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3699 = v19 + 1976;
      const uint8_t* v3700 = (const uint8_t*) v3699;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3701 = __riscv_vle8_v_u8mf2(v3700, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3702 = __riscv_vand_vx_u8mf2(v3701, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3703 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3702);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3704 = __riscv_vsrl_vx_u8mf2(v3701, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3705 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3704);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3706 = v21 + 207;
      const int8_t* v3707 = (const int8_t*) v3706;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3708 = *(const int8_t *)(v3707);
      const uint8_t* v3709 = v21 + 239;
      const int8_t* v3710 = (const int8_t*) v3709;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3711 = *(const int8_t *)(v3710);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3712 = __riscv_vwmacc_vx_i16m1(v3682, v3708, v3703, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3713 = __riscv_vwmacc_vx_i16m1(v3683, v3711, v3705, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3714 = v19 + 1984;
      const uint8_t* v3715 = (const uint8_t*) v3714;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3716 = __riscv_vle8_v_u8mf2(v3715, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3717 = __riscv_vand_vx_u8mf2(v3716, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3718 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3717);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3719 = __riscv_vsrl_vx_u8mf2(v3716, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3720 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3719);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3721 = v21 + 208;
      const int8_t* v3722 = (const int8_t*) v3721;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3723 = *(const int8_t *)(v3722);
      const uint8_t* v3724 = v21 + 240;
      const int8_t* v3725 = (const int8_t*) v3724;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3726 = *(const int8_t *)(v3725);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3727 = __riscv_vwmacc_vx_i16m1(v3697, v3723, v3718, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3728 = __riscv_vwmacc_vx_i16m1(v3698, v3726, v3720, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3729 = v19 + 1992;
      const uint8_t* v3730 = (const uint8_t*) v3729;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3731 = __riscv_vle8_v_u8mf2(v3730, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3732 = __riscv_vand_vx_u8mf2(v3731, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3733 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3732);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3734 = __riscv_vsrl_vx_u8mf2(v3731, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3735 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3734);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3736 = v21 + 208;
      const int8_t* v3737 = (const int8_t*) v3736;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3738 = *(const int8_t *)(v3737);
      const uint8_t* v3739 = v21 + 240;
      const int8_t* v3740 = (const int8_t*) v3739;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3741 = *(const int8_t *)(v3740);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3742 = __riscv_vwmacc_vx_i16m1(v3712, v3738, v3733, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3743 = __riscv_vwmacc_vx_i16m1(v3713, v3741, v3735, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3744 = v19 + 2000;
      const uint8_t* v3745 = (const uint8_t*) v3744;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3746 = __riscv_vle8_v_u8mf2(v3745, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3747 = __riscv_vand_vx_u8mf2(v3746, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3748 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3747);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3749 = __riscv_vsrl_vx_u8mf2(v3746, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3750 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3749);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3751 = v21 + 209;
      const int8_t* v3752 = (const int8_t*) v3751;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3753 = *(const int8_t *)(v3752);
      const uint8_t* v3754 = v21 + 241;
      const int8_t* v3755 = (const int8_t*) v3754;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3756 = *(const int8_t *)(v3755);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3757 = __riscv_vwmacc_vx_i16m1(v3727, v3753, v3748, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3758 = __riscv_vwmacc_vx_i16m1(v3728, v3756, v3750, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3759 = v19 + 2008;
      const uint8_t* v3760 = (const uint8_t*) v3759;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3761 = __riscv_vle8_v_u8mf2(v3760, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3762 = __riscv_vand_vx_u8mf2(v3761, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3763 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3762);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3764 = __riscv_vsrl_vx_u8mf2(v3761, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3765 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3764);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3766 = v21 + 209;
      const int8_t* v3767 = (const int8_t*) v3766;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3768 = *(const int8_t *)(v3767);
      const uint8_t* v3769 = v21 + 241;
      const int8_t* v3770 = (const int8_t*) v3769;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3771 = *(const int8_t *)(v3770);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3772 = __riscv_vwmacc_vx_i16m1(v3742, v3768, v3763, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3773 = __riscv_vwmacc_vx_i16m1(v3743, v3771, v3765, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3774 = v19 + 2016;
      const uint8_t* v3775 = (const uint8_t*) v3774;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3776 = __riscv_vle8_v_u8mf2(v3775, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3777 = __riscv_vand_vx_u8mf2(v3776, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3778 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3777);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3779 = __riscv_vsrl_vx_u8mf2(v3776, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3780 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3779);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3781 = v21 + 210;
      const int8_t* v3782 = (const int8_t*) v3781;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3783 = *(const int8_t *)(v3782);
      const uint8_t* v3784 = v21 + 242;
      const int8_t* v3785 = (const int8_t*) v3784;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3786 = *(const int8_t *)(v3785);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3787 = __riscv_vwmacc_vx_i16m1(v3757, v3783, v3778, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3788 = __riscv_vwmacc_vx_i16m1(v3758, v3786, v3780, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3789 = v19 + 2024;
      const uint8_t* v3790 = (const uint8_t*) v3789;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3791 = __riscv_vle8_v_u8mf2(v3790, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3792 = __riscv_vand_vx_u8mf2(v3791, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3793 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3792);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3794 = __riscv_vsrl_vx_u8mf2(v3791, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3795 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3794);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3796 = v21 + 210;
      const int8_t* v3797 = (const int8_t*) v3796;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3798 = *(const int8_t *)(v3797);
      const uint8_t* v3799 = v21 + 242;
      const int8_t* v3800 = (const int8_t*) v3799;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3801 = *(const int8_t *)(v3800);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3802 = __riscv_vwmacc_vx_i16m1(v3772, v3798, v3793, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3803 = __riscv_vwmacc_vx_i16m1(v3773, v3801, v3795, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3804 = v19 + 2032;
      const uint8_t* v3805 = (const uint8_t*) v3804;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3806 = __riscv_vle8_v_u8mf2(v3805, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3807 = __riscv_vand_vx_u8mf2(v3806, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3808 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3807);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3809 = __riscv_vsrl_vx_u8mf2(v3806, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3810 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3809);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3811 = v21 + 211;
      const int8_t* v3812 = (const int8_t*) v3811;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3813 = *(const int8_t *)(v3812);
      const uint8_t* v3814 = v21 + 243;
      const int8_t* v3815 = (const int8_t*) v3814;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3816 = *(const int8_t *)(v3815);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3817 = __riscv_vwmacc_vx_i16m1(v3787, v3813, v3808, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3818 = __riscv_vwmacc_vx_i16m1(v3788, v3816, v3810, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3819 = v19 + 2040;
      const uint8_t* v3820 = (const uint8_t*) v3819;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3821 = __riscv_vle8_v_u8mf2(v3820, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3822 = __riscv_vand_vx_u8mf2(v3821, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3823 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3822);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3824 = __riscv_vsrl_vx_u8mf2(v3821, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3825 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3824);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3826 = v21 + 211;
      const int8_t* v3827 = (const int8_t*) v3826;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3828 = *(const int8_t *)(v3827);
      const uint8_t* v3829 = v21 + 243;
      const int8_t* v3830 = (const int8_t*) v3829;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3831 = *(const int8_t *)(v3830);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3832 = __riscv_vwmacc_vx_i16m1(v3802, v3828, v3823, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3833 = __riscv_vwmacc_vx_i16m1(v3803, v3831, v3825, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v3834 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3835 = __riscv_vwmacc_vv_i32m2(v3834, v2238, v3817, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3836 = __riscv_vwmacc_vv_i32m2(v3835, v2255, v3818, 8);
      v34 = v3836;
      vint32m2_t v3837 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3838 = __riscv_vwmacc_vv_i32m2(v3837, v2306, v3832, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v3839 = __riscv_vwmacc_vv_i32m2(v3838, v2323, v3833, 8);
      v38 = v3839;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3840 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3841 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3842 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v3843 = __riscv_vmv_v_x_i16m1(0, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3844 = v19 + 2048;
      const uint8_t* v3845 = (const uint8_t*) v3844;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3846 = __riscv_vle8_v_u8mf2(v3845, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3847 = __riscv_vand_vx_u8mf2(v3846, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3848 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3847);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3849 = __riscv_vsrl_vx_u8mf2(v3846, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3850 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3849);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3851 = v21 + 212;
      const int8_t* v3852 = (const int8_t*) v3851;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3853 = *(const int8_t *)(v3852);
      const uint8_t* v3854 = v21 + 244;
      const int8_t* v3855 = (const int8_t*) v3854;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3856 = *(const int8_t *)(v3855);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3857 = __riscv_vwmacc_vx_i16m1(v3840, v3853, v3848, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3858 = __riscv_vwmacc_vx_i16m1(v3841, v3856, v3850, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3859 = v19 + 2056;
      const uint8_t* v3860 = (const uint8_t*) v3859;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3861 = __riscv_vle8_v_u8mf2(v3860, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3862 = __riscv_vand_vx_u8mf2(v3861, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3863 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3862);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3864 = __riscv_vsrl_vx_u8mf2(v3861, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3865 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3864);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3866 = v21 + 212;
      const int8_t* v3867 = (const int8_t*) v3866;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3868 = *(const int8_t *)(v3867);
      const uint8_t* v3869 = v21 + 244;
      const int8_t* v3870 = (const int8_t*) v3869;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3871 = *(const int8_t *)(v3870);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3872 = __riscv_vwmacc_vx_i16m1(v3842, v3868, v3863, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3873 = __riscv_vwmacc_vx_i16m1(v3843, v3871, v3865, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3874 = v19 + 2064;
      const uint8_t* v3875 = (const uint8_t*) v3874;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3876 = __riscv_vle8_v_u8mf2(v3875, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3877 = __riscv_vand_vx_u8mf2(v3876, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3878 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3877);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3879 = __riscv_vsrl_vx_u8mf2(v3876, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3880 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3879);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3881 = v21 + 213;
      const int8_t* v3882 = (const int8_t*) v3881;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3883 = *(const int8_t *)(v3882);
      const uint8_t* v3884 = v21 + 245;
      const int8_t* v3885 = (const int8_t*) v3884;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3886 = *(const int8_t *)(v3885);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3887 = __riscv_vwmacc_vx_i16m1(v3857, v3883, v3878, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3888 = __riscv_vwmacc_vx_i16m1(v3858, v3886, v3880, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3889 = v19 + 2072;
      const uint8_t* v3890 = (const uint8_t*) v3889;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3891 = __riscv_vle8_v_u8mf2(v3890, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3892 = __riscv_vand_vx_u8mf2(v3891, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3893 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3892);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3894 = __riscv_vsrl_vx_u8mf2(v3891, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3895 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3894);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3896 = v21 + 213;
      const int8_t* v3897 = (const int8_t*) v3896;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3898 = *(const int8_t *)(v3897);
      const uint8_t* v3899 = v21 + 245;
      const int8_t* v3900 = (const int8_t*) v3899;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3901 = *(const int8_t *)(v3900);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3902 = __riscv_vwmacc_vx_i16m1(v3872, v3898, v3893, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3903 = __riscv_vwmacc_vx_i16m1(v3873, v3901, v3895, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3904 = v19 + 2080;
      const uint8_t* v3905 = (const uint8_t*) v3904;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3906 = __riscv_vle8_v_u8mf2(v3905, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3907 = __riscv_vand_vx_u8mf2(v3906, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3908 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3907);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3909 = __riscv_vsrl_vx_u8mf2(v3906, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3910 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3909);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3911 = v21 + 214;
      const int8_t* v3912 = (const int8_t*) v3911;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3913 = *(const int8_t *)(v3912);
      const uint8_t* v3914 = v21 + 246;
      const int8_t* v3915 = (const int8_t*) v3914;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3916 = *(const int8_t *)(v3915);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3917 = __riscv_vwmacc_vx_i16m1(v3887, v3913, v3908, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3918 = __riscv_vwmacc_vx_i16m1(v3888, v3916, v3910, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3919 = v19 + 2088;
      const uint8_t* v3920 = (const uint8_t*) v3919;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3921 = __riscv_vle8_v_u8mf2(v3920, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3922 = __riscv_vand_vx_u8mf2(v3921, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3923 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3922);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3924 = __riscv_vsrl_vx_u8mf2(v3921, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3925 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3924);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3926 = v21 + 214;
      const int8_t* v3927 = (const int8_t*) v3926;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3928 = *(const int8_t *)(v3927);
      const uint8_t* v3929 = v21 + 246;
      const int8_t* v3930 = (const int8_t*) v3929;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3931 = *(const int8_t *)(v3930);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3932 = __riscv_vwmacc_vx_i16m1(v3902, v3928, v3923, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3933 = __riscv_vwmacc_vx_i16m1(v3903, v3931, v3925, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3934 = v19 + 2096;
      const uint8_t* v3935 = (const uint8_t*) v3934;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3936 = __riscv_vle8_v_u8mf2(v3935, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3937 = __riscv_vand_vx_u8mf2(v3936, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3938 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3937);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3939 = __riscv_vsrl_vx_u8mf2(v3936, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3940 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3939);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3941 = v21 + 215;
      const int8_t* v3942 = (const int8_t*) v3941;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3943 = *(const int8_t *)(v3942);
      const uint8_t* v3944 = v21 + 247;
      const int8_t* v3945 = (const int8_t*) v3944;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3946 = *(const int8_t *)(v3945);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3947 = __riscv_vwmacc_vx_i16m1(v3917, v3943, v3938, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3948 = __riscv_vwmacc_vx_i16m1(v3918, v3946, v3940, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3949 = v19 + 2104;
      const uint8_t* v3950 = (const uint8_t*) v3949;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3951 = __riscv_vle8_v_u8mf2(v3950, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3952 = __riscv_vand_vx_u8mf2(v3951, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3953 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3952);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3954 = __riscv_vsrl_vx_u8mf2(v3951, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3955 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3954);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3956 = v21 + 215;
      const int8_t* v3957 = (const int8_t*) v3956;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3958 = *(const int8_t *)(v3957);
      const uint8_t* v3959 = v21 + 247;
      const int8_t* v3960 = (const int8_t*) v3959;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3961 = *(const int8_t *)(v3960);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3962 = __riscv_vwmacc_vx_i16m1(v3932, v3958, v3953, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3963 = __riscv_vwmacc_vx_i16m1(v3933, v3961, v3955, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3964 = v19 + 2112;
      const uint8_t* v3965 = (const uint8_t*) v3964;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3966 = __riscv_vle8_v_u8mf2(v3965, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3967 = __riscv_vand_vx_u8mf2(v3966, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3968 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3967);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3969 = __riscv_vsrl_vx_u8mf2(v3966, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3970 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3969);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3971 = v21 + 216;
      const int8_t* v3972 = (const int8_t*) v3971;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3973 = *(const int8_t *)(v3972);
      const uint8_t* v3974 = v21 + 248;
      const int8_t* v3975 = (const int8_t*) v3974;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3976 = *(const int8_t *)(v3975);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3977 = __riscv_vwmacc_vx_i16m1(v3947, v3973, v3968, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3978 = __riscv_vwmacc_vx_i16m1(v3948, v3976, v3970, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3979 = v19 + 2120;
      const uint8_t* v3980 = (const uint8_t*) v3979;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3981 = __riscv_vle8_v_u8mf2(v3980, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3982 = __riscv_vand_vx_u8mf2(v3981, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3983 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3982);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3984 = __riscv_vsrl_vx_u8mf2(v3981, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3984);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v3986 = v21 + 216;
      const int8_t* v3987 = (const int8_t*) v3986;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3988 = *(const int8_t *)(v3987);
      const uint8_t* v3989 = v21 + 248;
      const int8_t* v3990 = (const int8_t*) v3989;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v3991 = *(const int8_t *)(v3990);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3992 = __riscv_vwmacc_vx_i16m1(v3962, v3988, v3983, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v3993 = __riscv_vwmacc_vx_i16m1(v3963, v3991, v3985, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v3994 = v19 + 2128;
      const uint8_t* v3995 = (const uint8_t*) v3994;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v3996 = __riscv_vle8_v_u8mf2(v3995, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v3997 = __riscv_vand_vx_u8mf2(v3996, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v3998 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3997);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v3999 = __riscv_vsrl_vx_u8mf2(v3996, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4000 = __riscv_vreinterpret_v_u8mf2_i8mf2(v3999);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4001 = v21 + 217;
      const int8_t* v4002 = (const int8_t*) v4001;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4003 = *(const int8_t *)(v4002);
      const uint8_t* v4004 = v21 + 249;
      const int8_t* v4005 = (const int8_t*) v4004;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4006 = *(const int8_t *)(v4005);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4007 = __riscv_vwmacc_vx_i16m1(v3977, v4003, v3998, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4008 = __riscv_vwmacc_vx_i16m1(v3978, v4006, v4000, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4009 = v19 + 2136;
      const uint8_t* v4010 = (const uint8_t*) v4009;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4011 = __riscv_vle8_v_u8mf2(v4010, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4012 = __riscv_vand_vx_u8mf2(v4011, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4013 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4012);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4014 = __riscv_vsrl_vx_u8mf2(v4011, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4015 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4014);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4016 = v21 + 217;
      const int8_t* v4017 = (const int8_t*) v4016;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4018 = *(const int8_t *)(v4017);
      const uint8_t* v4019 = v21 + 249;
      const int8_t* v4020 = (const int8_t*) v4019;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4021 = *(const int8_t *)(v4020);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4022 = __riscv_vwmacc_vx_i16m1(v3992, v4018, v4013, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4023 = __riscv_vwmacc_vx_i16m1(v3993, v4021, v4015, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4024 = v19 + 2144;
      const uint8_t* v4025 = (const uint8_t*) v4024;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4026 = __riscv_vle8_v_u8mf2(v4025, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4027 = __riscv_vand_vx_u8mf2(v4026, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4028 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4027);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4029 = __riscv_vsrl_vx_u8mf2(v4026, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4030 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4029);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4031 = v21 + 218;
      const int8_t* v4032 = (const int8_t*) v4031;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4033 = *(const int8_t *)(v4032);
      const uint8_t* v4034 = v21 + 250;
      const int8_t* v4035 = (const int8_t*) v4034;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4036 = *(const int8_t *)(v4035);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4037 = __riscv_vwmacc_vx_i16m1(v4007, v4033, v4028, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4038 = __riscv_vwmacc_vx_i16m1(v4008, v4036, v4030, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4039 = v19 + 2152;
      const uint8_t* v4040 = (const uint8_t*) v4039;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4041 = __riscv_vle8_v_u8mf2(v4040, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4042 = __riscv_vand_vx_u8mf2(v4041, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4043 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4042);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4044 = __riscv_vsrl_vx_u8mf2(v4041, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4045 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4044);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4046 = v21 + 218;
      const int8_t* v4047 = (const int8_t*) v4046;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4048 = *(const int8_t *)(v4047);
      const uint8_t* v4049 = v21 + 250;
      const int8_t* v4050 = (const int8_t*) v4049;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4051 = *(const int8_t *)(v4050);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4052 = __riscv_vwmacc_vx_i16m1(v4022, v4048, v4043, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4053 = __riscv_vwmacc_vx_i16m1(v4023, v4051, v4045, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4054 = v19 + 2160;
      const uint8_t* v4055 = (const uint8_t*) v4054;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4056 = __riscv_vle8_v_u8mf2(v4055, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4057 = __riscv_vand_vx_u8mf2(v4056, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4058 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4057);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4059 = __riscv_vsrl_vx_u8mf2(v4056, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4060 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4059);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4061 = v21 + 219;
      const int8_t* v4062 = (const int8_t*) v4061;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4063 = *(const int8_t *)(v4062);
      const uint8_t* v4064 = v21 + 251;
      const int8_t* v4065 = (const int8_t*) v4064;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4066 = *(const int8_t *)(v4065);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4067 = __riscv_vwmacc_vx_i16m1(v4037, v4063, v4058, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4068 = __riscv_vwmacc_vx_i16m1(v4038, v4066, v4060, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4069 = v19 + 2168;
      const uint8_t* v4070 = (const uint8_t*) v4069;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4071 = __riscv_vle8_v_u8mf2(v4070, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4072 = __riscv_vand_vx_u8mf2(v4071, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4073 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4072);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4074 = __riscv_vsrl_vx_u8mf2(v4071, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4075 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4074);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4076 = v21 + 219;
      const int8_t* v4077 = (const int8_t*) v4076;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4078 = *(const int8_t *)(v4077);
      const uint8_t* v4079 = v21 + 251;
      const int8_t* v4080 = (const int8_t*) v4079;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4081 = *(const int8_t *)(v4080);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4082 = __riscv_vwmacc_vx_i16m1(v4052, v4078, v4073, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4083 = __riscv_vwmacc_vx_i16m1(v4053, v4081, v4075, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4084 = v19 + 2176;
      const uint8_t* v4085 = (const uint8_t*) v4084;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4086 = __riscv_vle8_v_u8mf2(v4085, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4087 = __riscv_vand_vx_u8mf2(v4086, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4088 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4087);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4089 = __riscv_vsrl_vx_u8mf2(v4086, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4090 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4089);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4091 = v21 + 220;
      const int8_t* v4092 = (const int8_t*) v4091;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4093 = *(const int8_t *)(v4092);
      const uint8_t* v4094 = v21 + 252;
      const int8_t* v4095 = (const int8_t*) v4094;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4096 = *(const int8_t *)(v4095);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4097 = __riscv_vwmacc_vx_i16m1(v4067, v4093, v4088, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4098 = __riscv_vwmacc_vx_i16m1(v4068, v4096, v4090, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4099 = v19 + 2184;
      const uint8_t* v4100 = (const uint8_t*) v4099;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4101 = __riscv_vle8_v_u8mf2(v4100, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4102 = __riscv_vand_vx_u8mf2(v4101, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4103 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4102);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4104 = __riscv_vsrl_vx_u8mf2(v4101, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4105 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4104);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4106 = v21 + 220;
      const int8_t* v4107 = (const int8_t*) v4106;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4108 = *(const int8_t *)(v4107);
      const uint8_t* v4109 = v21 + 252;
      const int8_t* v4110 = (const int8_t*) v4109;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4111 = *(const int8_t *)(v4110);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4112 = __riscv_vwmacc_vx_i16m1(v4082, v4108, v4103, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4113 = __riscv_vwmacc_vx_i16m1(v4083, v4111, v4105, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4114 = v19 + 2192;
      const uint8_t* v4115 = (const uint8_t*) v4114;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4116 = __riscv_vle8_v_u8mf2(v4115, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4117 = __riscv_vand_vx_u8mf2(v4116, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4118 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4117);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4119 = __riscv_vsrl_vx_u8mf2(v4116, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4120 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4119);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4121 = v21 + 221;
      const int8_t* v4122 = (const int8_t*) v4121;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4123 = *(const int8_t *)(v4122);
      const uint8_t* v4124 = v21 + 253;
      const int8_t* v4125 = (const int8_t*) v4124;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4126 = *(const int8_t *)(v4125);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4127 = __riscv_vwmacc_vx_i16m1(v4097, v4123, v4118, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4128 = __riscv_vwmacc_vx_i16m1(v4098, v4126, v4120, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4129 = v19 + 2200;
      const uint8_t* v4130 = (const uint8_t*) v4129;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4131 = __riscv_vle8_v_u8mf2(v4130, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4132 = __riscv_vand_vx_u8mf2(v4131, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4133 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4132);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4134 = __riscv_vsrl_vx_u8mf2(v4131, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4135 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4134);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4136 = v21 + 221;
      const int8_t* v4137 = (const int8_t*) v4136;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4138 = *(const int8_t *)(v4137);
      const uint8_t* v4139 = v21 + 253;
      const int8_t* v4140 = (const int8_t*) v4139;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4141 = *(const int8_t *)(v4140);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4142 = __riscv_vwmacc_vx_i16m1(v4112, v4138, v4133, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4143 = __riscv_vwmacc_vx_i16m1(v4113, v4141, v4135, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4144 = v19 + 2208;
      const uint8_t* v4145 = (const uint8_t*) v4144;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4146 = __riscv_vle8_v_u8mf2(v4145, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4147 = __riscv_vand_vx_u8mf2(v4146, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4148 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4147);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4149 = __riscv_vsrl_vx_u8mf2(v4146, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4150 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4149);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4151 = v21 + 222;
      const int8_t* v4152 = (const int8_t*) v4151;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4153 = *(const int8_t *)(v4152);
      const uint8_t* v4154 = v21 + 254;
      const int8_t* v4155 = (const int8_t*) v4154;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4156 = *(const int8_t *)(v4155);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4157 = __riscv_vwmacc_vx_i16m1(v4127, v4153, v4148, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4158 = __riscv_vwmacc_vx_i16m1(v4128, v4156, v4150, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4159 = v19 + 2216;
      const uint8_t* v4160 = (const uint8_t*) v4159;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4161 = __riscv_vle8_v_u8mf2(v4160, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4162 = __riscv_vand_vx_u8mf2(v4161, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4163 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4162);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4164 = __riscv_vsrl_vx_u8mf2(v4161, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4165 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4164);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4166 = v21 + 222;
      const int8_t* v4167 = (const int8_t*) v4166;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4168 = *(const int8_t *)(v4167);
      const uint8_t* v4169 = v21 + 254;
      const int8_t* v4170 = (const int8_t*) v4169;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4171 = *(const int8_t *)(v4170);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4172 = __riscv_vwmacc_vx_i16m1(v4142, v4168, v4163, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4173 = __riscv_vwmacc_vx_i16m1(v4143, v4171, v4165, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4174 = v19 + 2224;
      const uint8_t* v4175 = (const uint8_t*) v4174;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4176 = __riscv_vle8_v_u8mf2(v4175, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4177 = __riscv_vand_vx_u8mf2(v4176, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4178 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4177);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4179 = __riscv_vsrl_vx_u8mf2(v4176, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4180 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4179);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4181 = v21 + 223;
      const int8_t* v4182 = (const int8_t*) v4181;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4183 = *(const int8_t *)(v4182);
      const uint8_t* v4184 = v21 + 255;
      const int8_t* v4185 = (const int8_t*) v4184;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4186 = *(const int8_t *)(v4185);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4187 = __riscv_vwmacc_vx_i16m1(v4157, v4183, v4178, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4188 = __riscv_vwmacc_vx_i16m1(v4158, v4186, v4180, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4189 = v19 + 2232;
      const uint8_t* v4190 = (const uint8_t*) v4189;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4191 = __riscv_vle8_v_u8mf2(v4190, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4192 = __riscv_vand_vx_u8mf2(v4191, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4193 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4192);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4194 = __riscv_vsrl_vx_u8mf2(v4191, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4195 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4194);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4196 = v21 + 223;
      const int8_t* v4197 = (const int8_t*) v4196;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4198 = *(const int8_t *)(v4197);
      const uint8_t* v4199 = v21 + 255;
      const int8_t* v4200 = (const int8_t*) v4199;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4201 = *(const int8_t *)(v4200);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4202 = __riscv_vwmacc_vx_i16m1(v4172, v4198, v4193, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4203 = __riscv_vwmacc_vx_i16m1(v4173, v4201, v4195, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4204 = v19 + 2240;
      const uint8_t* v4205 = (const uint8_t*) v4204;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4206 = __riscv_vle8_v_u8mf2(v4205, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4207 = __riscv_vand_vx_u8mf2(v4206, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4208 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4207);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4209 = __riscv_vsrl_vx_u8mf2(v4206, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4210 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4209);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4211 = v21 + 224;
      const int8_t* v4212 = (const int8_t*) v4211;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4213 = *(const int8_t *)(v4212);
      const uint8_t* v4214 = v21 + 256;
      const int8_t* v4215 = (const int8_t*) v4214;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4216 = *(const int8_t *)(v4215);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4217 = __riscv_vwmacc_vx_i16m1(v4187, v4213, v4208, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4218 = __riscv_vwmacc_vx_i16m1(v4188, v4216, v4210, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4219 = v19 + 2248;
      const uint8_t* v4220 = (const uint8_t*) v4219;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4221 = __riscv_vle8_v_u8mf2(v4220, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4222 = __riscv_vand_vx_u8mf2(v4221, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4223 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4222);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4224 = __riscv_vsrl_vx_u8mf2(v4221, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4225 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4224);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4226 = v21 + 224;
      const int8_t* v4227 = (const int8_t*) v4226;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4228 = *(const int8_t *)(v4227);
      const uint8_t* v4229 = v21 + 256;
      const int8_t* v4230 = (const int8_t*) v4229;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4231 = *(const int8_t *)(v4230);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4232 = __riscv_vwmacc_vx_i16m1(v4202, v4228, v4223, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4233 = __riscv_vwmacc_vx_i16m1(v4203, v4231, v4225, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4234 = v19 + 2256;
      const uint8_t* v4235 = (const uint8_t*) v4234;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4236 = __riscv_vle8_v_u8mf2(v4235, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4237 = __riscv_vand_vx_u8mf2(v4236, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4238 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4237);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4239 = __riscv_vsrl_vx_u8mf2(v4236, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4240 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4239);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4241 = v21 + 225;
      const int8_t* v4242 = (const int8_t*) v4241;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4243 = *(const int8_t *)(v4242);
      const uint8_t* v4244 = v21 + 257;
      const int8_t* v4245 = (const int8_t*) v4244;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4246 = *(const int8_t *)(v4245);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4247 = __riscv_vwmacc_vx_i16m1(v4217, v4243, v4238, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4248 = __riscv_vwmacc_vx_i16m1(v4218, v4246, v4240, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4249 = v19 + 2264;
      const uint8_t* v4250 = (const uint8_t*) v4249;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4251 = __riscv_vle8_v_u8mf2(v4250, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4252 = __riscv_vand_vx_u8mf2(v4251, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4253 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4252);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4254 = __riscv_vsrl_vx_u8mf2(v4251, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4255 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4254);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4256 = v21 + 225;
      const int8_t* v4257 = (const int8_t*) v4256;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4258 = *(const int8_t *)(v4257);
      const uint8_t* v4259 = v21 + 257;
      const int8_t* v4260 = (const int8_t*) v4259;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4261 = *(const int8_t *)(v4260);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4262 = __riscv_vwmacc_vx_i16m1(v4232, v4258, v4253, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4263 = __riscv_vwmacc_vx_i16m1(v4233, v4261, v4255, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4264 = v19 + 2272;
      const uint8_t* v4265 = (const uint8_t*) v4264;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4266 = __riscv_vle8_v_u8mf2(v4265, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4267 = __riscv_vand_vx_u8mf2(v4266, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4268 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4267);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4269 = __riscv_vsrl_vx_u8mf2(v4266, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4270 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4269);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4271 = v21 + 226;
      const int8_t* v4272 = (const int8_t*) v4271;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4273 = *(const int8_t *)(v4272);
      const uint8_t* v4274 = v21 + 258;
      const int8_t* v4275 = (const int8_t*) v4274;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4276 = *(const int8_t *)(v4275);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4277 = __riscv_vwmacc_vx_i16m1(v4247, v4273, v4268, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4278 = __riscv_vwmacc_vx_i16m1(v4248, v4276, v4270, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4279 = v19 + 2280;
      const uint8_t* v4280 = (const uint8_t*) v4279;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4281 = __riscv_vle8_v_u8mf2(v4280, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4282 = __riscv_vand_vx_u8mf2(v4281, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4283 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4282);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4284 = __riscv_vsrl_vx_u8mf2(v4281, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4285 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4284);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4286 = v21 + 226;
      const int8_t* v4287 = (const int8_t*) v4286;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4288 = *(const int8_t *)(v4287);
      const uint8_t* v4289 = v21 + 258;
      const int8_t* v4290 = (const int8_t*) v4289;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4291 = *(const int8_t *)(v4290);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4292 = __riscv_vwmacc_vx_i16m1(v4262, v4288, v4283, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4293 = __riscv_vwmacc_vx_i16m1(v4263, v4291, v4285, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4294 = v19 + 2288;
      const uint8_t* v4295 = (const uint8_t*) v4294;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4296 = __riscv_vle8_v_u8mf2(v4295, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4297 = __riscv_vand_vx_u8mf2(v4296, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4298 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4297);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4299 = __riscv_vsrl_vx_u8mf2(v4296, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4300 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4299);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4301 = v21 + 227;
      const int8_t* v4302 = (const int8_t*) v4301;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4303 = *(const int8_t *)(v4302);
      const uint8_t* v4304 = v21 + 259;
      const int8_t* v4305 = (const int8_t*) v4304;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4306 = *(const int8_t *)(v4305);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4307 = __riscv_vwmacc_vx_i16m1(v4277, v4303, v4298, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4308 = __riscv_vwmacc_vx_i16m1(v4278, v4306, v4300, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v4309 = v19 + 2296;
      const uint8_t* v4310 = (const uint8_t*) v4309;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v4311 = __riscv_vle8_v_u8mf2(v4310, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v4312 = __riscv_vand_vx_u8mf2(v4311, 0x0F, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4313 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4312);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v4314 = __riscv_vsrl_vx_u8mf2(v4311, 4, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v4315 = __riscv_vreinterpret_v_u8mf2_i8mf2(v4314);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v4316 = v21 + 227;
      const int8_t* v4317 = (const int8_t*) v4316;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4318 = *(const int8_t *)(v4317);
      const uint8_t* v4319 = v21 + 259;
      const int8_t* v4320 = (const int8_t*) v4319;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v4321 = *(const int8_t *)(v4320);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4322 = __riscv_vwmacc_vx_i16m1(v4292, v4318, v4313, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v4323 = __riscv_vwmacc_vx_i16m1(v4293, v4321, v4315, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v4324 = v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4325 = __riscv_vwmacc_vv_i32m2(v4324, v2238, v4307, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4326 = __riscv_vwmacc_vv_i32m2(v4325, v2255, v4308, 8);
      v34 = v4326;
      vint32m2_t v4327 = v38;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4328 = __riscv_vwmacc_vv_i32m2(v4327, v2306, v4322, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v4329 = __riscv_vwmacc_vv_i32m2(v4328, v2323, v4323, 8);
      v38 = v4329;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v4330 = (const _Float16*) v19;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v4331 = __riscv_vle16_v_f16m1(v4330, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v4332 = __riscv_vfwcvt_f_f_v_f32m2(v4331, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v4333 = __riscv_vfmul_vf_f32m2(v4332, v23, 8);
      vint32m2_t v4334 = v34;
      vfloat32m2_t v4335 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4336 = __riscv_vfcvt_f_x_v_f32m2(v4334, 8);
      vfloat32m2_t v4337 = __riscv_vfmacc_vv_f32m2(v4335, v4336, v4333, 8);
      vint32m2_t v4338 = v36;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4339 = __riscv_vfcvt_f_x_v_f32m2(v4338, 8);
      vfloat32m2_t v4340 = __riscv_vfnmsac_vv_f32m2(v4337, v28, v4339, 8);
      v13 = v4340;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v4341 = v19 + 16;
      const _Float16* v4342 = (const _Float16*) v4341;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v4343 = __riscv_vle16_v_f16m1(v4342, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v4344 = __riscv_vfwcvt_f_f_v_f32m2(v4343, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v4345 = __riscv_vfmul_vf_f32m2(v4344, v23, 8);
      vint32m2_t v4346 = v38;
      vfloat32m2_t v4347 = v15;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4348 = __riscv_vfcvt_f_x_v_f32m2(v4346, 8);
      vfloat32m2_t v4349 = __riscv_vfmacc_vv_f32m2(v4347, v4348, v4345, 8);
      vint32m2_t v4350 = v40;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v4351 = __riscv_vfcvt_f_x_v_f32m2(v4350, 8);
      vfloat32m2_t v4352 = __riscv_vfnmsac_vv_f32m2(v4349, v33, v4351, 8);
      v15 = v4352;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v4353 = v9 * 16;
    float* v4354 = v2 + v4353;
    vfloat32m2_t v4355 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v4354, v4355, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v4356 = v9 * 16;
    size_t v4357 = v4356 + 8;
    float* v4358 = v2 + v4357;
    vfloat32m2_t v4359 = v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v4358, v4359, 8);
  }
  vint32m1_t v4360 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


