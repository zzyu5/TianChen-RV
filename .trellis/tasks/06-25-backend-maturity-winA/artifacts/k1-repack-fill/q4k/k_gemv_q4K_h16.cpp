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
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
    v13 = v14;
    for (size_t v15 = 0; v15 < v7; v15 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v16 = v15 * 2304;
      const uint8_t* v17 = v12 + v16;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v18 = v15 * 292;
      const uint8_t* v19 = v4 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const float* v20 = (const float*) v19;
      float v21 = *(const float *)(v20);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v22 = v17 + 32;
      const _Float16* v23 = (const _Float16*) v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v24 = __riscv_vle16_v_f16m1(v23, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v25 = __riscv_vfwcvt_f_f_v_f32m2(v24, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v26 = __riscv_vfmul_vf_f32m2(v25, v21, 16);
      vint32m2_t v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v28 = __riscv_vmv_v_x_i32m2(0, 16);
      v27 = v28;
      vint32m2_t v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v30 = __riscv_vmv_v_x_i32m2(0, 16);
      v29 = v30;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v31 = v17 + 64;
      const uint8_t* v32 = (const uint8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v33 = __riscv_vle8_v_u8mf2(v32, 16);
      const uint8_t* v34 = v17 + 192;
      const uint8_t* v35 = (const uint8_t*) v34;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v36 = __riscv_vle8_v_u8mf2(v35, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v37 = __riscv_vand_vx_u8mf2(v33, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v38 = __riscv_vsrl_vx_u8mf2(v33, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v39 = __riscv_vand_vx_u8mf2(v36, 0x03, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v40 = __riscv_vsll_vx_u8mf2(v39, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v41 = __riscv_vand_vx_u8mf2(v36, 0x0C, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v42 = __riscv_vsll_vx_u8mf2(v41, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v43 = __riscv_vor_vv_u8mf2(v40, v37, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v44 = __riscv_vor_vv_u8mf2(v42, v38, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v45 = __riscv_vzext_vf2_u16m1(v43, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v46 = __riscv_vreinterpret_v_u16m1_i16m1(v45);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v47 = __riscv_vzext_vf2_u16m1(v44, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v48 = __riscv_vreinterpret_v_u16m1_i16m1(v47);
      const uint8_t* v49 = v17 + 80;
      const uint8_t* v50 = (const uint8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v51 = __riscv_vle8_v_u8mf2(v50, 16);
      const uint8_t* v52 = v17 + 208;
      const uint8_t* v53 = (const uint8_t*) v52;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v54 = __riscv_vle8_v_u8mf2(v53, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v55 = __riscv_vand_vx_u8mf2(v51, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v56 = __riscv_vsrl_vx_u8mf2(v51, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v57 = __riscv_vand_vx_u8mf2(v54, 0x03, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v58 = __riscv_vsll_vx_u8mf2(v57, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v59 = __riscv_vand_vx_u8mf2(v54, 0x0C, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v60 = __riscv_vsll_vx_u8mf2(v59, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v61 = __riscv_vor_vv_u8mf2(v58, v55, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v62 = __riscv_vor_vv_u8mf2(v60, v56, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v63 = __riscv_vzext_vf2_u16m1(v61, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v64 = __riscv_vreinterpret_v_u16m1_i16m1(v63);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v65 = __riscv_vzext_vf2_u16m1(v62, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v66 = __riscv_vreinterpret_v_u16m1_i16m1(v65);
      const uint8_t* v67 = v17 + 96;
      const uint8_t* v68 = (const uint8_t*) v67;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v69 = __riscv_vle8_v_u8mf2(v68, 16);
      const uint8_t* v70 = v17 + 224;
      const uint8_t* v71 = (const uint8_t*) v70;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v72 = __riscv_vle8_v_u8mf2(v71, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v73 = __riscv_vand_vx_u8mf2(v69, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v74 = __riscv_vsrl_vx_u8mf2(v69, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v75 = __riscv_vand_vx_u8mf2(v72, 0x03, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v76 = __riscv_vsll_vx_u8mf2(v75, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v77 = __riscv_vand_vx_u8mf2(v72, 0x0C, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v78 = __riscv_vsll_vx_u8mf2(v77, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v79 = __riscv_vor_vv_u8mf2(v76, v73, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v80 = __riscv_vor_vv_u8mf2(v78, v74, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v81 = __riscv_vzext_vf2_u16m1(v79, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v82 = __riscv_vreinterpret_v_u16m1_i16m1(v81);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v83 = __riscv_vzext_vf2_u16m1(v80, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v84 = __riscv_vreinterpret_v_u16m1_i16m1(v83);
      const uint8_t* v85 = v17 + 112;
      const uint8_t* v86 = (const uint8_t*) v85;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v87 = __riscv_vle8_v_u8mf2(v86, 16);
      const uint8_t* v88 = v17 + 240;
      const uint8_t* v89 = (const uint8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v90 = __riscv_vle8_v_u8mf2(v89, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v91 = __riscv_vand_vx_u8mf2(v87, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v92 = __riscv_vsrl_vx_u8mf2(v87, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v93 = __riscv_vand_vx_u8mf2(v90, 0x03, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v94 = __riscv_vsll_vx_u8mf2(v93, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v95 = __riscv_vand_vx_u8mf2(v90, 0x0C, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8mf2
      vuint8mf2_t v96 = __riscv_vsll_vx_u8mf2(v95, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v97 = __riscv_vor_vv_u8mf2(v94, v91, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v98 = __riscv_vor_vv_u8mf2(v96, v92, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v99 = __riscv_vzext_vf2_u16m1(v97, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v100 = __riscv_vreinterpret_v_u16m1_i16m1(v99);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v101 = __riscv_vzext_vf2_u16m1(v98, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v102 = __riscv_vreinterpret_v_u16m1_i16m1(v101);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v103 = v19 + 260;
      const int16_t* v104 = (const int16_t*) v103;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v105 = *(const int16_t *)(v104);
      const uint8_t* v106 = v19 + 262;
      const int16_t* v107 = (const int16_t*) v106;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v108 = *(const int16_t *)(v107);
      int32_t v109 = v105 + v108;
      vint32m2_t v110 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v111 = __riscv_vwmacc_vx_i32m2(v110, v109, v48, 16);
      v29 = v111;
      const uint8_t* v112 = v19 + 264;
      const int16_t* v113 = (const int16_t*) v112;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v114 = *(const int16_t *)(v113);
      const uint8_t* v115 = v19 + 266;
      const int16_t* v116 = (const int16_t*) v115;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v117 = *(const int16_t *)(v116);
      int32_t v118 = v114 + v117;
      vint32m2_t v119 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v120 = __riscv_vwmacc_vx_i32m2(v119, v118, v66, 16);
      v29 = v120;
      const uint8_t* v121 = v19 + 268;
      const int16_t* v122 = (const int16_t*) v121;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v123 = *(const int16_t *)(v122);
      const uint8_t* v124 = v19 + 270;
      const int16_t* v125 = (const int16_t*) v124;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v126 = *(const int16_t *)(v125);
      int32_t v127 = v123 + v126;
      vint32m2_t v128 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v129 = __riscv_vwmacc_vx_i32m2(v128, v127, v84, 16);
      v29 = v129;
      const uint8_t* v130 = v19 + 272;
      const int16_t* v131 = (const int16_t*) v130;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v132 = *(const int16_t *)(v131);
      const uint8_t* v133 = v19 + 274;
      const int16_t* v134 = (const int16_t*) v133;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v135 = *(const int16_t *)(v134);
      int32_t v136 = v132 + v135;
      vint32m2_t v137 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v138 = __riscv_vwmacc_vx_i32m2(v137, v136, v102, 16);
      v29 = v138;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v139 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v140 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v141 = v17 + 256;
      const uint8_t* v142 = (const uint8_t*) v141;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v143 = __riscv_vle8_v_u8mf2(v142, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v144 = __riscv_vand_vx_u8mf2(v143, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v145 = __riscv_vreinterpret_v_u8mf2_i8mf2(v144);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v146 = __riscv_vsrl_vx_u8mf2(v143, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v147 = __riscv_vreinterpret_v_u8mf2_i8mf2(v146);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v148 = v19 + 4;
      const int8_t* v149 = (const int8_t*) v148;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v150 = *(const int8_t *)(v149);
      const uint8_t* v151 = v19 + 36;
      const int8_t* v152 = (const int8_t*) v151;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v153 = *(const int8_t *)(v152);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v154 = __riscv_vwmacc_vx_i16m1(v139, v150, v145, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v155 = __riscv_vwmacc_vx_i16m1(v140, v153, v147, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v156 = v17 + 272;
      const uint8_t* v157 = (const uint8_t*) v156;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v158 = __riscv_vle8_v_u8mf2(v157, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v159 = __riscv_vand_vx_u8mf2(v158, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v160 = __riscv_vreinterpret_v_u8mf2_i8mf2(v159);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v161 = __riscv_vsrl_vx_u8mf2(v158, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v162 = __riscv_vreinterpret_v_u8mf2_i8mf2(v161);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v163 = v19 + 5;
      const int8_t* v164 = (const int8_t*) v163;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v165 = *(const int8_t *)(v164);
      const uint8_t* v166 = v19 + 37;
      const int8_t* v167 = (const int8_t*) v166;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v168 = *(const int8_t *)(v167);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v169 = __riscv_vwmacc_vx_i16m1(v154, v165, v160, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v170 = __riscv_vwmacc_vx_i16m1(v155, v168, v162, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v171 = v17 + 288;
      const uint8_t* v172 = (const uint8_t*) v171;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v173 = __riscv_vle8_v_u8mf2(v172, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v174 = __riscv_vand_vx_u8mf2(v173, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v175 = __riscv_vreinterpret_v_u8mf2_i8mf2(v174);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v176 = __riscv_vsrl_vx_u8mf2(v173, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v177 = __riscv_vreinterpret_v_u8mf2_i8mf2(v176);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v178 = v19 + 6;
      const int8_t* v179 = (const int8_t*) v178;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v180 = *(const int8_t *)(v179);
      const uint8_t* v181 = v19 + 38;
      const int8_t* v182 = (const int8_t*) v181;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v183 = *(const int8_t *)(v182);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v184 = __riscv_vwmacc_vx_i16m1(v169, v180, v175, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v185 = __riscv_vwmacc_vx_i16m1(v170, v183, v177, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v186 = v17 + 304;
      const uint8_t* v187 = (const uint8_t*) v186;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v188 = __riscv_vle8_v_u8mf2(v187, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v189 = __riscv_vand_vx_u8mf2(v188, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v190 = __riscv_vreinterpret_v_u8mf2_i8mf2(v189);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v191 = __riscv_vsrl_vx_u8mf2(v188, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v192 = __riscv_vreinterpret_v_u8mf2_i8mf2(v191);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v193 = v19 + 7;
      const int8_t* v194 = (const int8_t*) v193;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v195 = *(const int8_t *)(v194);
      const uint8_t* v196 = v19 + 39;
      const int8_t* v197 = (const int8_t*) v196;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v198 = *(const int8_t *)(v197);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v199 = __riscv_vwmacc_vx_i16m1(v184, v195, v190, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v200 = __riscv_vwmacc_vx_i16m1(v185, v198, v192, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v201 = v17 + 320;
      const uint8_t* v202 = (const uint8_t*) v201;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v203 = __riscv_vle8_v_u8mf2(v202, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v204 = __riscv_vand_vx_u8mf2(v203, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v205 = __riscv_vreinterpret_v_u8mf2_i8mf2(v204);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v206 = __riscv_vsrl_vx_u8mf2(v203, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v207 = __riscv_vreinterpret_v_u8mf2_i8mf2(v206);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v208 = v19 + 8;
      const int8_t* v209 = (const int8_t*) v208;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v210 = *(const int8_t *)(v209);
      const uint8_t* v211 = v19 + 40;
      const int8_t* v212 = (const int8_t*) v211;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v213 = *(const int8_t *)(v212);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v214 = __riscv_vwmacc_vx_i16m1(v199, v210, v205, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v215 = __riscv_vwmacc_vx_i16m1(v200, v213, v207, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v216 = v17 + 336;
      const uint8_t* v217 = (const uint8_t*) v216;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v218 = __riscv_vle8_v_u8mf2(v217, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v219 = __riscv_vand_vx_u8mf2(v218, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v220 = __riscv_vreinterpret_v_u8mf2_i8mf2(v219);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v221 = __riscv_vsrl_vx_u8mf2(v218, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v222 = __riscv_vreinterpret_v_u8mf2_i8mf2(v221);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v223 = v19 + 9;
      const int8_t* v224 = (const int8_t*) v223;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v225 = *(const int8_t *)(v224);
      const uint8_t* v226 = v19 + 41;
      const int8_t* v227 = (const int8_t*) v226;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v228 = *(const int8_t *)(v227);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v229 = __riscv_vwmacc_vx_i16m1(v214, v225, v220, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v230 = __riscv_vwmacc_vx_i16m1(v215, v228, v222, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v231 = v17 + 352;
      const uint8_t* v232 = (const uint8_t*) v231;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v233 = __riscv_vle8_v_u8mf2(v232, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v234 = __riscv_vand_vx_u8mf2(v233, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v235 = __riscv_vreinterpret_v_u8mf2_i8mf2(v234);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v236 = __riscv_vsrl_vx_u8mf2(v233, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v237 = __riscv_vreinterpret_v_u8mf2_i8mf2(v236);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v238 = v19 + 10;
      const int8_t* v239 = (const int8_t*) v238;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v240 = *(const int8_t *)(v239);
      const uint8_t* v241 = v19 + 42;
      const int8_t* v242 = (const int8_t*) v241;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v243 = *(const int8_t *)(v242);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v244 = __riscv_vwmacc_vx_i16m1(v229, v240, v235, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v245 = __riscv_vwmacc_vx_i16m1(v230, v243, v237, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v246 = v17 + 368;
      const uint8_t* v247 = (const uint8_t*) v246;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v248 = __riscv_vle8_v_u8mf2(v247, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v249 = __riscv_vand_vx_u8mf2(v248, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v250 = __riscv_vreinterpret_v_u8mf2_i8mf2(v249);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v251 = __riscv_vsrl_vx_u8mf2(v248, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v252 = __riscv_vreinterpret_v_u8mf2_i8mf2(v251);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v253 = v19 + 11;
      const int8_t* v254 = (const int8_t*) v253;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v255 = *(const int8_t *)(v254);
      const uint8_t* v256 = v19 + 43;
      const int8_t* v257 = (const int8_t*) v256;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v258 = *(const int8_t *)(v257);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v259 = __riscv_vwmacc_vx_i16m1(v244, v255, v250, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v260 = __riscv_vwmacc_vx_i16m1(v245, v258, v252, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v261 = v17 + 384;
      const uint8_t* v262 = (const uint8_t*) v261;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v263 = __riscv_vle8_v_u8mf2(v262, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v264 = __riscv_vand_vx_u8mf2(v263, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v265 = __riscv_vreinterpret_v_u8mf2_i8mf2(v264);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v266 = __riscv_vsrl_vx_u8mf2(v263, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v267 = __riscv_vreinterpret_v_u8mf2_i8mf2(v266);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v268 = v19 + 12;
      const int8_t* v269 = (const int8_t*) v268;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v270 = *(const int8_t *)(v269);
      const uint8_t* v271 = v19 + 44;
      const int8_t* v272 = (const int8_t*) v271;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v273 = *(const int8_t *)(v272);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v274 = __riscv_vwmacc_vx_i16m1(v259, v270, v265, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v275 = __riscv_vwmacc_vx_i16m1(v260, v273, v267, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v276 = v17 + 400;
      const uint8_t* v277 = (const uint8_t*) v276;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v278 = __riscv_vle8_v_u8mf2(v277, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v279 = __riscv_vand_vx_u8mf2(v278, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v280 = __riscv_vreinterpret_v_u8mf2_i8mf2(v279);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v281 = __riscv_vsrl_vx_u8mf2(v278, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v282 = __riscv_vreinterpret_v_u8mf2_i8mf2(v281);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v283 = v19 + 13;
      const int8_t* v284 = (const int8_t*) v283;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v285 = *(const int8_t *)(v284);
      const uint8_t* v286 = v19 + 45;
      const int8_t* v287 = (const int8_t*) v286;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v288 = *(const int8_t *)(v287);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v289 = __riscv_vwmacc_vx_i16m1(v274, v285, v280, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v290 = __riscv_vwmacc_vx_i16m1(v275, v288, v282, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v291 = v17 + 416;
      const uint8_t* v292 = (const uint8_t*) v291;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v293 = __riscv_vle8_v_u8mf2(v292, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v294 = __riscv_vand_vx_u8mf2(v293, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v295 = __riscv_vreinterpret_v_u8mf2_i8mf2(v294);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v296 = __riscv_vsrl_vx_u8mf2(v293, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v297 = __riscv_vreinterpret_v_u8mf2_i8mf2(v296);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v298 = v19 + 14;
      const int8_t* v299 = (const int8_t*) v298;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v300 = *(const int8_t *)(v299);
      const uint8_t* v301 = v19 + 46;
      const int8_t* v302 = (const int8_t*) v301;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v303 = *(const int8_t *)(v302);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v304 = __riscv_vwmacc_vx_i16m1(v289, v300, v295, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v305 = __riscv_vwmacc_vx_i16m1(v290, v303, v297, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v306 = v17 + 432;
      const uint8_t* v307 = (const uint8_t*) v306;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v308 = __riscv_vle8_v_u8mf2(v307, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v309 = __riscv_vand_vx_u8mf2(v308, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v310 = __riscv_vreinterpret_v_u8mf2_i8mf2(v309);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v311 = __riscv_vsrl_vx_u8mf2(v308, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v312 = __riscv_vreinterpret_v_u8mf2_i8mf2(v311);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v313 = v19 + 15;
      const int8_t* v314 = (const int8_t*) v313;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v315 = *(const int8_t *)(v314);
      const uint8_t* v316 = v19 + 47;
      const int8_t* v317 = (const int8_t*) v316;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v318 = *(const int8_t *)(v317);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v319 = __riscv_vwmacc_vx_i16m1(v304, v315, v310, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v320 = __riscv_vwmacc_vx_i16m1(v305, v318, v312, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v321 = v17 + 448;
      const uint8_t* v322 = (const uint8_t*) v321;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v323 = __riscv_vle8_v_u8mf2(v322, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v324 = __riscv_vand_vx_u8mf2(v323, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v325 = __riscv_vreinterpret_v_u8mf2_i8mf2(v324);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v326 = __riscv_vsrl_vx_u8mf2(v323, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v327 = __riscv_vreinterpret_v_u8mf2_i8mf2(v326);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v328 = v19 + 16;
      const int8_t* v329 = (const int8_t*) v328;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v330 = *(const int8_t *)(v329);
      const uint8_t* v331 = v19 + 48;
      const int8_t* v332 = (const int8_t*) v331;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v333 = *(const int8_t *)(v332);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v334 = __riscv_vwmacc_vx_i16m1(v319, v330, v325, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v335 = __riscv_vwmacc_vx_i16m1(v320, v333, v327, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v336 = v17 + 464;
      const uint8_t* v337 = (const uint8_t*) v336;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v338 = __riscv_vle8_v_u8mf2(v337, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v339 = __riscv_vand_vx_u8mf2(v338, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v340 = __riscv_vreinterpret_v_u8mf2_i8mf2(v339);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v341 = __riscv_vsrl_vx_u8mf2(v338, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v342 = __riscv_vreinterpret_v_u8mf2_i8mf2(v341);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v343 = v19 + 17;
      const int8_t* v344 = (const int8_t*) v343;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v345 = *(const int8_t *)(v344);
      const uint8_t* v346 = v19 + 49;
      const int8_t* v347 = (const int8_t*) v346;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v348 = *(const int8_t *)(v347);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v349 = __riscv_vwmacc_vx_i16m1(v334, v345, v340, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v350 = __riscv_vwmacc_vx_i16m1(v335, v348, v342, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v351 = v17 + 480;
      const uint8_t* v352 = (const uint8_t*) v351;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v353 = __riscv_vle8_v_u8mf2(v352, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v354 = __riscv_vand_vx_u8mf2(v353, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v355 = __riscv_vreinterpret_v_u8mf2_i8mf2(v354);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v356 = __riscv_vsrl_vx_u8mf2(v353, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v357 = __riscv_vreinterpret_v_u8mf2_i8mf2(v356);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v358 = v19 + 18;
      const int8_t* v359 = (const int8_t*) v358;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v360 = *(const int8_t *)(v359);
      const uint8_t* v361 = v19 + 50;
      const int8_t* v362 = (const int8_t*) v361;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v363 = *(const int8_t *)(v362);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v364 = __riscv_vwmacc_vx_i16m1(v349, v360, v355, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v365 = __riscv_vwmacc_vx_i16m1(v350, v363, v357, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v366 = v17 + 496;
      const uint8_t* v367 = (const uint8_t*) v366;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v368 = __riscv_vle8_v_u8mf2(v367, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v369 = __riscv_vand_vx_u8mf2(v368, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v370 = __riscv_vreinterpret_v_u8mf2_i8mf2(v369);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v371 = __riscv_vsrl_vx_u8mf2(v368, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v372 = __riscv_vreinterpret_v_u8mf2_i8mf2(v371);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v373 = v19 + 19;
      const int8_t* v374 = (const int8_t*) v373;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v375 = *(const int8_t *)(v374);
      const uint8_t* v376 = v19 + 51;
      const int8_t* v377 = (const int8_t*) v376;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v378 = *(const int8_t *)(v377);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v379 = __riscv_vwmacc_vx_i16m1(v364, v375, v370, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v380 = __riscv_vwmacc_vx_i16m1(v365, v378, v372, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v381 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v382 = __riscv_vwmacc_vv_i32m2(v381, v46, v379, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v383 = __riscv_vwmacc_vv_i32m2(v382, v64, v380, 16);
      v27 = v383;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v384 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v385 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v386 = v17 + 512;
      const uint8_t* v387 = (const uint8_t*) v386;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v388 = __riscv_vle8_v_u8mf2(v387, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v389 = __riscv_vand_vx_u8mf2(v388, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v390 = __riscv_vreinterpret_v_u8mf2_i8mf2(v389);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v391 = __riscv_vsrl_vx_u8mf2(v388, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v392 = __riscv_vreinterpret_v_u8mf2_i8mf2(v391);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v393 = v19 + 20;
      const int8_t* v394 = (const int8_t*) v393;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v395 = *(const int8_t *)(v394);
      const uint8_t* v396 = v19 + 52;
      const int8_t* v397 = (const int8_t*) v396;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v398 = *(const int8_t *)(v397);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v399 = __riscv_vwmacc_vx_i16m1(v384, v395, v390, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v400 = __riscv_vwmacc_vx_i16m1(v385, v398, v392, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v401 = v17 + 528;
      const uint8_t* v402 = (const uint8_t*) v401;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v403 = __riscv_vle8_v_u8mf2(v402, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v404 = __riscv_vand_vx_u8mf2(v403, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v405 = __riscv_vreinterpret_v_u8mf2_i8mf2(v404);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v406 = __riscv_vsrl_vx_u8mf2(v403, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v407 = __riscv_vreinterpret_v_u8mf2_i8mf2(v406);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v408 = v19 + 21;
      const int8_t* v409 = (const int8_t*) v408;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v410 = *(const int8_t *)(v409);
      const uint8_t* v411 = v19 + 53;
      const int8_t* v412 = (const int8_t*) v411;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v413 = *(const int8_t *)(v412);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v414 = __riscv_vwmacc_vx_i16m1(v399, v410, v405, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v415 = __riscv_vwmacc_vx_i16m1(v400, v413, v407, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v416 = v17 + 544;
      const uint8_t* v417 = (const uint8_t*) v416;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v418 = __riscv_vle8_v_u8mf2(v417, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v419 = __riscv_vand_vx_u8mf2(v418, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v420 = __riscv_vreinterpret_v_u8mf2_i8mf2(v419);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v421 = __riscv_vsrl_vx_u8mf2(v418, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v422 = __riscv_vreinterpret_v_u8mf2_i8mf2(v421);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v423 = v19 + 22;
      const int8_t* v424 = (const int8_t*) v423;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v425 = *(const int8_t *)(v424);
      const uint8_t* v426 = v19 + 54;
      const int8_t* v427 = (const int8_t*) v426;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v428 = *(const int8_t *)(v427);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v429 = __riscv_vwmacc_vx_i16m1(v414, v425, v420, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v430 = __riscv_vwmacc_vx_i16m1(v415, v428, v422, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v431 = v17 + 560;
      const uint8_t* v432 = (const uint8_t*) v431;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v433 = __riscv_vle8_v_u8mf2(v432, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v434 = __riscv_vand_vx_u8mf2(v433, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v435 = __riscv_vreinterpret_v_u8mf2_i8mf2(v434);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v436 = __riscv_vsrl_vx_u8mf2(v433, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v437 = __riscv_vreinterpret_v_u8mf2_i8mf2(v436);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v438 = v19 + 23;
      const int8_t* v439 = (const int8_t*) v438;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v440 = *(const int8_t *)(v439);
      const uint8_t* v441 = v19 + 55;
      const int8_t* v442 = (const int8_t*) v441;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v443 = *(const int8_t *)(v442);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v444 = __riscv_vwmacc_vx_i16m1(v429, v440, v435, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v445 = __riscv_vwmacc_vx_i16m1(v430, v443, v437, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v446 = v17 + 576;
      const uint8_t* v447 = (const uint8_t*) v446;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v448 = __riscv_vle8_v_u8mf2(v447, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v449 = __riscv_vand_vx_u8mf2(v448, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v450 = __riscv_vreinterpret_v_u8mf2_i8mf2(v449);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v451 = __riscv_vsrl_vx_u8mf2(v448, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v452 = __riscv_vreinterpret_v_u8mf2_i8mf2(v451);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v453 = v19 + 24;
      const int8_t* v454 = (const int8_t*) v453;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v455 = *(const int8_t *)(v454);
      const uint8_t* v456 = v19 + 56;
      const int8_t* v457 = (const int8_t*) v456;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v458 = *(const int8_t *)(v457);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v459 = __riscv_vwmacc_vx_i16m1(v444, v455, v450, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v460 = __riscv_vwmacc_vx_i16m1(v445, v458, v452, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v461 = v17 + 592;
      const uint8_t* v462 = (const uint8_t*) v461;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v463 = __riscv_vle8_v_u8mf2(v462, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v464 = __riscv_vand_vx_u8mf2(v463, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v465 = __riscv_vreinterpret_v_u8mf2_i8mf2(v464);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v466 = __riscv_vsrl_vx_u8mf2(v463, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v467 = __riscv_vreinterpret_v_u8mf2_i8mf2(v466);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v468 = v19 + 25;
      const int8_t* v469 = (const int8_t*) v468;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v470 = *(const int8_t *)(v469);
      const uint8_t* v471 = v19 + 57;
      const int8_t* v472 = (const int8_t*) v471;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v473 = *(const int8_t *)(v472);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v474 = __riscv_vwmacc_vx_i16m1(v459, v470, v465, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v475 = __riscv_vwmacc_vx_i16m1(v460, v473, v467, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v476 = v17 + 608;
      const uint8_t* v477 = (const uint8_t*) v476;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v478 = __riscv_vle8_v_u8mf2(v477, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v479 = __riscv_vand_vx_u8mf2(v478, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v480 = __riscv_vreinterpret_v_u8mf2_i8mf2(v479);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v481 = __riscv_vsrl_vx_u8mf2(v478, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v482 = __riscv_vreinterpret_v_u8mf2_i8mf2(v481);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v483 = v19 + 26;
      const int8_t* v484 = (const int8_t*) v483;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v485 = *(const int8_t *)(v484);
      const uint8_t* v486 = v19 + 58;
      const int8_t* v487 = (const int8_t*) v486;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v488 = *(const int8_t *)(v487);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v489 = __riscv_vwmacc_vx_i16m1(v474, v485, v480, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v490 = __riscv_vwmacc_vx_i16m1(v475, v488, v482, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v491 = v17 + 624;
      const uint8_t* v492 = (const uint8_t*) v491;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v493 = __riscv_vle8_v_u8mf2(v492, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v494 = __riscv_vand_vx_u8mf2(v493, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v495 = __riscv_vreinterpret_v_u8mf2_i8mf2(v494);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v496 = __riscv_vsrl_vx_u8mf2(v493, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v497 = __riscv_vreinterpret_v_u8mf2_i8mf2(v496);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v498 = v19 + 27;
      const int8_t* v499 = (const int8_t*) v498;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v500 = *(const int8_t *)(v499);
      const uint8_t* v501 = v19 + 59;
      const int8_t* v502 = (const int8_t*) v501;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v503 = *(const int8_t *)(v502);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v504 = __riscv_vwmacc_vx_i16m1(v489, v500, v495, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v505 = __riscv_vwmacc_vx_i16m1(v490, v503, v497, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v506 = v17 + 640;
      const uint8_t* v507 = (const uint8_t*) v506;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v508 = __riscv_vle8_v_u8mf2(v507, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v509 = __riscv_vand_vx_u8mf2(v508, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v510 = __riscv_vreinterpret_v_u8mf2_i8mf2(v509);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v511 = __riscv_vsrl_vx_u8mf2(v508, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v512 = __riscv_vreinterpret_v_u8mf2_i8mf2(v511);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v513 = v19 + 28;
      const int8_t* v514 = (const int8_t*) v513;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v515 = *(const int8_t *)(v514);
      const uint8_t* v516 = v19 + 60;
      const int8_t* v517 = (const int8_t*) v516;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v518 = *(const int8_t *)(v517);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v519 = __riscv_vwmacc_vx_i16m1(v504, v515, v510, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v520 = __riscv_vwmacc_vx_i16m1(v505, v518, v512, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v521 = v17 + 656;
      const uint8_t* v522 = (const uint8_t*) v521;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v523 = __riscv_vle8_v_u8mf2(v522, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v524 = __riscv_vand_vx_u8mf2(v523, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v525 = __riscv_vreinterpret_v_u8mf2_i8mf2(v524);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v526 = __riscv_vsrl_vx_u8mf2(v523, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v527 = __riscv_vreinterpret_v_u8mf2_i8mf2(v526);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v528 = v19 + 29;
      const int8_t* v529 = (const int8_t*) v528;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v530 = *(const int8_t *)(v529);
      const uint8_t* v531 = v19 + 61;
      const int8_t* v532 = (const int8_t*) v531;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v533 = *(const int8_t *)(v532);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v534 = __riscv_vwmacc_vx_i16m1(v519, v530, v525, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v535 = __riscv_vwmacc_vx_i16m1(v520, v533, v527, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v536 = v17 + 672;
      const uint8_t* v537 = (const uint8_t*) v536;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v538 = __riscv_vle8_v_u8mf2(v537, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v539 = __riscv_vand_vx_u8mf2(v538, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v540 = __riscv_vreinterpret_v_u8mf2_i8mf2(v539);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v541 = __riscv_vsrl_vx_u8mf2(v538, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v542 = __riscv_vreinterpret_v_u8mf2_i8mf2(v541);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v543 = v19 + 30;
      const int8_t* v544 = (const int8_t*) v543;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v545 = *(const int8_t *)(v544);
      const uint8_t* v546 = v19 + 62;
      const int8_t* v547 = (const int8_t*) v546;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v548 = *(const int8_t *)(v547);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v549 = __riscv_vwmacc_vx_i16m1(v534, v545, v540, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v550 = __riscv_vwmacc_vx_i16m1(v535, v548, v542, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v551 = v17 + 688;
      const uint8_t* v552 = (const uint8_t*) v551;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v553 = __riscv_vle8_v_u8mf2(v552, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v554 = __riscv_vand_vx_u8mf2(v553, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v555 = __riscv_vreinterpret_v_u8mf2_i8mf2(v554);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v556 = __riscv_vsrl_vx_u8mf2(v553, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v557 = __riscv_vreinterpret_v_u8mf2_i8mf2(v556);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v558 = v19 + 31;
      const int8_t* v559 = (const int8_t*) v558;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v560 = *(const int8_t *)(v559);
      const uint8_t* v561 = v19 + 63;
      const int8_t* v562 = (const int8_t*) v561;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v563 = *(const int8_t *)(v562);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v564 = __riscv_vwmacc_vx_i16m1(v549, v560, v555, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v565 = __riscv_vwmacc_vx_i16m1(v550, v563, v557, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v566 = v17 + 704;
      const uint8_t* v567 = (const uint8_t*) v566;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v568 = __riscv_vle8_v_u8mf2(v567, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v569 = __riscv_vand_vx_u8mf2(v568, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v570 = __riscv_vreinterpret_v_u8mf2_i8mf2(v569);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v571 = __riscv_vsrl_vx_u8mf2(v568, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v572 = __riscv_vreinterpret_v_u8mf2_i8mf2(v571);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v573 = v19 + 32;
      const int8_t* v574 = (const int8_t*) v573;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v575 = *(const int8_t *)(v574);
      const uint8_t* v576 = v19 + 64;
      const int8_t* v577 = (const int8_t*) v576;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v578 = *(const int8_t *)(v577);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v579 = __riscv_vwmacc_vx_i16m1(v564, v575, v570, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v580 = __riscv_vwmacc_vx_i16m1(v565, v578, v572, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v581 = v17 + 720;
      const uint8_t* v582 = (const uint8_t*) v581;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v583 = __riscv_vle8_v_u8mf2(v582, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v584 = __riscv_vand_vx_u8mf2(v583, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v585 = __riscv_vreinterpret_v_u8mf2_i8mf2(v584);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v586 = __riscv_vsrl_vx_u8mf2(v583, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v587 = __riscv_vreinterpret_v_u8mf2_i8mf2(v586);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v588 = v19 + 33;
      const int8_t* v589 = (const int8_t*) v588;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v590 = *(const int8_t *)(v589);
      const uint8_t* v591 = v19 + 65;
      const int8_t* v592 = (const int8_t*) v591;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v593 = *(const int8_t *)(v592);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v594 = __riscv_vwmacc_vx_i16m1(v579, v590, v585, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v595 = __riscv_vwmacc_vx_i16m1(v580, v593, v587, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v596 = v17 + 736;
      const uint8_t* v597 = (const uint8_t*) v596;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v598 = __riscv_vle8_v_u8mf2(v597, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v599 = __riscv_vand_vx_u8mf2(v598, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v600 = __riscv_vreinterpret_v_u8mf2_i8mf2(v599);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v601 = __riscv_vsrl_vx_u8mf2(v598, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v602 = __riscv_vreinterpret_v_u8mf2_i8mf2(v601);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v603 = v19 + 34;
      const int8_t* v604 = (const int8_t*) v603;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v605 = *(const int8_t *)(v604);
      const uint8_t* v606 = v19 + 66;
      const int8_t* v607 = (const int8_t*) v606;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v608 = *(const int8_t *)(v607);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v609 = __riscv_vwmacc_vx_i16m1(v594, v605, v600, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v610 = __riscv_vwmacc_vx_i16m1(v595, v608, v602, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v611 = v17 + 752;
      const uint8_t* v612 = (const uint8_t*) v611;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v613 = __riscv_vle8_v_u8mf2(v612, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v614 = __riscv_vand_vx_u8mf2(v613, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v615 = __riscv_vreinterpret_v_u8mf2_i8mf2(v614);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v616 = __riscv_vsrl_vx_u8mf2(v613, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v617 = __riscv_vreinterpret_v_u8mf2_i8mf2(v616);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v618 = v19 + 35;
      const int8_t* v619 = (const int8_t*) v618;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v620 = *(const int8_t *)(v619);
      const uint8_t* v621 = v19 + 67;
      const int8_t* v622 = (const int8_t*) v621;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v623 = *(const int8_t *)(v622);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v624 = __riscv_vwmacc_vx_i16m1(v609, v620, v615, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v625 = __riscv_vwmacc_vx_i16m1(v610, v623, v617, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v626 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v627 = __riscv_vwmacc_vv_i32m2(v626, v46, v624, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v628 = __riscv_vwmacc_vv_i32m2(v627, v64, v625, 16);
      v27 = v628;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v629 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v630 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v631 = v17 + 768;
      const uint8_t* v632 = (const uint8_t*) v631;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v633 = __riscv_vle8_v_u8mf2(v632, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v634 = __riscv_vand_vx_u8mf2(v633, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v635 = __riscv_vreinterpret_v_u8mf2_i8mf2(v634);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v636 = __riscv_vsrl_vx_u8mf2(v633, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v637 = __riscv_vreinterpret_v_u8mf2_i8mf2(v636);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v638 = v19 + 68;
      const int8_t* v639 = (const int8_t*) v638;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v640 = *(const int8_t *)(v639);
      const uint8_t* v641 = v19 + 100;
      const int8_t* v642 = (const int8_t*) v641;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v643 = *(const int8_t *)(v642);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v644 = __riscv_vwmacc_vx_i16m1(v629, v640, v635, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v645 = __riscv_vwmacc_vx_i16m1(v630, v643, v637, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v646 = v17 + 784;
      const uint8_t* v647 = (const uint8_t*) v646;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v648 = __riscv_vle8_v_u8mf2(v647, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v649 = __riscv_vand_vx_u8mf2(v648, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v650 = __riscv_vreinterpret_v_u8mf2_i8mf2(v649);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v651 = __riscv_vsrl_vx_u8mf2(v648, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v652 = __riscv_vreinterpret_v_u8mf2_i8mf2(v651);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v653 = v19 + 69;
      const int8_t* v654 = (const int8_t*) v653;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v655 = *(const int8_t *)(v654);
      const uint8_t* v656 = v19 + 101;
      const int8_t* v657 = (const int8_t*) v656;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v658 = *(const int8_t *)(v657);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v659 = __riscv_vwmacc_vx_i16m1(v644, v655, v650, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v660 = __riscv_vwmacc_vx_i16m1(v645, v658, v652, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v661 = v17 + 800;
      const uint8_t* v662 = (const uint8_t*) v661;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v663 = __riscv_vle8_v_u8mf2(v662, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v664 = __riscv_vand_vx_u8mf2(v663, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v665 = __riscv_vreinterpret_v_u8mf2_i8mf2(v664);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v666 = __riscv_vsrl_vx_u8mf2(v663, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v667 = __riscv_vreinterpret_v_u8mf2_i8mf2(v666);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v668 = v19 + 70;
      const int8_t* v669 = (const int8_t*) v668;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v670 = *(const int8_t *)(v669);
      const uint8_t* v671 = v19 + 102;
      const int8_t* v672 = (const int8_t*) v671;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v673 = *(const int8_t *)(v672);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v674 = __riscv_vwmacc_vx_i16m1(v659, v670, v665, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v675 = __riscv_vwmacc_vx_i16m1(v660, v673, v667, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v676 = v17 + 816;
      const uint8_t* v677 = (const uint8_t*) v676;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v678 = __riscv_vle8_v_u8mf2(v677, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v679 = __riscv_vand_vx_u8mf2(v678, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v680 = __riscv_vreinterpret_v_u8mf2_i8mf2(v679);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v681 = __riscv_vsrl_vx_u8mf2(v678, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v682 = __riscv_vreinterpret_v_u8mf2_i8mf2(v681);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v683 = v19 + 71;
      const int8_t* v684 = (const int8_t*) v683;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v685 = *(const int8_t *)(v684);
      const uint8_t* v686 = v19 + 103;
      const int8_t* v687 = (const int8_t*) v686;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v688 = *(const int8_t *)(v687);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v689 = __riscv_vwmacc_vx_i16m1(v674, v685, v680, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v690 = __riscv_vwmacc_vx_i16m1(v675, v688, v682, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v691 = v17 + 832;
      const uint8_t* v692 = (const uint8_t*) v691;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v693 = __riscv_vle8_v_u8mf2(v692, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v694 = __riscv_vand_vx_u8mf2(v693, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v695 = __riscv_vreinterpret_v_u8mf2_i8mf2(v694);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v696 = __riscv_vsrl_vx_u8mf2(v693, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v697 = __riscv_vreinterpret_v_u8mf2_i8mf2(v696);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v698 = v19 + 72;
      const int8_t* v699 = (const int8_t*) v698;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v700 = *(const int8_t *)(v699);
      const uint8_t* v701 = v19 + 104;
      const int8_t* v702 = (const int8_t*) v701;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v703 = *(const int8_t *)(v702);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v704 = __riscv_vwmacc_vx_i16m1(v689, v700, v695, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v705 = __riscv_vwmacc_vx_i16m1(v690, v703, v697, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v706 = v17 + 848;
      const uint8_t* v707 = (const uint8_t*) v706;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v708 = __riscv_vle8_v_u8mf2(v707, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v709 = __riscv_vand_vx_u8mf2(v708, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v710 = __riscv_vreinterpret_v_u8mf2_i8mf2(v709);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v711 = __riscv_vsrl_vx_u8mf2(v708, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v712 = __riscv_vreinterpret_v_u8mf2_i8mf2(v711);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v713 = v19 + 73;
      const int8_t* v714 = (const int8_t*) v713;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v715 = *(const int8_t *)(v714);
      const uint8_t* v716 = v19 + 105;
      const int8_t* v717 = (const int8_t*) v716;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v718 = *(const int8_t *)(v717);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v719 = __riscv_vwmacc_vx_i16m1(v704, v715, v710, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v720 = __riscv_vwmacc_vx_i16m1(v705, v718, v712, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v721 = v17 + 864;
      const uint8_t* v722 = (const uint8_t*) v721;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v723 = __riscv_vle8_v_u8mf2(v722, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v724 = __riscv_vand_vx_u8mf2(v723, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v725 = __riscv_vreinterpret_v_u8mf2_i8mf2(v724);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v726 = __riscv_vsrl_vx_u8mf2(v723, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v727 = __riscv_vreinterpret_v_u8mf2_i8mf2(v726);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v728 = v19 + 74;
      const int8_t* v729 = (const int8_t*) v728;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v730 = *(const int8_t *)(v729);
      const uint8_t* v731 = v19 + 106;
      const int8_t* v732 = (const int8_t*) v731;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v733 = *(const int8_t *)(v732);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v734 = __riscv_vwmacc_vx_i16m1(v719, v730, v725, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v735 = __riscv_vwmacc_vx_i16m1(v720, v733, v727, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v736 = v17 + 880;
      const uint8_t* v737 = (const uint8_t*) v736;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v738 = __riscv_vle8_v_u8mf2(v737, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v739 = __riscv_vand_vx_u8mf2(v738, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v740 = __riscv_vreinterpret_v_u8mf2_i8mf2(v739);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v741 = __riscv_vsrl_vx_u8mf2(v738, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v742 = __riscv_vreinterpret_v_u8mf2_i8mf2(v741);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v743 = v19 + 75;
      const int8_t* v744 = (const int8_t*) v743;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v745 = *(const int8_t *)(v744);
      const uint8_t* v746 = v19 + 107;
      const int8_t* v747 = (const int8_t*) v746;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v748 = *(const int8_t *)(v747);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v749 = __riscv_vwmacc_vx_i16m1(v734, v745, v740, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v750 = __riscv_vwmacc_vx_i16m1(v735, v748, v742, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v751 = v17 + 896;
      const uint8_t* v752 = (const uint8_t*) v751;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v753 = __riscv_vle8_v_u8mf2(v752, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v754 = __riscv_vand_vx_u8mf2(v753, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v755 = __riscv_vreinterpret_v_u8mf2_i8mf2(v754);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v756 = __riscv_vsrl_vx_u8mf2(v753, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v757 = __riscv_vreinterpret_v_u8mf2_i8mf2(v756);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v758 = v19 + 76;
      const int8_t* v759 = (const int8_t*) v758;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v760 = *(const int8_t *)(v759);
      const uint8_t* v761 = v19 + 108;
      const int8_t* v762 = (const int8_t*) v761;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v763 = *(const int8_t *)(v762);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v764 = __riscv_vwmacc_vx_i16m1(v749, v760, v755, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v765 = __riscv_vwmacc_vx_i16m1(v750, v763, v757, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v766 = v17 + 912;
      const uint8_t* v767 = (const uint8_t*) v766;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v768 = __riscv_vle8_v_u8mf2(v767, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v769 = __riscv_vand_vx_u8mf2(v768, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v770 = __riscv_vreinterpret_v_u8mf2_i8mf2(v769);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v771 = __riscv_vsrl_vx_u8mf2(v768, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v772 = __riscv_vreinterpret_v_u8mf2_i8mf2(v771);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v773 = v19 + 77;
      const int8_t* v774 = (const int8_t*) v773;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v775 = *(const int8_t *)(v774);
      const uint8_t* v776 = v19 + 109;
      const int8_t* v777 = (const int8_t*) v776;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v778 = *(const int8_t *)(v777);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v779 = __riscv_vwmacc_vx_i16m1(v764, v775, v770, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v780 = __riscv_vwmacc_vx_i16m1(v765, v778, v772, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v781 = v17 + 928;
      const uint8_t* v782 = (const uint8_t*) v781;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v783 = __riscv_vle8_v_u8mf2(v782, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v784 = __riscv_vand_vx_u8mf2(v783, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v785 = __riscv_vreinterpret_v_u8mf2_i8mf2(v784);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v786 = __riscv_vsrl_vx_u8mf2(v783, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v787 = __riscv_vreinterpret_v_u8mf2_i8mf2(v786);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v788 = v19 + 78;
      const int8_t* v789 = (const int8_t*) v788;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v790 = *(const int8_t *)(v789);
      const uint8_t* v791 = v19 + 110;
      const int8_t* v792 = (const int8_t*) v791;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v793 = *(const int8_t *)(v792);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v794 = __riscv_vwmacc_vx_i16m1(v779, v790, v785, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v795 = __riscv_vwmacc_vx_i16m1(v780, v793, v787, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v796 = v17 + 944;
      const uint8_t* v797 = (const uint8_t*) v796;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v798 = __riscv_vle8_v_u8mf2(v797, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v799 = __riscv_vand_vx_u8mf2(v798, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v800 = __riscv_vreinterpret_v_u8mf2_i8mf2(v799);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v801 = __riscv_vsrl_vx_u8mf2(v798, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v802 = __riscv_vreinterpret_v_u8mf2_i8mf2(v801);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v803 = v19 + 79;
      const int8_t* v804 = (const int8_t*) v803;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v805 = *(const int8_t *)(v804);
      const uint8_t* v806 = v19 + 111;
      const int8_t* v807 = (const int8_t*) v806;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v808 = *(const int8_t *)(v807);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v809 = __riscv_vwmacc_vx_i16m1(v794, v805, v800, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v810 = __riscv_vwmacc_vx_i16m1(v795, v808, v802, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v811 = v17 + 960;
      const uint8_t* v812 = (const uint8_t*) v811;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v813 = __riscv_vle8_v_u8mf2(v812, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v814 = __riscv_vand_vx_u8mf2(v813, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v815 = __riscv_vreinterpret_v_u8mf2_i8mf2(v814);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v816 = __riscv_vsrl_vx_u8mf2(v813, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v817 = __riscv_vreinterpret_v_u8mf2_i8mf2(v816);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v818 = v19 + 80;
      const int8_t* v819 = (const int8_t*) v818;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v820 = *(const int8_t *)(v819);
      const uint8_t* v821 = v19 + 112;
      const int8_t* v822 = (const int8_t*) v821;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v823 = *(const int8_t *)(v822);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v824 = __riscv_vwmacc_vx_i16m1(v809, v820, v815, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v825 = __riscv_vwmacc_vx_i16m1(v810, v823, v817, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v826 = v17 + 976;
      const uint8_t* v827 = (const uint8_t*) v826;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v828 = __riscv_vle8_v_u8mf2(v827, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v829 = __riscv_vand_vx_u8mf2(v828, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v830 = __riscv_vreinterpret_v_u8mf2_i8mf2(v829);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v831 = __riscv_vsrl_vx_u8mf2(v828, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v832 = __riscv_vreinterpret_v_u8mf2_i8mf2(v831);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v833 = v19 + 81;
      const int8_t* v834 = (const int8_t*) v833;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v835 = *(const int8_t *)(v834);
      const uint8_t* v836 = v19 + 113;
      const int8_t* v837 = (const int8_t*) v836;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v838 = *(const int8_t *)(v837);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v839 = __riscv_vwmacc_vx_i16m1(v824, v835, v830, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v840 = __riscv_vwmacc_vx_i16m1(v825, v838, v832, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v841 = v17 + 992;
      const uint8_t* v842 = (const uint8_t*) v841;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v843 = __riscv_vle8_v_u8mf2(v842, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v844 = __riscv_vand_vx_u8mf2(v843, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v845 = __riscv_vreinterpret_v_u8mf2_i8mf2(v844);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v846 = __riscv_vsrl_vx_u8mf2(v843, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v847 = __riscv_vreinterpret_v_u8mf2_i8mf2(v846);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v848 = v19 + 82;
      const int8_t* v849 = (const int8_t*) v848;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v850 = *(const int8_t *)(v849);
      const uint8_t* v851 = v19 + 114;
      const int8_t* v852 = (const int8_t*) v851;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v853 = *(const int8_t *)(v852);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v854 = __riscv_vwmacc_vx_i16m1(v839, v850, v845, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v855 = __riscv_vwmacc_vx_i16m1(v840, v853, v847, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v856 = v17 + 1008;
      const uint8_t* v857 = (const uint8_t*) v856;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v858 = __riscv_vle8_v_u8mf2(v857, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v859 = __riscv_vand_vx_u8mf2(v858, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v860 = __riscv_vreinterpret_v_u8mf2_i8mf2(v859);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v861 = __riscv_vsrl_vx_u8mf2(v858, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v862 = __riscv_vreinterpret_v_u8mf2_i8mf2(v861);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v863 = v19 + 83;
      const int8_t* v864 = (const int8_t*) v863;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v865 = *(const int8_t *)(v864);
      const uint8_t* v866 = v19 + 115;
      const int8_t* v867 = (const int8_t*) v866;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v868 = *(const int8_t *)(v867);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v869 = __riscv_vwmacc_vx_i16m1(v854, v865, v860, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v870 = __riscv_vwmacc_vx_i16m1(v855, v868, v862, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v871 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v872 = __riscv_vwmacc_vv_i32m2(v871, v82, v869, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v873 = __riscv_vwmacc_vv_i32m2(v872, v100, v870, 16);
      v27 = v873;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v874 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v875 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v876 = v17 + 1024;
      const uint8_t* v877 = (const uint8_t*) v876;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v878 = __riscv_vle8_v_u8mf2(v877, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v879 = __riscv_vand_vx_u8mf2(v878, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v880 = __riscv_vreinterpret_v_u8mf2_i8mf2(v879);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v881 = __riscv_vsrl_vx_u8mf2(v878, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v882 = __riscv_vreinterpret_v_u8mf2_i8mf2(v881);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v883 = v19 + 84;
      const int8_t* v884 = (const int8_t*) v883;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v885 = *(const int8_t *)(v884);
      const uint8_t* v886 = v19 + 116;
      const int8_t* v887 = (const int8_t*) v886;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v888 = *(const int8_t *)(v887);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v889 = __riscv_vwmacc_vx_i16m1(v874, v885, v880, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v890 = __riscv_vwmacc_vx_i16m1(v875, v888, v882, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v891 = v17 + 1040;
      const uint8_t* v892 = (const uint8_t*) v891;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v893 = __riscv_vle8_v_u8mf2(v892, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v894 = __riscv_vand_vx_u8mf2(v893, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v895 = __riscv_vreinterpret_v_u8mf2_i8mf2(v894);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v896 = __riscv_vsrl_vx_u8mf2(v893, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v897 = __riscv_vreinterpret_v_u8mf2_i8mf2(v896);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v898 = v19 + 85;
      const int8_t* v899 = (const int8_t*) v898;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v900 = *(const int8_t *)(v899);
      const uint8_t* v901 = v19 + 117;
      const int8_t* v902 = (const int8_t*) v901;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v903 = *(const int8_t *)(v902);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v904 = __riscv_vwmacc_vx_i16m1(v889, v900, v895, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v905 = __riscv_vwmacc_vx_i16m1(v890, v903, v897, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v906 = v17 + 1056;
      const uint8_t* v907 = (const uint8_t*) v906;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v908 = __riscv_vle8_v_u8mf2(v907, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v909 = __riscv_vand_vx_u8mf2(v908, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v910 = __riscv_vreinterpret_v_u8mf2_i8mf2(v909);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v911 = __riscv_vsrl_vx_u8mf2(v908, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v912 = __riscv_vreinterpret_v_u8mf2_i8mf2(v911);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v913 = v19 + 86;
      const int8_t* v914 = (const int8_t*) v913;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v915 = *(const int8_t *)(v914);
      const uint8_t* v916 = v19 + 118;
      const int8_t* v917 = (const int8_t*) v916;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v918 = *(const int8_t *)(v917);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v919 = __riscv_vwmacc_vx_i16m1(v904, v915, v910, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v920 = __riscv_vwmacc_vx_i16m1(v905, v918, v912, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v921 = v17 + 1072;
      const uint8_t* v922 = (const uint8_t*) v921;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v923 = __riscv_vle8_v_u8mf2(v922, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v924 = __riscv_vand_vx_u8mf2(v923, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v925 = __riscv_vreinterpret_v_u8mf2_i8mf2(v924);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v926 = __riscv_vsrl_vx_u8mf2(v923, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v927 = __riscv_vreinterpret_v_u8mf2_i8mf2(v926);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v928 = v19 + 87;
      const int8_t* v929 = (const int8_t*) v928;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v930 = *(const int8_t *)(v929);
      const uint8_t* v931 = v19 + 119;
      const int8_t* v932 = (const int8_t*) v931;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v933 = *(const int8_t *)(v932);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v934 = __riscv_vwmacc_vx_i16m1(v919, v930, v925, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v935 = __riscv_vwmacc_vx_i16m1(v920, v933, v927, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v936 = v17 + 1088;
      const uint8_t* v937 = (const uint8_t*) v936;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v938 = __riscv_vle8_v_u8mf2(v937, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v939 = __riscv_vand_vx_u8mf2(v938, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v940 = __riscv_vreinterpret_v_u8mf2_i8mf2(v939);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v941 = __riscv_vsrl_vx_u8mf2(v938, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v942 = __riscv_vreinterpret_v_u8mf2_i8mf2(v941);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v943 = v19 + 88;
      const int8_t* v944 = (const int8_t*) v943;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v945 = *(const int8_t *)(v944);
      const uint8_t* v946 = v19 + 120;
      const int8_t* v947 = (const int8_t*) v946;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v948 = *(const int8_t *)(v947);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v949 = __riscv_vwmacc_vx_i16m1(v934, v945, v940, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v950 = __riscv_vwmacc_vx_i16m1(v935, v948, v942, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v951 = v17 + 1104;
      const uint8_t* v952 = (const uint8_t*) v951;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v953 = __riscv_vle8_v_u8mf2(v952, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v954 = __riscv_vand_vx_u8mf2(v953, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v955 = __riscv_vreinterpret_v_u8mf2_i8mf2(v954);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v956 = __riscv_vsrl_vx_u8mf2(v953, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v957 = __riscv_vreinterpret_v_u8mf2_i8mf2(v956);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v958 = v19 + 89;
      const int8_t* v959 = (const int8_t*) v958;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v960 = *(const int8_t *)(v959);
      const uint8_t* v961 = v19 + 121;
      const int8_t* v962 = (const int8_t*) v961;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v963 = *(const int8_t *)(v962);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v964 = __riscv_vwmacc_vx_i16m1(v949, v960, v955, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v965 = __riscv_vwmacc_vx_i16m1(v950, v963, v957, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v966 = v17 + 1120;
      const uint8_t* v967 = (const uint8_t*) v966;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v968 = __riscv_vle8_v_u8mf2(v967, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v969 = __riscv_vand_vx_u8mf2(v968, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v970 = __riscv_vreinterpret_v_u8mf2_i8mf2(v969);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v971 = __riscv_vsrl_vx_u8mf2(v968, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v972 = __riscv_vreinterpret_v_u8mf2_i8mf2(v971);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v973 = v19 + 90;
      const int8_t* v974 = (const int8_t*) v973;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v975 = *(const int8_t *)(v974);
      const uint8_t* v976 = v19 + 122;
      const int8_t* v977 = (const int8_t*) v976;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v978 = *(const int8_t *)(v977);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v979 = __riscv_vwmacc_vx_i16m1(v964, v975, v970, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v980 = __riscv_vwmacc_vx_i16m1(v965, v978, v972, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v981 = v17 + 1136;
      const uint8_t* v982 = (const uint8_t*) v981;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v983 = __riscv_vle8_v_u8mf2(v982, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v984 = __riscv_vand_vx_u8mf2(v983, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v985 = __riscv_vreinterpret_v_u8mf2_i8mf2(v984);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v986 = __riscv_vsrl_vx_u8mf2(v983, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v987 = __riscv_vreinterpret_v_u8mf2_i8mf2(v986);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v988 = v19 + 91;
      const int8_t* v989 = (const int8_t*) v988;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v990 = *(const int8_t *)(v989);
      const uint8_t* v991 = v19 + 123;
      const int8_t* v992 = (const int8_t*) v991;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v993 = *(const int8_t *)(v992);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v994 = __riscv_vwmacc_vx_i16m1(v979, v990, v985, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v995 = __riscv_vwmacc_vx_i16m1(v980, v993, v987, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v996 = v17 + 1152;
      const uint8_t* v997 = (const uint8_t*) v996;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v998 = __riscv_vle8_v_u8mf2(v997, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v999 = __riscv_vand_vx_u8mf2(v998, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1000 = __riscv_vreinterpret_v_u8mf2_i8mf2(v999);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1001 = __riscv_vsrl_vx_u8mf2(v998, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1002 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1001);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1003 = v19 + 92;
      const int8_t* v1004 = (const int8_t*) v1003;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1005 = *(const int8_t *)(v1004);
      const uint8_t* v1006 = v19 + 124;
      const int8_t* v1007 = (const int8_t*) v1006;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1008 = *(const int8_t *)(v1007);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1009 = __riscv_vwmacc_vx_i16m1(v994, v1005, v1000, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1010 = __riscv_vwmacc_vx_i16m1(v995, v1008, v1002, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1011 = v17 + 1168;
      const uint8_t* v1012 = (const uint8_t*) v1011;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1013 = __riscv_vle8_v_u8mf2(v1012, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1014 = __riscv_vand_vx_u8mf2(v1013, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1015 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1014);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1016 = __riscv_vsrl_vx_u8mf2(v1013, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1017 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1016);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1018 = v19 + 93;
      const int8_t* v1019 = (const int8_t*) v1018;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1020 = *(const int8_t *)(v1019);
      const uint8_t* v1021 = v19 + 125;
      const int8_t* v1022 = (const int8_t*) v1021;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1023 = *(const int8_t *)(v1022);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1024 = __riscv_vwmacc_vx_i16m1(v1009, v1020, v1015, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1025 = __riscv_vwmacc_vx_i16m1(v1010, v1023, v1017, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1026 = v17 + 1184;
      const uint8_t* v1027 = (const uint8_t*) v1026;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1028 = __riscv_vle8_v_u8mf2(v1027, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1029 = __riscv_vand_vx_u8mf2(v1028, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1030 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1029);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1031 = __riscv_vsrl_vx_u8mf2(v1028, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1032 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1031);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1033 = v19 + 94;
      const int8_t* v1034 = (const int8_t*) v1033;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1035 = *(const int8_t *)(v1034);
      const uint8_t* v1036 = v19 + 126;
      const int8_t* v1037 = (const int8_t*) v1036;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1038 = *(const int8_t *)(v1037);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1039 = __riscv_vwmacc_vx_i16m1(v1024, v1035, v1030, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1040 = __riscv_vwmacc_vx_i16m1(v1025, v1038, v1032, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1041 = v17 + 1200;
      const uint8_t* v1042 = (const uint8_t*) v1041;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1043 = __riscv_vle8_v_u8mf2(v1042, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1044 = __riscv_vand_vx_u8mf2(v1043, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1045 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1044);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1046 = __riscv_vsrl_vx_u8mf2(v1043, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1047 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1046);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1048 = v19 + 95;
      const int8_t* v1049 = (const int8_t*) v1048;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1050 = *(const int8_t *)(v1049);
      const uint8_t* v1051 = v19 + 127;
      const int8_t* v1052 = (const int8_t*) v1051;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1053 = *(const int8_t *)(v1052);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1054 = __riscv_vwmacc_vx_i16m1(v1039, v1050, v1045, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1055 = __riscv_vwmacc_vx_i16m1(v1040, v1053, v1047, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1056 = v17 + 1216;
      const uint8_t* v1057 = (const uint8_t*) v1056;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1058 = __riscv_vle8_v_u8mf2(v1057, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1059 = __riscv_vand_vx_u8mf2(v1058, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1060 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1059);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1061 = __riscv_vsrl_vx_u8mf2(v1058, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1062 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1061);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1063 = v19 + 96;
      const int8_t* v1064 = (const int8_t*) v1063;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1065 = *(const int8_t *)(v1064);
      const uint8_t* v1066 = v19 + 128;
      const int8_t* v1067 = (const int8_t*) v1066;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1068 = *(const int8_t *)(v1067);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1069 = __riscv_vwmacc_vx_i16m1(v1054, v1065, v1060, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1070 = __riscv_vwmacc_vx_i16m1(v1055, v1068, v1062, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1071 = v17 + 1232;
      const uint8_t* v1072 = (const uint8_t*) v1071;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1073 = __riscv_vle8_v_u8mf2(v1072, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1074 = __riscv_vand_vx_u8mf2(v1073, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1075 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1074);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1076 = __riscv_vsrl_vx_u8mf2(v1073, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1077 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1076);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1078 = v19 + 97;
      const int8_t* v1079 = (const int8_t*) v1078;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1080 = *(const int8_t *)(v1079);
      const uint8_t* v1081 = v19 + 129;
      const int8_t* v1082 = (const int8_t*) v1081;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1083 = *(const int8_t *)(v1082);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1084 = __riscv_vwmacc_vx_i16m1(v1069, v1080, v1075, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1085 = __riscv_vwmacc_vx_i16m1(v1070, v1083, v1077, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1086 = v17 + 1248;
      const uint8_t* v1087 = (const uint8_t*) v1086;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1088 = __riscv_vle8_v_u8mf2(v1087, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1089 = __riscv_vand_vx_u8mf2(v1088, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1090 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1089);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1091 = __riscv_vsrl_vx_u8mf2(v1088, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1092 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1091);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1093 = v19 + 98;
      const int8_t* v1094 = (const int8_t*) v1093;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1095 = *(const int8_t *)(v1094);
      const uint8_t* v1096 = v19 + 130;
      const int8_t* v1097 = (const int8_t*) v1096;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1098 = *(const int8_t *)(v1097);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1099 = __riscv_vwmacc_vx_i16m1(v1084, v1095, v1090, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1100 = __riscv_vwmacc_vx_i16m1(v1085, v1098, v1092, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1101 = v17 + 1264;
      const uint8_t* v1102 = (const uint8_t*) v1101;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1103 = __riscv_vle8_v_u8mf2(v1102, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1104 = __riscv_vand_vx_u8mf2(v1103, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1105 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1104);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1106 = __riscv_vsrl_vx_u8mf2(v1103, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1107 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1106);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1108 = v19 + 99;
      const int8_t* v1109 = (const int8_t*) v1108;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1110 = *(const int8_t *)(v1109);
      const uint8_t* v1111 = v19 + 131;
      const int8_t* v1112 = (const int8_t*) v1111;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1113 = *(const int8_t *)(v1112);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1114 = __riscv_vwmacc_vx_i16m1(v1099, v1110, v1105, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1115 = __riscv_vwmacc_vx_i16m1(v1100, v1113, v1107, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1116 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1117 = __riscv_vwmacc_vv_i32m2(v1116, v82, v1114, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1118 = __riscv_vwmacc_vv_i32m2(v1117, v100, v1115, 16);
      v27 = v1118;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_unpack_superhalf
      const uint8_t* v1119 = v17 + 128;
      const uint8_t* v1120 = (const uint8_t*) v1119;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1121 = __riscv_vle8_v_u8mf2(v1120, 16);
      const uint8_t* v1122 = v17 + 192;
      const uint8_t* v1123 = (const uint8_t*) v1122;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1124 = __riscv_vle8_v_u8mf2(v1123, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1125 = __riscv_vand_vx_u8mf2(v1121, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1126 = __riscv_vsrl_vx_u8mf2(v1121, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1127 = __riscv_vand_vx_u8mf2(v1124, 0x30, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1128 = __riscv_vand_vx_u8mf2(v1124, 0xC0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1129 = __riscv_vsrl_vx_u8mf2(v1128, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1130 = __riscv_vor_vv_u8mf2(v1127, v1125, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1131 = __riscv_vor_vv_u8mf2(v1129, v1126, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1132 = __riscv_vzext_vf2_u16m1(v1130, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1133 = __riscv_vreinterpret_v_u16m1_i16m1(v1132);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1134 = __riscv_vzext_vf2_u16m1(v1131, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1135 = __riscv_vreinterpret_v_u16m1_i16m1(v1134);
      const uint8_t* v1136 = v17 + 144;
      const uint8_t* v1137 = (const uint8_t*) v1136;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1138 = __riscv_vle8_v_u8mf2(v1137, 16);
      const uint8_t* v1139 = v17 + 208;
      const uint8_t* v1140 = (const uint8_t*) v1139;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1141 = __riscv_vle8_v_u8mf2(v1140, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1142 = __riscv_vand_vx_u8mf2(v1138, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1143 = __riscv_vsrl_vx_u8mf2(v1138, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1144 = __riscv_vand_vx_u8mf2(v1141, 0x30, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1145 = __riscv_vand_vx_u8mf2(v1141, 0xC0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1146 = __riscv_vsrl_vx_u8mf2(v1145, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1147 = __riscv_vor_vv_u8mf2(v1144, v1142, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1148 = __riscv_vor_vv_u8mf2(v1146, v1143, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1149 = __riscv_vzext_vf2_u16m1(v1147, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1150 = __riscv_vreinterpret_v_u16m1_i16m1(v1149);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1151 = __riscv_vzext_vf2_u16m1(v1148, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1152 = __riscv_vreinterpret_v_u16m1_i16m1(v1151);
      const uint8_t* v1153 = v17 + 160;
      const uint8_t* v1154 = (const uint8_t*) v1153;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1155 = __riscv_vle8_v_u8mf2(v1154, 16);
      const uint8_t* v1156 = v17 + 224;
      const uint8_t* v1157 = (const uint8_t*) v1156;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1158 = __riscv_vle8_v_u8mf2(v1157, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1159 = __riscv_vand_vx_u8mf2(v1155, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1160 = __riscv_vsrl_vx_u8mf2(v1155, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1161 = __riscv_vand_vx_u8mf2(v1158, 0x30, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1162 = __riscv_vand_vx_u8mf2(v1158, 0xC0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1163 = __riscv_vsrl_vx_u8mf2(v1162, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1164 = __riscv_vor_vv_u8mf2(v1161, v1159, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1165 = __riscv_vor_vv_u8mf2(v1163, v1160, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1166 = __riscv_vzext_vf2_u16m1(v1164, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1167 = __riscv_vreinterpret_v_u16m1_i16m1(v1166);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1168 = __riscv_vzext_vf2_u16m1(v1165, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1169 = __riscv_vreinterpret_v_u16m1_i16m1(v1168);
      const uint8_t* v1170 = v17 + 176;
      const uint8_t* v1171 = (const uint8_t*) v1170;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1172 = __riscv_vle8_v_u8mf2(v1171, 16);
      const uint8_t* v1173 = v17 + 240;
      const uint8_t* v1174 = (const uint8_t*) v1173;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1175 = __riscv_vle8_v_u8mf2(v1174, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1176 = __riscv_vand_vx_u8mf2(v1172, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1177 = __riscv_vsrl_vx_u8mf2(v1172, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1178 = __riscv_vand_vx_u8mf2(v1175, 0x30, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1179 = __riscv_vand_vx_u8mf2(v1175, 0xC0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1180 = __riscv_vsrl_vx_u8mf2(v1179, 2, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1181 = __riscv_vor_vv_u8mf2(v1178, v1176, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
      vuint8mf2_t v1182 = __riscv_vor_vv_u8mf2(v1180, v1177, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1183 = __riscv_vzext_vf2_u16m1(v1181, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1184 = __riscv_vreinterpret_v_u16m1_i16m1(v1183);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m1
      vuint16m1_t v1185 = __riscv_vzext_vf2_u16m1(v1182, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m1_i16m1
      vint16m1_t v1186 = __riscv_vreinterpret_v_u16m1_i16m1(v1185);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_bsums_fold
      const uint8_t* v1187 = v19 + 276;
      const int16_t* v1188 = (const int16_t*) v1187;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1189 = *(const int16_t *)(v1188);
      const uint8_t* v1190 = v19 + 278;
      const int16_t* v1191 = (const int16_t*) v1190;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1192 = *(const int16_t *)(v1191);
      int32_t v1193 = v1189 + v1192;
      vint32m2_t v1194 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v1195 = __riscv_vwmacc_vx_i32m2(v1194, v1193, v1135, 16);
      v29 = v1195;
      const uint8_t* v1196 = v19 + 280;
      const int16_t* v1197 = (const int16_t*) v1196;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1198 = *(const int16_t *)(v1197);
      const uint8_t* v1199 = v19 + 282;
      const int16_t* v1200 = (const int16_t*) v1199;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1201 = *(const int16_t *)(v1200);
      int32_t v1202 = v1198 + v1201;
      vint32m2_t v1203 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v1204 = __riscv_vwmacc_vx_i32m2(v1203, v1202, v1152, 16);
      v29 = v1204;
      const uint8_t* v1205 = v19 + 284;
      const int16_t* v1206 = (const int16_t*) v1205;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1207 = *(const int16_t *)(v1206);
      const uint8_t* v1208 = v19 + 286;
      const int16_t* v1209 = (const int16_t*) v1208;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1210 = *(const int16_t *)(v1209);
      int32_t v1211 = v1207 + v1210;
      vint32m2_t v1212 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v1213 = __riscv_vwmacc_vx_i32m2(v1212, v1211, v1169, 16);
      v29 = v1213;
      const uint8_t* v1214 = v19 + 288;
      const int16_t* v1215 = (const int16_t*) v1214;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1216 = *(const int16_t *)(v1215);
      const uint8_t* v1217 = v19 + 290;
      const int16_t* v1218 = (const int16_t*) v1217;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_bsum_scalar
      int32_t v1219 = *(const int16_t *)(v1218);
      int32_t v1220 = v1216 + v1219;
      vint32m2_t v1221 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v1222 = __riscv_vwmacc_vx_i32m2(v1221, v1220, v1186, 16);
      v29 = v1222;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1223 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1224 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1225 = v17 + 1280;
      const uint8_t* v1226 = (const uint8_t*) v1225;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1227 = __riscv_vle8_v_u8mf2(v1226, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1228 = __riscv_vand_vx_u8mf2(v1227, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1229 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1228);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1230 = __riscv_vsrl_vx_u8mf2(v1227, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1231 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1230);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1232 = v19 + 132;
      const int8_t* v1233 = (const int8_t*) v1232;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1234 = *(const int8_t *)(v1233);
      const uint8_t* v1235 = v19 + 164;
      const int8_t* v1236 = (const int8_t*) v1235;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1237 = *(const int8_t *)(v1236);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1238 = __riscv_vwmacc_vx_i16m1(v1223, v1234, v1229, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1239 = __riscv_vwmacc_vx_i16m1(v1224, v1237, v1231, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1240 = v17 + 1296;
      const uint8_t* v1241 = (const uint8_t*) v1240;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1242 = __riscv_vle8_v_u8mf2(v1241, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1243 = __riscv_vand_vx_u8mf2(v1242, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1244 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1243);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1245 = __riscv_vsrl_vx_u8mf2(v1242, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1246 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1245);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1247 = v19 + 133;
      const int8_t* v1248 = (const int8_t*) v1247;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1249 = *(const int8_t *)(v1248);
      const uint8_t* v1250 = v19 + 165;
      const int8_t* v1251 = (const int8_t*) v1250;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1252 = *(const int8_t *)(v1251);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1253 = __riscv_vwmacc_vx_i16m1(v1238, v1249, v1244, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1254 = __riscv_vwmacc_vx_i16m1(v1239, v1252, v1246, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1255 = v17 + 1312;
      const uint8_t* v1256 = (const uint8_t*) v1255;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1257 = __riscv_vle8_v_u8mf2(v1256, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1258 = __riscv_vand_vx_u8mf2(v1257, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1259 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1258);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1260 = __riscv_vsrl_vx_u8mf2(v1257, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1261 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1260);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1262 = v19 + 134;
      const int8_t* v1263 = (const int8_t*) v1262;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1264 = *(const int8_t *)(v1263);
      const uint8_t* v1265 = v19 + 166;
      const int8_t* v1266 = (const int8_t*) v1265;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1267 = *(const int8_t *)(v1266);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1268 = __riscv_vwmacc_vx_i16m1(v1253, v1264, v1259, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1269 = __riscv_vwmacc_vx_i16m1(v1254, v1267, v1261, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1270 = v17 + 1328;
      const uint8_t* v1271 = (const uint8_t*) v1270;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1272 = __riscv_vle8_v_u8mf2(v1271, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1273 = __riscv_vand_vx_u8mf2(v1272, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1274 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1273);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1275 = __riscv_vsrl_vx_u8mf2(v1272, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1276 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1275);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1277 = v19 + 135;
      const int8_t* v1278 = (const int8_t*) v1277;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1279 = *(const int8_t *)(v1278);
      const uint8_t* v1280 = v19 + 167;
      const int8_t* v1281 = (const int8_t*) v1280;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1282 = *(const int8_t *)(v1281);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1283 = __riscv_vwmacc_vx_i16m1(v1268, v1279, v1274, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1284 = __riscv_vwmacc_vx_i16m1(v1269, v1282, v1276, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1285 = v17 + 1344;
      const uint8_t* v1286 = (const uint8_t*) v1285;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1287 = __riscv_vle8_v_u8mf2(v1286, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1288 = __riscv_vand_vx_u8mf2(v1287, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1289 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1288);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1290 = __riscv_vsrl_vx_u8mf2(v1287, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1291 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1290);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1292 = v19 + 136;
      const int8_t* v1293 = (const int8_t*) v1292;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1294 = *(const int8_t *)(v1293);
      const uint8_t* v1295 = v19 + 168;
      const int8_t* v1296 = (const int8_t*) v1295;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1297 = *(const int8_t *)(v1296);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1298 = __riscv_vwmacc_vx_i16m1(v1283, v1294, v1289, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1299 = __riscv_vwmacc_vx_i16m1(v1284, v1297, v1291, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1300 = v17 + 1360;
      const uint8_t* v1301 = (const uint8_t*) v1300;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1302 = __riscv_vle8_v_u8mf2(v1301, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1303 = __riscv_vand_vx_u8mf2(v1302, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1304 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1303);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1305 = __riscv_vsrl_vx_u8mf2(v1302, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1306 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1305);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1307 = v19 + 137;
      const int8_t* v1308 = (const int8_t*) v1307;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1309 = *(const int8_t *)(v1308);
      const uint8_t* v1310 = v19 + 169;
      const int8_t* v1311 = (const int8_t*) v1310;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1312 = *(const int8_t *)(v1311);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1313 = __riscv_vwmacc_vx_i16m1(v1298, v1309, v1304, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1314 = __riscv_vwmacc_vx_i16m1(v1299, v1312, v1306, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1315 = v17 + 1376;
      const uint8_t* v1316 = (const uint8_t*) v1315;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1317 = __riscv_vle8_v_u8mf2(v1316, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1318 = __riscv_vand_vx_u8mf2(v1317, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1319 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1318);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1320 = __riscv_vsrl_vx_u8mf2(v1317, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1321 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1320);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1322 = v19 + 138;
      const int8_t* v1323 = (const int8_t*) v1322;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1324 = *(const int8_t *)(v1323);
      const uint8_t* v1325 = v19 + 170;
      const int8_t* v1326 = (const int8_t*) v1325;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1327 = *(const int8_t *)(v1326);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1328 = __riscv_vwmacc_vx_i16m1(v1313, v1324, v1319, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1329 = __riscv_vwmacc_vx_i16m1(v1314, v1327, v1321, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1330 = v17 + 1392;
      const uint8_t* v1331 = (const uint8_t*) v1330;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1332 = __riscv_vle8_v_u8mf2(v1331, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1333 = __riscv_vand_vx_u8mf2(v1332, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1334 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1333);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1335 = __riscv_vsrl_vx_u8mf2(v1332, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1336 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1335);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1337 = v19 + 139;
      const int8_t* v1338 = (const int8_t*) v1337;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1339 = *(const int8_t *)(v1338);
      const uint8_t* v1340 = v19 + 171;
      const int8_t* v1341 = (const int8_t*) v1340;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1342 = *(const int8_t *)(v1341);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1343 = __riscv_vwmacc_vx_i16m1(v1328, v1339, v1334, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1344 = __riscv_vwmacc_vx_i16m1(v1329, v1342, v1336, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1345 = v17 + 1408;
      const uint8_t* v1346 = (const uint8_t*) v1345;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1347 = __riscv_vle8_v_u8mf2(v1346, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1348 = __riscv_vand_vx_u8mf2(v1347, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1349 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1348);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1350 = __riscv_vsrl_vx_u8mf2(v1347, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1351 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1350);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1352 = v19 + 140;
      const int8_t* v1353 = (const int8_t*) v1352;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1354 = *(const int8_t *)(v1353);
      const uint8_t* v1355 = v19 + 172;
      const int8_t* v1356 = (const int8_t*) v1355;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1357 = *(const int8_t *)(v1356);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1358 = __riscv_vwmacc_vx_i16m1(v1343, v1354, v1349, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1359 = __riscv_vwmacc_vx_i16m1(v1344, v1357, v1351, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1360 = v17 + 1424;
      const uint8_t* v1361 = (const uint8_t*) v1360;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1362 = __riscv_vle8_v_u8mf2(v1361, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1363 = __riscv_vand_vx_u8mf2(v1362, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1364 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1363);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1365 = __riscv_vsrl_vx_u8mf2(v1362, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1366 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1365);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1367 = v19 + 141;
      const int8_t* v1368 = (const int8_t*) v1367;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1369 = *(const int8_t *)(v1368);
      const uint8_t* v1370 = v19 + 173;
      const int8_t* v1371 = (const int8_t*) v1370;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1372 = *(const int8_t *)(v1371);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1373 = __riscv_vwmacc_vx_i16m1(v1358, v1369, v1364, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1374 = __riscv_vwmacc_vx_i16m1(v1359, v1372, v1366, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1375 = v17 + 1440;
      const uint8_t* v1376 = (const uint8_t*) v1375;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1377 = __riscv_vle8_v_u8mf2(v1376, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1378 = __riscv_vand_vx_u8mf2(v1377, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1379 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1378);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1380 = __riscv_vsrl_vx_u8mf2(v1377, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1381 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1380);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1382 = v19 + 142;
      const int8_t* v1383 = (const int8_t*) v1382;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1384 = *(const int8_t *)(v1383);
      const uint8_t* v1385 = v19 + 174;
      const int8_t* v1386 = (const int8_t*) v1385;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1387 = *(const int8_t *)(v1386);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1388 = __riscv_vwmacc_vx_i16m1(v1373, v1384, v1379, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1389 = __riscv_vwmacc_vx_i16m1(v1374, v1387, v1381, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1390 = v17 + 1456;
      const uint8_t* v1391 = (const uint8_t*) v1390;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1392 = __riscv_vle8_v_u8mf2(v1391, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1393 = __riscv_vand_vx_u8mf2(v1392, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1394 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1393);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1395 = __riscv_vsrl_vx_u8mf2(v1392, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1396 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1395);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1397 = v19 + 143;
      const int8_t* v1398 = (const int8_t*) v1397;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1399 = *(const int8_t *)(v1398);
      const uint8_t* v1400 = v19 + 175;
      const int8_t* v1401 = (const int8_t*) v1400;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1402 = *(const int8_t *)(v1401);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1403 = __riscv_vwmacc_vx_i16m1(v1388, v1399, v1394, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1404 = __riscv_vwmacc_vx_i16m1(v1389, v1402, v1396, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1405 = v17 + 1472;
      const uint8_t* v1406 = (const uint8_t*) v1405;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1407 = __riscv_vle8_v_u8mf2(v1406, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1408 = __riscv_vand_vx_u8mf2(v1407, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1409 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1408);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1410 = __riscv_vsrl_vx_u8mf2(v1407, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1411 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1410);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1412 = v19 + 144;
      const int8_t* v1413 = (const int8_t*) v1412;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1414 = *(const int8_t *)(v1413);
      const uint8_t* v1415 = v19 + 176;
      const int8_t* v1416 = (const int8_t*) v1415;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1417 = *(const int8_t *)(v1416);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1418 = __riscv_vwmacc_vx_i16m1(v1403, v1414, v1409, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1419 = __riscv_vwmacc_vx_i16m1(v1404, v1417, v1411, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1420 = v17 + 1488;
      const uint8_t* v1421 = (const uint8_t*) v1420;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1422 = __riscv_vle8_v_u8mf2(v1421, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1423 = __riscv_vand_vx_u8mf2(v1422, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1424 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1423);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1425 = __riscv_vsrl_vx_u8mf2(v1422, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1426 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1425);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1427 = v19 + 145;
      const int8_t* v1428 = (const int8_t*) v1427;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1429 = *(const int8_t *)(v1428);
      const uint8_t* v1430 = v19 + 177;
      const int8_t* v1431 = (const int8_t*) v1430;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1432 = *(const int8_t *)(v1431);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1433 = __riscv_vwmacc_vx_i16m1(v1418, v1429, v1424, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1434 = __riscv_vwmacc_vx_i16m1(v1419, v1432, v1426, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1435 = v17 + 1504;
      const uint8_t* v1436 = (const uint8_t*) v1435;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1437 = __riscv_vle8_v_u8mf2(v1436, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1438 = __riscv_vand_vx_u8mf2(v1437, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1439 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1438);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1440 = __riscv_vsrl_vx_u8mf2(v1437, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1441 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1440);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1442 = v19 + 146;
      const int8_t* v1443 = (const int8_t*) v1442;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1444 = *(const int8_t *)(v1443);
      const uint8_t* v1445 = v19 + 178;
      const int8_t* v1446 = (const int8_t*) v1445;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1447 = *(const int8_t *)(v1446);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1448 = __riscv_vwmacc_vx_i16m1(v1433, v1444, v1439, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1449 = __riscv_vwmacc_vx_i16m1(v1434, v1447, v1441, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1450 = v17 + 1520;
      const uint8_t* v1451 = (const uint8_t*) v1450;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1452 = __riscv_vle8_v_u8mf2(v1451, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1453 = __riscv_vand_vx_u8mf2(v1452, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1454 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1453);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1455 = __riscv_vsrl_vx_u8mf2(v1452, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1456 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1455);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1457 = v19 + 147;
      const int8_t* v1458 = (const int8_t*) v1457;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1459 = *(const int8_t *)(v1458);
      const uint8_t* v1460 = v19 + 179;
      const int8_t* v1461 = (const int8_t*) v1460;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1462 = *(const int8_t *)(v1461);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1463 = __riscv_vwmacc_vx_i16m1(v1448, v1459, v1454, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1464 = __riscv_vwmacc_vx_i16m1(v1449, v1462, v1456, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1465 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1466 = __riscv_vwmacc_vv_i32m2(v1465, v1133, v1463, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1467 = __riscv_vwmacc_vv_i32m2(v1466, v1150, v1464, 16);
      v27 = v1467;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1468 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1469 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1470 = v17 + 1536;
      const uint8_t* v1471 = (const uint8_t*) v1470;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1472 = __riscv_vle8_v_u8mf2(v1471, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1473 = __riscv_vand_vx_u8mf2(v1472, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1474 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1473);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1475 = __riscv_vsrl_vx_u8mf2(v1472, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1476 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1475);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1477 = v19 + 148;
      const int8_t* v1478 = (const int8_t*) v1477;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1479 = *(const int8_t *)(v1478);
      const uint8_t* v1480 = v19 + 180;
      const int8_t* v1481 = (const int8_t*) v1480;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1482 = *(const int8_t *)(v1481);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1483 = __riscv_vwmacc_vx_i16m1(v1468, v1479, v1474, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1484 = __riscv_vwmacc_vx_i16m1(v1469, v1482, v1476, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1485 = v17 + 1552;
      const uint8_t* v1486 = (const uint8_t*) v1485;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1487 = __riscv_vle8_v_u8mf2(v1486, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1488 = __riscv_vand_vx_u8mf2(v1487, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1489 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1488);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1490 = __riscv_vsrl_vx_u8mf2(v1487, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1491 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1490);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1492 = v19 + 149;
      const int8_t* v1493 = (const int8_t*) v1492;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1494 = *(const int8_t *)(v1493);
      const uint8_t* v1495 = v19 + 181;
      const int8_t* v1496 = (const int8_t*) v1495;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1497 = *(const int8_t *)(v1496);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1498 = __riscv_vwmacc_vx_i16m1(v1483, v1494, v1489, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1499 = __riscv_vwmacc_vx_i16m1(v1484, v1497, v1491, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1500 = v17 + 1568;
      const uint8_t* v1501 = (const uint8_t*) v1500;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1502 = __riscv_vle8_v_u8mf2(v1501, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1503 = __riscv_vand_vx_u8mf2(v1502, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1504 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1503);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1505 = __riscv_vsrl_vx_u8mf2(v1502, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1506 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1505);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1507 = v19 + 150;
      const int8_t* v1508 = (const int8_t*) v1507;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1509 = *(const int8_t *)(v1508);
      const uint8_t* v1510 = v19 + 182;
      const int8_t* v1511 = (const int8_t*) v1510;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1512 = *(const int8_t *)(v1511);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1513 = __riscv_vwmacc_vx_i16m1(v1498, v1509, v1504, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1514 = __riscv_vwmacc_vx_i16m1(v1499, v1512, v1506, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1515 = v17 + 1584;
      const uint8_t* v1516 = (const uint8_t*) v1515;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1517 = __riscv_vle8_v_u8mf2(v1516, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1518 = __riscv_vand_vx_u8mf2(v1517, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1519 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1518);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1520 = __riscv_vsrl_vx_u8mf2(v1517, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1521 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1520);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1522 = v19 + 151;
      const int8_t* v1523 = (const int8_t*) v1522;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1524 = *(const int8_t *)(v1523);
      const uint8_t* v1525 = v19 + 183;
      const int8_t* v1526 = (const int8_t*) v1525;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1527 = *(const int8_t *)(v1526);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1528 = __riscv_vwmacc_vx_i16m1(v1513, v1524, v1519, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1529 = __riscv_vwmacc_vx_i16m1(v1514, v1527, v1521, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1530 = v17 + 1600;
      const uint8_t* v1531 = (const uint8_t*) v1530;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1532 = __riscv_vle8_v_u8mf2(v1531, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1533 = __riscv_vand_vx_u8mf2(v1532, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1534 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1533);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1535 = __riscv_vsrl_vx_u8mf2(v1532, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1536 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1535);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1537 = v19 + 152;
      const int8_t* v1538 = (const int8_t*) v1537;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1539 = *(const int8_t *)(v1538);
      const uint8_t* v1540 = v19 + 184;
      const int8_t* v1541 = (const int8_t*) v1540;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1542 = *(const int8_t *)(v1541);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1543 = __riscv_vwmacc_vx_i16m1(v1528, v1539, v1534, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1544 = __riscv_vwmacc_vx_i16m1(v1529, v1542, v1536, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1545 = v17 + 1616;
      const uint8_t* v1546 = (const uint8_t*) v1545;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1547 = __riscv_vle8_v_u8mf2(v1546, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1548 = __riscv_vand_vx_u8mf2(v1547, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1549 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1548);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1550 = __riscv_vsrl_vx_u8mf2(v1547, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1551 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1550);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1552 = v19 + 153;
      const int8_t* v1553 = (const int8_t*) v1552;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1554 = *(const int8_t *)(v1553);
      const uint8_t* v1555 = v19 + 185;
      const int8_t* v1556 = (const int8_t*) v1555;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1557 = *(const int8_t *)(v1556);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1558 = __riscv_vwmacc_vx_i16m1(v1543, v1554, v1549, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1559 = __riscv_vwmacc_vx_i16m1(v1544, v1557, v1551, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1560 = v17 + 1632;
      const uint8_t* v1561 = (const uint8_t*) v1560;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1562 = __riscv_vle8_v_u8mf2(v1561, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1563 = __riscv_vand_vx_u8mf2(v1562, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1564 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1563);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1565 = __riscv_vsrl_vx_u8mf2(v1562, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1566 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1565);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1567 = v19 + 154;
      const int8_t* v1568 = (const int8_t*) v1567;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1569 = *(const int8_t *)(v1568);
      const uint8_t* v1570 = v19 + 186;
      const int8_t* v1571 = (const int8_t*) v1570;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1572 = *(const int8_t *)(v1571);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1573 = __riscv_vwmacc_vx_i16m1(v1558, v1569, v1564, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1574 = __riscv_vwmacc_vx_i16m1(v1559, v1572, v1566, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1575 = v17 + 1648;
      const uint8_t* v1576 = (const uint8_t*) v1575;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1577 = __riscv_vle8_v_u8mf2(v1576, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1578 = __riscv_vand_vx_u8mf2(v1577, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1579 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1578);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1580 = __riscv_vsrl_vx_u8mf2(v1577, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1581 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1580);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1582 = v19 + 155;
      const int8_t* v1583 = (const int8_t*) v1582;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1584 = *(const int8_t *)(v1583);
      const uint8_t* v1585 = v19 + 187;
      const int8_t* v1586 = (const int8_t*) v1585;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1587 = *(const int8_t *)(v1586);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1588 = __riscv_vwmacc_vx_i16m1(v1573, v1584, v1579, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1589 = __riscv_vwmacc_vx_i16m1(v1574, v1587, v1581, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1590 = v17 + 1664;
      const uint8_t* v1591 = (const uint8_t*) v1590;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1592 = __riscv_vle8_v_u8mf2(v1591, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1593 = __riscv_vand_vx_u8mf2(v1592, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1594 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1593);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1595 = __riscv_vsrl_vx_u8mf2(v1592, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1596 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1595);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1597 = v19 + 156;
      const int8_t* v1598 = (const int8_t*) v1597;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1599 = *(const int8_t *)(v1598);
      const uint8_t* v1600 = v19 + 188;
      const int8_t* v1601 = (const int8_t*) v1600;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1602 = *(const int8_t *)(v1601);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1603 = __riscv_vwmacc_vx_i16m1(v1588, v1599, v1594, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1604 = __riscv_vwmacc_vx_i16m1(v1589, v1602, v1596, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1605 = v17 + 1680;
      const uint8_t* v1606 = (const uint8_t*) v1605;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1607 = __riscv_vle8_v_u8mf2(v1606, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1608 = __riscv_vand_vx_u8mf2(v1607, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1609 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1608);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1610 = __riscv_vsrl_vx_u8mf2(v1607, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1611 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1610);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1612 = v19 + 157;
      const int8_t* v1613 = (const int8_t*) v1612;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1614 = *(const int8_t *)(v1613);
      const uint8_t* v1615 = v19 + 189;
      const int8_t* v1616 = (const int8_t*) v1615;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1617 = *(const int8_t *)(v1616);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1618 = __riscv_vwmacc_vx_i16m1(v1603, v1614, v1609, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1619 = __riscv_vwmacc_vx_i16m1(v1604, v1617, v1611, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1620 = v17 + 1696;
      const uint8_t* v1621 = (const uint8_t*) v1620;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1622 = __riscv_vle8_v_u8mf2(v1621, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1623 = __riscv_vand_vx_u8mf2(v1622, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1624 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1623);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1625 = __riscv_vsrl_vx_u8mf2(v1622, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1626 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1625);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1627 = v19 + 158;
      const int8_t* v1628 = (const int8_t*) v1627;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1629 = *(const int8_t *)(v1628);
      const uint8_t* v1630 = v19 + 190;
      const int8_t* v1631 = (const int8_t*) v1630;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1632 = *(const int8_t *)(v1631);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1633 = __riscv_vwmacc_vx_i16m1(v1618, v1629, v1624, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1634 = __riscv_vwmacc_vx_i16m1(v1619, v1632, v1626, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1635 = v17 + 1712;
      const uint8_t* v1636 = (const uint8_t*) v1635;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1637 = __riscv_vle8_v_u8mf2(v1636, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1638 = __riscv_vand_vx_u8mf2(v1637, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1639 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1638);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1640 = __riscv_vsrl_vx_u8mf2(v1637, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1641 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1640);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1642 = v19 + 159;
      const int8_t* v1643 = (const int8_t*) v1642;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1644 = *(const int8_t *)(v1643);
      const uint8_t* v1645 = v19 + 191;
      const int8_t* v1646 = (const int8_t*) v1645;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1647 = *(const int8_t *)(v1646);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1648 = __riscv_vwmacc_vx_i16m1(v1633, v1644, v1639, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1649 = __riscv_vwmacc_vx_i16m1(v1634, v1647, v1641, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1650 = v17 + 1728;
      const uint8_t* v1651 = (const uint8_t*) v1650;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1652 = __riscv_vle8_v_u8mf2(v1651, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1653 = __riscv_vand_vx_u8mf2(v1652, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1654 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1653);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1655 = __riscv_vsrl_vx_u8mf2(v1652, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1656 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1655);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1657 = v19 + 160;
      const int8_t* v1658 = (const int8_t*) v1657;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1659 = *(const int8_t *)(v1658);
      const uint8_t* v1660 = v19 + 192;
      const int8_t* v1661 = (const int8_t*) v1660;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1662 = *(const int8_t *)(v1661);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1663 = __riscv_vwmacc_vx_i16m1(v1648, v1659, v1654, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1664 = __riscv_vwmacc_vx_i16m1(v1649, v1662, v1656, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1665 = v17 + 1744;
      const uint8_t* v1666 = (const uint8_t*) v1665;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1667 = __riscv_vle8_v_u8mf2(v1666, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1668 = __riscv_vand_vx_u8mf2(v1667, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1669 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1668);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1670 = __riscv_vsrl_vx_u8mf2(v1667, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1671 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1670);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1672 = v19 + 161;
      const int8_t* v1673 = (const int8_t*) v1672;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1674 = *(const int8_t *)(v1673);
      const uint8_t* v1675 = v19 + 193;
      const int8_t* v1676 = (const int8_t*) v1675;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1677 = *(const int8_t *)(v1676);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1678 = __riscv_vwmacc_vx_i16m1(v1663, v1674, v1669, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1679 = __riscv_vwmacc_vx_i16m1(v1664, v1677, v1671, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1680 = v17 + 1760;
      const uint8_t* v1681 = (const uint8_t*) v1680;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1682 = __riscv_vle8_v_u8mf2(v1681, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1683 = __riscv_vand_vx_u8mf2(v1682, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1684 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1683);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1685 = __riscv_vsrl_vx_u8mf2(v1682, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1686 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1685);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1687 = v19 + 162;
      const int8_t* v1688 = (const int8_t*) v1687;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1689 = *(const int8_t *)(v1688);
      const uint8_t* v1690 = v19 + 194;
      const int8_t* v1691 = (const int8_t*) v1690;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1692 = *(const int8_t *)(v1691);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1693 = __riscv_vwmacc_vx_i16m1(v1678, v1689, v1684, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1694 = __riscv_vwmacc_vx_i16m1(v1679, v1692, v1686, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1695 = v17 + 1776;
      const uint8_t* v1696 = (const uint8_t*) v1695;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1697 = __riscv_vle8_v_u8mf2(v1696, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1698 = __riscv_vand_vx_u8mf2(v1697, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1699 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1698);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1700 = __riscv_vsrl_vx_u8mf2(v1697, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1701 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1700);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1702 = v19 + 163;
      const int8_t* v1703 = (const int8_t*) v1702;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1704 = *(const int8_t *)(v1703);
      const uint8_t* v1705 = v19 + 195;
      const int8_t* v1706 = (const int8_t*) v1705;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1707 = *(const int8_t *)(v1706);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1708 = __riscv_vwmacc_vx_i16m1(v1693, v1704, v1699, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1709 = __riscv_vwmacc_vx_i16m1(v1694, v1707, v1701, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1710 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1711 = __riscv_vwmacc_vv_i32m2(v1710, v1133, v1708, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1712 = __riscv_vwmacc_vv_i32m2(v1711, v1150, v1709, 16);
      v27 = v1712;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1713 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1714 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1715 = v17 + 1792;
      const uint8_t* v1716 = (const uint8_t*) v1715;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1717 = __riscv_vle8_v_u8mf2(v1716, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1718 = __riscv_vand_vx_u8mf2(v1717, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1719 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1718);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1720 = __riscv_vsrl_vx_u8mf2(v1717, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1721 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1720);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1722 = v19 + 196;
      const int8_t* v1723 = (const int8_t*) v1722;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1724 = *(const int8_t *)(v1723);
      const uint8_t* v1725 = v19 + 228;
      const int8_t* v1726 = (const int8_t*) v1725;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1727 = *(const int8_t *)(v1726);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1728 = __riscv_vwmacc_vx_i16m1(v1713, v1724, v1719, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1729 = __riscv_vwmacc_vx_i16m1(v1714, v1727, v1721, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1730 = v17 + 1808;
      const uint8_t* v1731 = (const uint8_t*) v1730;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1732 = __riscv_vle8_v_u8mf2(v1731, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1733 = __riscv_vand_vx_u8mf2(v1732, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1734 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1733);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1735 = __riscv_vsrl_vx_u8mf2(v1732, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1736 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1735);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1737 = v19 + 197;
      const int8_t* v1738 = (const int8_t*) v1737;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1739 = *(const int8_t *)(v1738);
      const uint8_t* v1740 = v19 + 229;
      const int8_t* v1741 = (const int8_t*) v1740;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1742 = *(const int8_t *)(v1741);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1743 = __riscv_vwmacc_vx_i16m1(v1728, v1739, v1734, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1744 = __riscv_vwmacc_vx_i16m1(v1729, v1742, v1736, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1745 = v17 + 1824;
      const uint8_t* v1746 = (const uint8_t*) v1745;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1747 = __riscv_vle8_v_u8mf2(v1746, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1748 = __riscv_vand_vx_u8mf2(v1747, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1749 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1748);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1750 = __riscv_vsrl_vx_u8mf2(v1747, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1751 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1750);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1752 = v19 + 198;
      const int8_t* v1753 = (const int8_t*) v1752;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1754 = *(const int8_t *)(v1753);
      const uint8_t* v1755 = v19 + 230;
      const int8_t* v1756 = (const int8_t*) v1755;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1757 = *(const int8_t *)(v1756);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1758 = __riscv_vwmacc_vx_i16m1(v1743, v1754, v1749, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1759 = __riscv_vwmacc_vx_i16m1(v1744, v1757, v1751, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1760 = v17 + 1840;
      const uint8_t* v1761 = (const uint8_t*) v1760;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1762 = __riscv_vle8_v_u8mf2(v1761, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1763 = __riscv_vand_vx_u8mf2(v1762, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1764 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1763);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1765 = __riscv_vsrl_vx_u8mf2(v1762, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1766 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1765);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1767 = v19 + 199;
      const int8_t* v1768 = (const int8_t*) v1767;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1769 = *(const int8_t *)(v1768);
      const uint8_t* v1770 = v19 + 231;
      const int8_t* v1771 = (const int8_t*) v1770;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1772 = *(const int8_t *)(v1771);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1773 = __riscv_vwmacc_vx_i16m1(v1758, v1769, v1764, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1774 = __riscv_vwmacc_vx_i16m1(v1759, v1772, v1766, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1775 = v17 + 1856;
      const uint8_t* v1776 = (const uint8_t*) v1775;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1777 = __riscv_vle8_v_u8mf2(v1776, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1778 = __riscv_vand_vx_u8mf2(v1777, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1779 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1778);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1780 = __riscv_vsrl_vx_u8mf2(v1777, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1781 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1780);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1782 = v19 + 200;
      const int8_t* v1783 = (const int8_t*) v1782;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1784 = *(const int8_t *)(v1783);
      const uint8_t* v1785 = v19 + 232;
      const int8_t* v1786 = (const int8_t*) v1785;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1787 = *(const int8_t *)(v1786);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1788 = __riscv_vwmacc_vx_i16m1(v1773, v1784, v1779, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1789 = __riscv_vwmacc_vx_i16m1(v1774, v1787, v1781, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1790 = v17 + 1872;
      const uint8_t* v1791 = (const uint8_t*) v1790;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1792 = __riscv_vle8_v_u8mf2(v1791, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1793 = __riscv_vand_vx_u8mf2(v1792, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1794 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1793);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1795 = __riscv_vsrl_vx_u8mf2(v1792, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1796 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1795);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1797 = v19 + 201;
      const int8_t* v1798 = (const int8_t*) v1797;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1799 = *(const int8_t *)(v1798);
      const uint8_t* v1800 = v19 + 233;
      const int8_t* v1801 = (const int8_t*) v1800;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1802 = *(const int8_t *)(v1801);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1803 = __riscv_vwmacc_vx_i16m1(v1788, v1799, v1794, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1804 = __riscv_vwmacc_vx_i16m1(v1789, v1802, v1796, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1805 = v17 + 1888;
      const uint8_t* v1806 = (const uint8_t*) v1805;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1807 = __riscv_vle8_v_u8mf2(v1806, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1808 = __riscv_vand_vx_u8mf2(v1807, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1809 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1808);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1810 = __riscv_vsrl_vx_u8mf2(v1807, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1811 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1810);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1812 = v19 + 202;
      const int8_t* v1813 = (const int8_t*) v1812;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1814 = *(const int8_t *)(v1813);
      const uint8_t* v1815 = v19 + 234;
      const int8_t* v1816 = (const int8_t*) v1815;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1817 = *(const int8_t *)(v1816);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1818 = __riscv_vwmacc_vx_i16m1(v1803, v1814, v1809, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1819 = __riscv_vwmacc_vx_i16m1(v1804, v1817, v1811, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1820 = v17 + 1904;
      const uint8_t* v1821 = (const uint8_t*) v1820;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1822 = __riscv_vle8_v_u8mf2(v1821, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1823 = __riscv_vand_vx_u8mf2(v1822, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1824 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1823);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1825 = __riscv_vsrl_vx_u8mf2(v1822, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1826 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1825);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1827 = v19 + 203;
      const int8_t* v1828 = (const int8_t*) v1827;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1829 = *(const int8_t *)(v1828);
      const uint8_t* v1830 = v19 + 235;
      const int8_t* v1831 = (const int8_t*) v1830;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1832 = *(const int8_t *)(v1831);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1833 = __riscv_vwmacc_vx_i16m1(v1818, v1829, v1824, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1834 = __riscv_vwmacc_vx_i16m1(v1819, v1832, v1826, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1835 = v17 + 1920;
      const uint8_t* v1836 = (const uint8_t*) v1835;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1837 = __riscv_vle8_v_u8mf2(v1836, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1838 = __riscv_vand_vx_u8mf2(v1837, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1839 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1838);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1840 = __riscv_vsrl_vx_u8mf2(v1837, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1841 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1840);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1842 = v19 + 204;
      const int8_t* v1843 = (const int8_t*) v1842;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1844 = *(const int8_t *)(v1843);
      const uint8_t* v1845 = v19 + 236;
      const int8_t* v1846 = (const int8_t*) v1845;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1847 = *(const int8_t *)(v1846);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1848 = __riscv_vwmacc_vx_i16m1(v1833, v1844, v1839, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1849 = __riscv_vwmacc_vx_i16m1(v1834, v1847, v1841, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1850 = v17 + 1936;
      const uint8_t* v1851 = (const uint8_t*) v1850;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1852 = __riscv_vle8_v_u8mf2(v1851, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1853 = __riscv_vand_vx_u8mf2(v1852, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1854 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1853);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1855 = __riscv_vsrl_vx_u8mf2(v1852, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1856 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1855);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1857 = v19 + 205;
      const int8_t* v1858 = (const int8_t*) v1857;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1859 = *(const int8_t *)(v1858);
      const uint8_t* v1860 = v19 + 237;
      const int8_t* v1861 = (const int8_t*) v1860;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1862 = *(const int8_t *)(v1861);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1863 = __riscv_vwmacc_vx_i16m1(v1848, v1859, v1854, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1864 = __riscv_vwmacc_vx_i16m1(v1849, v1862, v1856, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1865 = v17 + 1952;
      const uint8_t* v1866 = (const uint8_t*) v1865;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1867 = __riscv_vle8_v_u8mf2(v1866, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1868 = __riscv_vand_vx_u8mf2(v1867, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1869 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1868);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1870 = __riscv_vsrl_vx_u8mf2(v1867, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1871 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1870);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1872 = v19 + 206;
      const int8_t* v1873 = (const int8_t*) v1872;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1874 = *(const int8_t *)(v1873);
      const uint8_t* v1875 = v19 + 238;
      const int8_t* v1876 = (const int8_t*) v1875;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1877 = *(const int8_t *)(v1876);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1878 = __riscv_vwmacc_vx_i16m1(v1863, v1874, v1869, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1879 = __riscv_vwmacc_vx_i16m1(v1864, v1877, v1871, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1880 = v17 + 1968;
      const uint8_t* v1881 = (const uint8_t*) v1880;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1882 = __riscv_vle8_v_u8mf2(v1881, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1883 = __riscv_vand_vx_u8mf2(v1882, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1884 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1883);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1885 = __riscv_vsrl_vx_u8mf2(v1882, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1886 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1885);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1887 = v19 + 207;
      const int8_t* v1888 = (const int8_t*) v1887;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1889 = *(const int8_t *)(v1888);
      const uint8_t* v1890 = v19 + 239;
      const int8_t* v1891 = (const int8_t*) v1890;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1892 = *(const int8_t *)(v1891);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1893 = __riscv_vwmacc_vx_i16m1(v1878, v1889, v1884, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1894 = __riscv_vwmacc_vx_i16m1(v1879, v1892, v1886, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1895 = v17 + 1984;
      const uint8_t* v1896 = (const uint8_t*) v1895;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1897 = __riscv_vle8_v_u8mf2(v1896, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1898 = __riscv_vand_vx_u8mf2(v1897, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1899 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1898);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1900 = __riscv_vsrl_vx_u8mf2(v1897, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1901 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1900);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1902 = v19 + 208;
      const int8_t* v1903 = (const int8_t*) v1902;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1904 = *(const int8_t *)(v1903);
      const uint8_t* v1905 = v19 + 240;
      const int8_t* v1906 = (const int8_t*) v1905;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1907 = *(const int8_t *)(v1906);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1908 = __riscv_vwmacc_vx_i16m1(v1893, v1904, v1899, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1909 = __riscv_vwmacc_vx_i16m1(v1894, v1907, v1901, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1910 = v17 + 2000;
      const uint8_t* v1911 = (const uint8_t*) v1910;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1912 = __riscv_vle8_v_u8mf2(v1911, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1913 = __riscv_vand_vx_u8mf2(v1912, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1914 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1913);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1915 = __riscv_vsrl_vx_u8mf2(v1912, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1916 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1915);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1917 = v19 + 209;
      const int8_t* v1918 = (const int8_t*) v1917;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1919 = *(const int8_t *)(v1918);
      const uint8_t* v1920 = v19 + 241;
      const int8_t* v1921 = (const int8_t*) v1920;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1922 = *(const int8_t *)(v1921);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1923 = __riscv_vwmacc_vx_i16m1(v1908, v1919, v1914, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1924 = __riscv_vwmacc_vx_i16m1(v1909, v1922, v1916, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1925 = v17 + 2016;
      const uint8_t* v1926 = (const uint8_t*) v1925;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1927 = __riscv_vle8_v_u8mf2(v1926, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1928 = __riscv_vand_vx_u8mf2(v1927, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1929 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1928);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1930 = __riscv_vsrl_vx_u8mf2(v1927, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1931 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1930);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1932 = v19 + 210;
      const int8_t* v1933 = (const int8_t*) v1932;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1934 = *(const int8_t *)(v1933);
      const uint8_t* v1935 = v19 + 242;
      const int8_t* v1936 = (const int8_t*) v1935;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1937 = *(const int8_t *)(v1936);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1938 = __riscv_vwmacc_vx_i16m1(v1923, v1934, v1929, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1939 = __riscv_vwmacc_vx_i16m1(v1924, v1937, v1931, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1940 = v17 + 2032;
      const uint8_t* v1941 = (const uint8_t*) v1940;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1942 = __riscv_vle8_v_u8mf2(v1941, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1943 = __riscv_vand_vx_u8mf2(v1942, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1944 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1943);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1945 = __riscv_vsrl_vx_u8mf2(v1942, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1946 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1945);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1947 = v19 + 211;
      const int8_t* v1948 = (const int8_t*) v1947;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1949 = *(const int8_t *)(v1948);
      const uint8_t* v1950 = v19 + 243;
      const int8_t* v1951 = (const int8_t*) v1950;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1952 = *(const int8_t *)(v1951);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1953 = __riscv_vwmacc_vx_i16m1(v1938, v1949, v1944, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1954 = __riscv_vwmacc_vx_i16m1(v1939, v1952, v1946, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v1955 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1956 = __riscv_vwmacc_vv_i32m2(v1955, v1167, v1953, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v1957 = __riscv_vwmacc_vv_i32m2(v1956, v1184, v1954, 16);
      v27 = v1957;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1958 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v1959 = __riscv_vmv_v_x_i16m1(0, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1960 = v17 + 2048;
      const uint8_t* v1961 = (const uint8_t*) v1960;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1962 = __riscv_vle8_v_u8mf2(v1961, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1963 = __riscv_vand_vx_u8mf2(v1962, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1964 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1963);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1965 = __riscv_vsrl_vx_u8mf2(v1962, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1966 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1965);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1967 = v19 + 212;
      const int8_t* v1968 = (const int8_t*) v1967;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1969 = *(const int8_t *)(v1968);
      const uint8_t* v1970 = v19 + 244;
      const int8_t* v1971 = (const int8_t*) v1970;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1972 = *(const int8_t *)(v1971);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1973 = __riscv_vwmacc_vx_i16m1(v1958, v1969, v1964, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1974 = __riscv_vwmacc_vx_i16m1(v1959, v1972, v1966, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1975 = v17 + 2064;
      const uint8_t* v1976 = (const uint8_t*) v1975;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1977 = __riscv_vle8_v_u8mf2(v1976, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1978 = __riscv_vand_vx_u8mf2(v1977, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1979 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1978);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1980 = __riscv_vsrl_vx_u8mf2(v1977, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1981 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1980);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1982 = v19 + 213;
      const int8_t* v1983 = (const int8_t*) v1982;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1984 = *(const int8_t *)(v1983);
      const uint8_t* v1985 = v19 + 245;
      const int8_t* v1986 = (const int8_t*) v1985;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1987 = *(const int8_t *)(v1986);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1988 = __riscv_vwmacc_vx_i16m1(v1973, v1984, v1979, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v1989 = __riscv_vwmacc_vx_i16m1(v1974, v1987, v1981, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v1990 = v17 + 2080;
      const uint8_t* v1991 = (const uint8_t*) v1990;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v1992 = __riscv_vle8_v_u8mf2(v1991, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v1993 = __riscv_vand_vx_u8mf2(v1992, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1994 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1993);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v1995 = __riscv_vsrl_vx_u8mf2(v1992, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v1996 = __riscv_vreinterpret_v_u8mf2_i8mf2(v1995);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v1997 = v19 + 214;
      const int8_t* v1998 = (const int8_t*) v1997;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v1999 = *(const int8_t *)(v1998);
      const uint8_t* v2000 = v19 + 246;
      const int8_t* v2001 = (const int8_t*) v2000;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2002 = *(const int8_t *)(v2001);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2003 = __riscv_vwmacc_vx_i16m1(v1988, v1999, v1994, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2004 = __riscv_vwmacc_vx_i16m1(v1989, v2002, v1996, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2005 = v17 + 2096;
      const uint8_t* v2006 = (const uint8_t*) v2005;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2007 = __riscv_vle8_v_u8mf2(v2006, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2008 = __riscv_vand_vx_u8mf2(v2007, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2009 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2008);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2010 = __riscv_vsrl_vx_u8mf2(v2007, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2011 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2010);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2012 = v19 + 215;
      const int8_t* v2013 = (const int8_t*) v2012;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2014 = *(const int8_t *)(v2013);
      const uint8_t* v2015 = v19 + 247;
      const int8_t* v2016 = (const int8_t*) v2015;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2017 = *(const int8_t *)(v2016);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2018 = __riscv_vwmacc_vx_i16m1(v2003, v2014, v2009, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2019 = __riscv_vwmacc_vx_i16m1(v2004, v2017, v2011, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2020 = v17 + 2112;
      const uint8_t* v2021 = (const uint8_t*) v2020;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2022 = __riscv_vle8_v_u8mf2(v2021, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2023 = __riscv_vand_vx_u8mf2(v2022, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2024 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2023);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2025 = __riscv_vsrl_vx_u8mf2(v2022, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2026 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2025);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2027 = v19 + 216;
      const int8_t* v2028 = (const int8_t*) v2027;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2029 = *(const int8_t *)(v2028);
      const uint8_t* v2030 = v19 + 248;
      const int8_t* v2031 = (const int8_t*) v2030;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2032 = *(const int8_t *)(v2031);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2033 = __riscv_vwmacc_vx_i16m1(v2018, v2029, v2024, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2034 = __riscv_vwmacc_vx_i16m1(v2019, v2032, v2026, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2035 = v17 + 2128;
      const uint8_t* v2036 = (const uint8_t*) v2035;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2037 = __riscv_vle8_v_u8mf2(v2036, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2038 = __riscv_vand_vx_u8mf2(v2037, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2039 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2038);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2040 = __riscv_vsrl_vx_u8mf2(v2037, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2041 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2040);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2042 = v19 + 217;
      const int8_t* v2043 = (const int8_t*) v2042;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2044 = *(const int8_t *)(v2043);
      const uint8_t* v2045 = v19 + 249;
      const int8_t* v2046 = (const int8_t*) v2045;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2047 = *(const int8_t *)(v2046);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2048 = __riscv_vwmacc_vx_i16m1(v2033, v2044, v2039, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2049 = __riscv_vwmacc_vx_i16m1(v2034, v2047, v2041, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2050 = v17 + 2144;
      const uint8_t* v2051 = (const uint8_t*) v2050;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2052 = __riscv_vle8_v_u8mf2(v2051, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2053 = __riscv_vand_vx_u8mf2(v2052, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2054 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2053);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2055 = __riscv_vsrl_vx_u8mf2(v2052, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2056 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2055);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2057 = v19 + 218;
      const int8_t* v2058 = (const int8_t*) v2057;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2059 = *(const int8_t *)(v2058);
      const uint8_t* v2060 = v19 + 250;
      const int8_t* v2061 = (const int8_t*) v2060;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2062 = *(const int8_t *)(v2061);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2063 = __riscv_vwmacc_vx_i16m1(v2048, v2059, v2054, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2064 = __riscv_vwmacc_vx_i16m1(v2049, v2062, v2056, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2065 = v17 + 2160;
      const uint8_t* v2066 = (const uint8_t*) v2065;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2067 = __riscv_vle8_v_u8mf2(v2066, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2068 = __riscv_vand_vx_u8mf2(v2067, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2069 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2068);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2070 = __riscv_vsrl_vx_u8mf2(v2067, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2071 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2070);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2072 = v19 + 219;
      const int8_t* v2073 = (const int8_t*) v2072;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2074 = *(const int8_t *)(v2073);
      const uint8_t* v2075 = v19 + 251;
      const int8_t* v2076 = (const int8_t*) v2075;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2077 = *(const int8_t *)(v2076);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2078 = __riscv_vwmacc_vx_i16m1(v2063, v2074, v2069, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2079 = __riscv_vwmacc_vx_i16m1(v2064, v2077, v2071, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2080 = v17 + 2176;
      const uint8_t* v2081 = (const uint8_t*) v2080;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2082 = __riscv_vle8_v_u8mf2(v2081, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2083 = __riscv_vand_vx_u8mf2(v2082, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2084 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2083);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2085 = __riscv_vsrl_vx_u8mf2(v2082, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2086 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2085);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2087 = v19 + 220;
      const int8_t* v2088 = (const int8_t*) v2087;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2089 = *(const int8_t *)(v2088);
      const uint8_t* v2090 = v19 + 252;
      const int8_t* v2091 = (const int8_t*) v2090;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2092 = *(const int8_t *)(v2091);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2093 = __riscv_vwmacc_vx_i16m1(v2078, v2089, v2084, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2094 = __riscv_vwmacc_vx_i16m1(v2079, v2092, v2086, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2095 = v17 + 2192;
      const uint8_t* v2096 = (const uint8_t*) v2095;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2097 = __riscv_vle8_v_u8mf2(v2096, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2098 = __riscv_vand_vx_u8mf2(v2097, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2099 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2098);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2100 = __riscv_vsrl_vx_u8mf2(v2097, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2101 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2100);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2102 = v19 + 221;
      const int8_t* v2103 = (const int8_t*) v2102;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2104 = *(const int8_t *)(v2103);
      const uint8_t* v2105 = v19 + 253;
      const int8_t* v2106 = (const int8_t*) v2105;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2107 = *(const int8_t *)(v2106);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2108 = __riscv_vwmacc_vx_i16m1(v2093, v2104, v2099, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2109 = __riscv_vwmacc_vx_i16m1(v2094, v2107, v2101, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2110 = v17 + 2208;
      const uint8_t* v2111 = (const uint8_t*) v2110;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2112 = __riscv_vle8_v_u8mf2(v2111, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2113 = __riscv_vand_vx_u8mf2(v2112, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2114 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2113);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2115 = __riscv_vsrl_vx_u8mf2(v2112, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2116 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2115);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2117 = v19 + 222;
      const int8_t* v2118 = (const int8_t*) v2117;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2119 = *(const int8_t *)(v2118);
      const uint8_t* v2120 = v19 + 254;
      const int8_t* v2121 = (const int8_t*) v2120;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2122 = *(const int8_t *)(v2121);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2123 = __riscv_vwmacc_vx_i16m1(v2108, v2119, v2114, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2124 = __riscv_vwmacc_vx_i16m1(v2109, v2122, v2116, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2125 = v17 + 2224;
      const uint8_t* v2126 = (const uint8_t*) v2125;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2127 = __riscv_vle8_v_u8mf2(v2126, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2128 = __riscv_vand_vx_u8mf2(v2127, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2129 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2128);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2130 = __riscv_vsrl_vx_u8mf2(v2127, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2131 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2130);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2132 = v19 + 223;
      const int8_t* v2133 = (const int8_t*) v2132;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2134 = *(const int8_t *)(v2133);
      const uint8_t* v2135 = v19 + 255;
      const int8_t* v2136 = (const int8_t*) v2135;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2137 = *(const int8_t *)(v2136);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2138 = __riscv_vwmacc_vx_i16m1(v2123, v2134, v2129, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2139 = __riscv_vwmacc_vx_i16m1(v2124, v2137, v2131, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2140 = v17 + 2240;
      const uint8_t* v2141 = (const uint8_t*) v2140;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2142 = __riscv_vle8_v_u8mf2(v2141, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2143 = __riscv_vand_vx_u8mf2(v2142, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2144 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2143);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2145 = __riscv_vsrl_vx_u8mf2(v2142, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2146 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2145);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2147 = v19 + 224;
      const int8_t* v2148 = (const int8_t*) v2147;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2149 = *(const int8_t *)(v2148);
      const uint8_t* v2150 = v19 + 256;
      const int8_t* v2151 = (const int8_t*) v2150;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2152 = *(const int8_t *)(v2151);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2153 = __riscv_vwmacc_vx_i16m1(v2138, v2149, v2144, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2154 = __riscv_vwmacc_vx_i16m1(v2139, v2152, v2146, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2155 = v17 + 2256;
      const uint8_t* v2156 = (const uint8_t*) v2155;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2157 = __riscv_vle8_v_u8mf2(v2156, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2158 = __riscv_vand_vx_u8mf2(v2157, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2159 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2158);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2160 = __riscv_vsrl_vx_u8mf2(v2157, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2161 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2160);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2162 = v19 + 225;
      const int8_t* v2163 = (const int8_t*) v2162;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2164 = *(const int8_t *)(v2163);
      const uint8_t* v2165 = v19 + 257;
      const int8_t* v2166 = (const int8_t*) v2165;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2167 = *(const int8_t *)(v2166);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2168 = __riscv_vwmacc_vx_i16m1(v2153, v2164, v2159, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2169 = __riscv_vwmacc_vx_i16m1(v2154, v2167, v2161, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2170 = v17 + 2272;
      const uint8_t* v2171 = (const uint8_t*) v2170;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2172 = __riscv_vle8_v_u8mf2(v2171, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2173 = __riscv_vand_vx_u8mf2(v2172, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2174 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2173);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2175 = __riscv_vsrl_vx_u8mf2(v2172, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2176 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2175);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2177 = v19 + 226;
      const int8_t* v2178 = (const int8_t*) v2177;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2179 = *(const int8_t *)(v2178);
      const uint8_t* v2180 = v19 + 258;
      const int8_t* v2181 = (const int8_t*) v2180;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2182 = *(const int8_t *)(v2181);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2183 = __riscv_vwmacc_vx_i16m1(v2168, v2179, v2174, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2184 = __riscv_vwmacc_vx_i16m1(v2169, v2182, v2176, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
      const uint8_t* v2185 = v17 + 2288;
      const uint8_t* v2186 = (const uint8_t*) v2185;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
      vuint8mf2_t v2187 = __riscv_vle8_v_u8mf2(v2186, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
      vuint8mf2_t v2188 = __riscv_vand_vx_u8mf2(v2187, 0x0F, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2189 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2188);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
      vuint8mf2_t v2190 = __riscv_vsrl_vx_u8mf2(v2187, 4, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
      vint8mf2_t v2191 = __riscv_vreinterpret_v_u8mf2_i8mf2(v2190);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
      const uint8_t* v2192 = v19 + 227;
      const int8_t* v2193 = (const int8_t*) v2192;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2194 = *(const int8_t *)(v2193);
      const uint8_t* v2195 = v19 + 259;
      const int8_t* v2196 = (const int8_t*) v2195;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
      int32_t v2197 = *(const int8_t *)(v2196);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2198 = __riscv_vwmacc_vx_i16m1(v2183, v2194, v2189, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
      vint16m1_t v2199 = __riscv_vwmacc_vx_i16m1(v2184, v2197, v2191, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_subblock_fold
      vint32m2_t v2200 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2201 = __riscv_vwmacc_vv_i32m2(v2200, v1167, v2198, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i32m2
      vint32m2_t v2202 = __riscv_vwmacc_vv_i32m2(v2201, v1184, v2199, 16);
      v27 = v2202;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v2203 = (const _Float16*) v17;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v2204 = __riscv_vle16_v_f16m1(v2203, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwcvt_f_f_v_f32m2
      vfloat32m2_t v2205 = __riscv_vfwcvt_f_f_v_f32m2(v2204, 16);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
      vfloat32m2_t v2206 = __riscv_vfmul_vf_f32m2(v2205, v21, 16);
      vint32m2_t v2207 = v27;
      vfloat32m2_t v2208 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v2209 = __riscv_vfcvt_f_x_v_f32m2(v2207, 16);
      vfloat32m2_t v2210 = __riscv_vfmacc_vv_f32m2(v2208, v2209, v2206, 16);
      vint32m2_t v2211 = v29;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vv_f32m2
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v2212 = __riscv_vfcvt_f_x_v_f32m2(v2211, 16);
      vfloat32m2_t v2213 = __riscv_vfnmsac_vv_f32m2(v2210, v26, v2212, 16);
      v13 = v2213;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v2214 = v9 * 16;
    float* v2215 = v2 + v2214;
    vfloat32m2_t v2216 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_K_q8_K role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v2215, v2216, 16);
  }
  vint32m1_t v2217 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


