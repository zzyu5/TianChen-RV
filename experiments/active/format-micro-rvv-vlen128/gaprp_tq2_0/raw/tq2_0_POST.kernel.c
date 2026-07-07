#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v7;
  v7 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v8 = 0; v8 < v6; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v9 = v8 * 66;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v11 = v8 * 292;
    const uint8_t* v12 = v4 + v11;
    const uint8_t* v13 = v12 + 4;
    const int8_t* v14 = (const int8_t*) v13;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=chunk_dot
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16m4
    size_t v16 = __riscv_vsetvl_e16m4(32);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m4
    vint16m4_t v17 = __riscv_vmv_v_x_i16m4(0, v16);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v19 = (const uint8_t*) v10;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v20 = __riscv_vle8_v_u8m2(v19, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v21 = __riscv_vand_vx_u8m2(v20, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v22 = __riscv_vreinterpret_v_u8m2_i8m2(v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v23 = __riscv_vsub_vx_i8m2(v22, 1, v18);
    const int8_t* v24 = v14 + 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v25 = __riscv_vle8_v_i8m2(v24, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v26 = __riscv_vwmacc_vv_i16m4(v17, v23, v25, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v27 = __riscv_vsrl_vx_u8m2(v20, 2, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v28 = __riscv_vand_vx_u8m2(v27, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v29 = __riscv_vreinterpret_v_u8m2_i8m2(v28);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v30 = __riscv_vsub_vx_i8m2(v29, 1, v18);
    const int8_t* v31 = v14 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v32 = __riscv_vle8_v_i8m2(v31, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v33 = __riscv_vwmacc_vv_i16m4(v26, v30, v32, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v34 = __riscv_vsrl_vx_u8m2(v20, 4, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v35 = __riscv_vand_vx_u8m2(v34, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v36 = __riscv_vreinterpret_v_u8m2_i8m2(v35);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v37 = __riscv_vsub_vx_i8m2(v36, 1, v18);
    const int8_t* v38 = v14 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v39 = __riscv_vle8_v_i8m2(v38, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v40 = __riscv_vwmacc_vv_i16m4(v33, v37, v39, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v41 = __riscv_vsrl_vx_u8m2(v20, 6, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v42 = __riscv_vand_vx_u8m2(v41, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v43 = __riscv_vreinterpret_v_u8m2_i8m2(v42);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v44 = __riscv_vsub_vx_i8m2(v43, 1, v18);
    const int8_t* v45 = v14 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v46 = __riscv_vle8_v_i8m2(v45, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v47 = __riscv_vwmacc_vv_i16m4(v40, v44, v46, v18);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v48 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v49 = v10 + 32;
    const uint8_t* v50 = (const uint8_t*) v49;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v51 = __riscv_vle8_v_u8m2(v50, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v52 = __riscv_vand_vx_u8m2(v51, 0x03, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v53 = __riscv_vreinterpret_v_u8m2_i8m2(v52);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v54 = __riscv_vsub_vx_i8m2(v53, 1, v48);
    const int8_t* v55 = v14 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v56 = __riscv_vle8_v_i8m2(v55, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v57 = __riscv_vwmacc_vv_i16m4(v47, v54, v56, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v58 = __riscv_vsrl_vx_u8m2(v51, 2, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v59 = __riscv_vand_vx_u8m2(v58, 0x03, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v60 = __riscv_vreinterpret_v_u8m2_i8m2(v59);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v61 = __riscv_vsub_vx_i8m2(v60, 1, v48);
    const int8_t* v62 = v14 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v63 = __riscv_vle8_v_i8m2(v62, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v64 = __riscv_vwmacc_vv_i16m4(v57, v61, v63, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v65 = __riscv_vsrl_vx_u8m2(v51, 4, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v66 = __riscv_vand_vx_u8m2(v65, 0x03, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v67 = __riscv_vreinterpret_v_u8m2_i8m2(v66);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v68 = __riscv_vsub_vx_i8m2(v67, 1, v48);
    const int8_t* v69 = v14 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v70 = __riscv_vle8_v_i8m2(v69, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v71 = __riscv_vwmacc_vv_i16m4(v64, v68, v70, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v72 = __riscv_vsrl_vx_u8m2(v51, 6, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v73 = __riscv_vand_vx_u8m2(v72, 0x03, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v74 = __riscv_vreinterpret_v_u8m2_i8m2(v73);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v75 = __riscv_vsub_vx_i8m2(v74, 1, v48);
    const int8_t* v76 = v14 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v77 = __riscv_vle8_v_i8m2(v76, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m4
    vint16m4_t v78 = __riscv_vwmacc_vv_i16m4(v71, v75, v77, v48);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16m4
    size_t v79 = __riscv_vsetvl_e16m4(32);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v80 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v81 = __riscv_vwredsum_vs_i16m4_i32m1(v78, v80, v79);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v82 = __riscv_vmv_x_s_i32m1_i32(v81);
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v83 = v15;
    int v84 = v83 + v82;
    v15 = v84;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v85 = (const float*) v12;
    const float v86 = v85[0];
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    const uint8_t* v87 = v10 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v88 = (float)*(const _Float16 *)(v87);
    float v89 = v86 * v88;
    // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold
    int v90 = v15;
    float v91 = v7;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v7 = v91 + (float) v90 * v89;
  }
  float v92 = v7;
  // tcrv_emitc.source_op=tcrv_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v92;
  return;
}


