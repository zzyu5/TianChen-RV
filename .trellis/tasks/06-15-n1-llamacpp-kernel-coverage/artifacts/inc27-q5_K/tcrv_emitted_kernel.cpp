#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_ggml_vec_dot_q5_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  uint32_t v9[4];
  // tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10[8];
  // tcrv_emitc.local_variable=sums source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat32m2_t v11;
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v12 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v11 = v12;
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v13;
  v13 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v14 = 0; v14 < v6; v14 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v15 = v14 * 176;
    const uint8_t* v16 = v3 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v17 = v14 * 292;
    const uint8_t* v18 = v4 + v17;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_4bit
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v19 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v20 = v16 + 48;
    const uint8_t* v21 = (const uint8_t*) v20;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v22 = __riscv_vle8_v_u8m2(v21, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v23 = v16 + 16;
    const uint8_t* v24 = (const uint8_t*) v23;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v25 = __riscv_vle8_v_u8m2(v24, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v26 = __riscv_vand_vx_u8m2(v22, 0x0F, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v27 = __riscv_vand_vx_u8m2(v25, 0x01, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v28 = __riscv_vsll_vx_u8m2(v27, 0x04, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v29 = __riscv_vadd_vv_u8m2(v26, v28, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v30 = __riscv_vreinterpret_v_u8m2_i8m2(v29);
    int8_t* v31 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v31, v30, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v32 = __riscv_vsrl_vx_u8m2(v22, 0x04, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v33 = __riscv_vsrl_vx_u8m2(v25, 1, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v34 = __riscv_vand_vx_u8m2(v33, 0x01, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v35 = __riscv_vsll_vx_u8m2(v34, 0x04, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v36 = __riscv_vadd_vv_u8m2(v32, v35, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v37 = __riscv_vreinterpret_v_u8m2_i8m2(v36);
    int8_t* v38 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v38, v37, v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v39 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v40 = v16 + 80;
    const uint8_t* v41 = (const uint8_t*) v40;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v42 = __riscv_vle8_v_u8m2(v41, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v43 = v16 + 16;
    const uint8_t* v44 = (const uint8_t*) v43;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v45 = __riscv_vle8_v_u8m2(v44, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v46 = __riscv_vand_vx_u8m2(v42, 0x0F, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v47 = __riscv_vsrl_vx_u8m2(v45, 2, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v48 = __riscv_vand_vx_u8m2(v47, 0x01, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v49 = __riscv_vsll_vx_u8m2(v48, 0x04, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v50 = __riscv_vadd_vv_u8m2(v46, v49, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v51 = __riscv_vreinterpret_v_u8m2_i8m2(v50);
    int8_t* v52 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v52, v51, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v53 = __riscv_vsrl_vx_u8m2(v42, 0x04, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v54 = __riscv_vsrl_vx_u8m2(v45, 3, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v55 = __riscv_vand_vx_u8m2(v54, 0x01, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v56 = __riscv_vsll_vx_u8m2(v55, 0x04, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v57 = __riscv_vadd_vv_u8m2(v53, v56, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v58 = __riscv_vreinterpret_v_u8m2_i8m2(v57);
    int8_t* v59 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v59, v58, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v60 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v61 = v16 + 112;
    const uint8_t* v62 = (const uint8_t*) v61;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v63 = __riscv_vle8_v_u8m2(v62, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v64 = v16 + 16;
    const uint8_t* v65 = (const uint8_t*) v64;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v66 = __riscv_vle8_v_u8m2(v65, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v67 = __riscv_vand_vx_u8m2(v63, 0x0F, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v68 = __riscv_vsrl_vx_u8m2(v66, 4, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v69 = __riscv_vand_vx_u8m2(v68, 0x01, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v70 = __riscv_vsll_vx_u8m2(v69, 0x04, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v71 = __riscv_vadd_vv_u8m2(v67, v70, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v72 = __riscv_vreinterpret_v_u8m2_i8m2(v71);
    int8_t* v73 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v73, v72, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v74 = __riscv_vsrl_vx_u8m2(v63, 0x04, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v75 = __riscv_vsrl_vx_u8m2(v66, 5, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v76 = __riscv_vand_vx_u8m2(v75, 0x01, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v77 = __riscv_vsll_vx_u8m2(v76, 0x04, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v78 = __riscv_vadd_vv_u8m2(v74, v77, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v79 = __riscv_vreinterpret_v_u8m2_i8m2(v78);
    int8_t* v80 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v80, v79, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v81 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v82 = v16 + 144;
    const uint8_t* v83 = (const uint8_t*) v82;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v84 = __riscv_vle8_v_u8m2(v83, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v85 = v16 + 16;
    const uint8_t* v86 = (const uint8_t*) v85;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v87 = __riscv_vle8_v_u8m2(v86, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v88 = __riscv_vand_vx_u8m2(v84, 0x0F, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v89 = __riscv_vsrl_vx_u8m2(v87, 6, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v90 = __riscv_vand_vx_u8m2(v89, 0x01, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v91 = __riscv_vsll_vx_u8m2(v90, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v92 = __riscv_vadd_vv_u8m2(v88, v91, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v93 = __riscv_vreinterpret_v_u8m2_i8m2(v92);
    int8_t* v94 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v94, v93, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v95 = __riscv_vsrl_vx_u8m2(v84, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v96 = __riscv_vsrl_vx_u8m2(v87, 7, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v97 = __riscv_vand_vx_u8m2(v96, 0x01, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v98 = __riscv_vsll_vx_u8m2(v97, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_u8m2
    vuint8m2_t v99 = __riscv_vadd_vv_u8m2(v95, v98, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v100 = __riscv_vreinterpret_v_u8m2_i8m2(v99);
    int8_t* v101 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v101, v100, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_bit_dance
    const uint8_t* v102 = v16 + 4;
    const uint32_t* v103 = (const uint32_t*) v102;
    const uint32_t v104 = v103[0];
    uint32_t v105 = (uint32_t) v104;
    const uint32_t v106 = v103[1];
    uint32_t v107 = (uint32_t) v106;
    const uint32_t v108 = v103[2];
    uint32_t v109 = (uint32_t) v108;
    uint32_t v110 = v107 >> 6;
    uint32_t v111 = v110 & 0x03030303;
    uint32_t v112 = v111 << 4;
    uint32_t v113 = v109 >> 4;
    uint32_t v114 = v113 & 0x0f0f0f0f;
    uint32_t v115 = v114 | v112;
    uint32_t v116 = v107 & 0x3f3f3f3f;
    uint32_t v117 = v105 >> 6;
    uint32_t v118 = v117 & 0x03030303;
    uint32_t v119 = v118 << 4;
    uint32_t v120 = v109 & 0x0f0f0f0f;
    uint32_t v121 = v120 | v119;
    uint32_t v122 = v105 & 0x3f3f3f3f;
    v9[0] = v122;
    v9[1] = v121;
    v9[2] = v116;
    v9[3] = v115;
    uint32_t* v123 = &v9[0];
    const uint8_t* v124 = (const uint8_t*) v123;
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v125;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v126 = __riscv_vmv_v_x_i32m2(0, 8);
    v125 = v126;
    const uint8_t* v127 = v18 + 4;
    const int8_t* v128 = (const int8_t*) v127;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v129 = 0; v129 < 8; v129 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const uint8_t v130 = v124[v129];
      int v131 = (int) v130;
      size_t v132 = v129 * 32;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v133 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v134 = v128 + v132;
      const int8_t* v135 = v8 + v132;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v136 = __riscv_vle8_v_i8mf2(v134, v133);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v137 = __riscv_vle8_v_i8mf2(v135, v133);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v138 = __riscv_vwmul_vv_i16m1(v136, v137, v133);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v139 = v125;
      vint32m2_t v140 = __riscv_vwmacc_vx_i32m2(v139, v131, v138, v133);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v125 = v140;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v141 = __riscv_vsetvl_e8mf2(8);
      size_t v142 = v132 + 8;
      const int8_t* v143 = v128 + v142;
      const int8_t* v144 = v8 + v142;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v145 = __riscv_vle8_v_i8mf2(v143, v141);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v146 = __riscv_vle8_v_i8mf2(v144, v141);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v147 = __riscv_vwmul_vv_i16m1(v145, v146, v141);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v148 = v125;
      vint32m2_t v149 = __riscv_vwmacc_vx_i32m2(v148, v131, v147, v141);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v125 = v149;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v150 = __riscv_vsetvl_e8mf2(8);
      size_t v151 = v132 + 16;
      const int8_t* v152 = v128 + v151;
      const int8_t* v153 = v8 + v151;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v154 = __riscv_vle8_v_i8mf2(v152, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v155 = __riscv_vle8_v_i8mf2(v153, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v156 = __riscv_vwmul_vv_i16m1(v154, v155, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v157 = v125;
      vint32m2_t v158 = __riscv_vwmacc_vx_i32m2(v157, v131, v156, v150);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v125 = v158;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v159 = __riscv_vsetvl_e8mf2(8);
      size_t v160 = v132 + 24;
      const int8_t* v161 = v128 + v160;
      const int8_t* v162 = v8 + v160;
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v163 = __riscv_vle8_v_i8mf2(v161, v159);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v164 = __riscv_vle8_v_i8mf2(v162, v159);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v165 = __riscv_vwmul_vv_i16m1(v163, v164, v159);
      // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v166 = v125;
      vint32m2_t v167 = __riscv_vwmacc_vx_i32m2(v166, v131, v165, v159);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v125 = v167;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v168 = (const float*) v18;
    const float v169 = v168[0];
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v170 = v18 + 260;
    const int16_t* v171 = (const int16_t*) v170;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v172;
    v172 = 0;
    const int16_t v173 = v171[0];
    int v174 = (int) v173;
    const uint8_t v175 = v124[8];
    int v176 = (int) v175;
    int v177 = v174 * v176;
    int v178 = v172;
    int v179 = v178 + v177;
    v172 = v179;
    const int16_t v180 = v171[1];
    int v181 = (int) v180;
    const uint8_t v182 = v124[8];
    int v183 = (int) v182;
    int v184 = v181 * v183;
    int v185 = v172;
    int v186 = v185 + v184;
    v172 = v186;
    const int16_t v187 = v171[2];
    int v188 = (int) v187;
    const uint8_t v189 = v124[9];
    int v190 = (int) v189;
    int v191 = v188 * v190;
    int v192 = v172;
    int v193 = v192 + v191;
    v172 = v193;
    const int16_t v194 = v171[3];
    int v195 = (int) v194;
    const uint8_t v196 = v124[9];
    int v197 = (int) v196;
    int v198 = v195 * v197;
    int v199 = v172;
    int v200 = v199 + v198;
    v172 = v200;
    const int16_t v201 = v171[4];
    int v202 = (int) v201;
    const uint8_t v203 = v124[10];
    int v204 = (int) v203;
    int v205 = v202 * v204;
    int v206 = v172;
    int v207 = v206 + v205;
    v172 = v207;
    const int16_t v208 = v171[5];
    int v209 = (int) v208;
    const uint8_t v210 = v124[10];
    int v211 = (int) v210;
    int v212 = v209 * v211;
    int v213 = v172;
    int v214 = v213 + v212;
    v172 = v214;
    const int16_t v215 = v171[6];
    int v216 = (int) v215;
    const uint8_t v217 = v124[11];
    int v218 = (int) v217;
    int v219 = v216 * v218;
    int v220 = v172;
    int v221 = v220 + v219;
    v172 = v221;
    const int16_t v222 = v171[7];
    int v223 = (int) v222;
    const uint8_t v224 = v124[11];
    int v225 = (int) v224;
    int v226 = v223 * v225;
    int v227 = v172;
    int v228 = v227 + v226;
    v172 = v228;
    const int16_t v229 = v171[8];
    int v230 = (int) v229;
    const uint8_t v231 = v124[12];
    int v232 = (int) v231;
    int v233 = v230 * v232;
    int v234 = v172;
    int v235 = v234 + v233;
    v172 = v235;
    const int16_t v236 = v171[9];
    int v237 = (int) v236;
    const uint8_t v238 = v124[12];
    int v239 = (int) v238;
    int v240 = v237 * v239;
    int v241 = v172;
    int v242 = v241 + v240;
    v172 = v242;
    const int16_t v243 = v171[10];
    int v244 = (int) v243;
    const uint8_t v245 = v124[13];
    int v246 = (int) v245;
    int v247 = v244 * v246;
    int v248 = v172;
    int v249 = v248 + v247;
    v172 = v249;
    const int16_t v250 = v171[11];
    int v251 = (int) v250;
    const uint8_t v252 = v124[13];
    int v253 = (int) v252;
    int v254 = v251 * v253;
    int v255 = v172;
    int v256 = v255 + v254;
    v172 = v256;
    const int16_t v257 = v171[12];
    int v258 = (int) v257;
    const uint8_t v259 = v124[14];
    int v260 = (int) v259;
    int v261 = v258 * v260;
    int v262 = v172;
    int v263 = v262 + v261;
    v172 = v263;
    const int16_t v264 = v171[13];
    int v265 = (int) v264;
    const uint8_t v266 = v124[14];
    int v267 = (int) v266;
    int v268 = v265 * v267;
    int v269 = v172;
    int v270 = v269 + v268;
    v172 = v270;
    const int16_t v271 = v171[14];
    int v272 = (int) v271;
    const uint8_t v273 = v124[15];
    int v274 = (int) v273;
    int v275 = v272 * v274;
    int v276 = v172;
    int v277 = v276 + v275;
    v172 = v277;
    const int16_t v278 = v171[15];
    int v279 = (int) v278;
    const uint8_t v280 = v124[15];
    int v281 = (int) v280;
    int v282 = v279 * v281;
    int v283 = v172;
    int v284 = v283 + v282;
    v172 = v284;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v285 = (float)*(const _Float16 *)(v16);
    float v286 = v285 * v169;
    vint32m2_t v287 = v125;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v288 = __riscv_vfcvt_f_x_v_f32m2(v287, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v289 = __riscv_vfmul_vf_f32m2(v288, v286, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v290 = v11;
    vfloat32m2_t v291 = __riscv_vfadd_vv_f32m2(v290, v289, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v291;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v292 = v16 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v293 = (float)*(const _Float16 *)(v292);
    float v294 = v293 * v169;
    // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=min_subtract
    int v295 = v172;
    float v296 = v13;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v296 - v294 * (float) v295;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v297 = &v10[0];
  vfloat32m2_t v298 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v297, v298, 8);
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v299 = v13;
  float v300 = v10[0];
  float v301 = v299 + v300;
  float v302 = v10[1];
  float v303 = v301 + v302;
  float v304 = v10[2];
  float v305 = v303 + v304;
  float v306 = v10[3];
  float v307 = v305 + v306;
  float v308 = v10[4];
  float v309 = v307 + v308;
  float v310 = v10[5];
  float v311 = v309 + v310;
  float v312 = v10[6];
  float v313 = v311 + v312;
  float v314 = v10[7];
  float v315 = v313 + v314;
  // tcrv_emitc.source_op=tcrv_rvv.q5_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v315;
  return;
}


