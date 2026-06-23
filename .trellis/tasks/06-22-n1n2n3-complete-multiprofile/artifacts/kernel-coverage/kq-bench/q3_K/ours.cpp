#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q3_K_q8_K_kernel_ggml_vec_dot_q3_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  uint32_t v9[4];
  // tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10[8];
  // tcrv_emitc.local_variable=sums source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat32m2_t v11;
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v12 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v11 = v12;
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v13 = 0; v13 < v6; v13 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v14 = v13 * 110;
    const uint8_t* v15 = v3 + v14;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v16 = v13 * 292;
    const uint8_t* v17 = v4 + v16;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_2bit_subtractive_hmask
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v19 = v15 + 32;
    const uint8_t* v20 = (const uint8_t*) v19;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v21 = __riscv_vle8_v_u8m2(v20, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=hmask_high_bit_plane
    const uint8_t* v22 = (const uint8_t*) v15;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v23 = __riscv_vle8_v_u8m2(v22, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v24 = __riscv_vand_vx_u8m2(v21, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v25 = __riscv_vand_vx_u8m2(v23, 0x01, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v26 = __riscv_vsll_vx_u8m2(v25, 0x02, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v27 = __riscv_vor_vv_u8m2(v24, v26, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v28 = __riscv_vreinterpret_v_u8m2_i8m2(v27);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v29 = __riscv_vsub_vx_i8m2(v28, 4, v18);
    int8_t* v30 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v30, v29, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v31 = __riscv_vsrl_vx_u8m2(v21, 2, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v32 = __riscv_vand_vx_u8m2(v31, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v33 = __riscv_vsrl_vx_u8m2(v23, 1, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v34 = __riscv_vand_vx_u8m2(v33, 0x01, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v35 = __riscv_vsll_vx_u8m2(v34, 0x02, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v36 = __riscv_vor_vv_u8m2(v32, v35, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v37 = __riscv_vreinterpret_v_u8m2_i8m2(v36);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v38 = __riscv_vsub_vx_i8m2(v37, 4, v18);
    int8_t* v39 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v39, v38, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v40 = __riscv_vsrl_vx_u8m2(v21, 4, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v41 = __riscv_vand_vx_u8m2(v40, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v42 = __riscv_vsrl_vx_u8m2(v23, 2, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v43 = __riscv_vand_vx_u8m2(v42, 0x01, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v44 = __riscv_vsll_vx_u8m2(v43, 0x02, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v45 = __riscv_vor_vv_u8m2(v41, v44, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v46 = __riscv_vreinterpret_v_u8m2_i8m2(v45);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v47 = __riscv_vsub_vx_i8m2(v46, 4, v18);
    int8_t* v48 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v48, v47, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v49 = __riscv_vsrl_vx_u8m2(v21, 6, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v50 = __riscv_vand_vx_u8m2(v49, 0x03, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v51 = __riscv_vsrl_vx_u8m2(v23, 3, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v52 = __riscv_vand_vx_u8m2(v51, 0x01, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v53 = __riscv_vsll_vx_u8m2(v52, 0x02, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v54 = __riscv_vor_vv_u8m2(v50, v53, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v55 = __riscv_vreinterpret_v_u8m2_i8m2(v54);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v56 = __riscv_vsub_vx_i8m2(v55, 4, v18);
    int8_t* v57 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v57, v56, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v58 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v59 = v15 + 64;
    const uint8_t* v60 = (const uint8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v61 = __riscv_vle8_v_u8m2(v60, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=hmask_high_bit_plane
    const uint8_t* v62 = (const uint8_t*) v15;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v63 = __riscv_vle8_v_u8m2(v62, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v64 = __riscv_vand_vx_u8m2(v61, 0x03, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v65 = __riscv_vsrl_vx_u8m2(v63, 4, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v66 = __riscv_vand_vx_u8m2(v65, 0x01, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v67 = __riscv_vsll_vx_u8m2(v66, 0x02, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v68 = __riscv_vor_vv_u8m2(v64, v67, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v69 = __riscv_vreinterpret_v_u8m2_i8m2(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v70 = __riscv_vsub_vx_i8m2(v69, 4, v58);
    int8_t* v71 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v71, v70, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v72 = __riscv_vsrl_vx_u8m2(v61, 2, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v73 = __riscv_vand_vx_u8m2(v72, 0x03, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v74 = __riscv_vsrl_vx_u8m2(v63, 5, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v75 = __riscv_vand_vx_u8m2(v74, 0x01, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v76 = __riscv_vsll_vx_u8m2(v75, 0x02, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v77 = __riscv_vor_vv_u8m2(v73, v76, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v78 = __riscv_vreinterpret_v_u8m2_i8m2(v77);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v79 = __riscv_vsub_vx_i8m2(v78, 4, v58);
    int8_t* v80 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v80, v79, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v81 = __riscv_vsrl_vx_u8m2(v61, 4, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v82 = __riscv_vand_vx_u8m2(v81, 0x03, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v83 = __riscv_vsrl_vx_u8m2(v63, 6, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v84 = __riscv_vand_vx_u8m2(v83, 0x01, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v85 = __riscv_vsll_vx_u8m2(v84, 0x02, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v86 = __riscv_vor_vv_u8m2(v82, v85, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v87 = __riscv_vreinterpret_v_u8m2_i8m2(v86);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v88 = __riscv_vsub_vx_i8m2(v87, 4, v58);
    int8_t* v89 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v89, v88, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v90 = __riscv_vsrl_vx_u8m2(v61, 6, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v91 = __riscv_vand_vx_u8m2(v90, 0x03, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v92 = __riscv_vsrl_vx_u8m2(v63, 7, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v93 = __riscv_vand_vx_u8m2(v92, 0x01, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v94 = __riscv_vsll_vx_u8m2(v93, 0x02, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v95 = __riscv_vor_vv_u8m2(v91, v94, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v96 = __riscv_vreinterpret_v_u8m2_i8m2(v95);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v97 = __riscv_vsub_vx_i8m2(v96, 4, v58);
    int8_t* v98 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v98, v97, v58);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=signed_scale_bit_dance
    const uint8_t* v99 = v15 + 96;
    const uint32_t* v100 = (const uint32_t*) v99;
    const uint32_t v101 = v100[0];
    uint32_t v102 = (uint32_t) v101;
    const uint32_t v103 = v100[1];
    uint32_t v104 = (uint32_t) v103;
    const uint32_t v105 = v100[2];
    uint32_t v106 = (uint32_t) v105;
    uint32_t v107 = v106 >> 0;
    uint32_t v108 = v107 & 0x03030303;
    uint32_t v109 = v108 << 4;
    uint32_t v110 = v102 & 0x0f0f0f0f;
    uint32_t v111 = v110 | v109;
    uint32_t v112 = v106 >> 2;
    uint32_t v113 = v112 & 0x03030303;
    uint32_t v114 = v113 << 4;
    uint32_t v115 = v104 & 0x0f0f0f0f;
    uint32_t v116 = v115 | v114;
    uint32_t v117 = v106 >> 4;
    uint32_t v118 = v117 & 0x03030303;
    uint32_t v119 = v118 << 4;
    uint32_t v120 = v102 >> 4;
    uint32_t v121 = v120 & 0x0f0f0f0f;
    uint32_t v122 = v121 | v119;
    uint32_t v123 = v106 >> 6;
    uint32_t v124 = v123 & 0x03030303;
    uint32_t v125 = v124 << 4;
    uint32_t v126 = v104 >> 4;
    uint32_t v127 = v126 & 0x0f0f0f0f;
    uint32_t v128 = v127 | v125;
    v9[0] = v111;
    v9[1] = v116;
    v9[2] = v122;
    v9[3] = v128;
    uint32_t* v129 = &v9[0];
    const int8_t* v130 = (const int8_t*) v129;
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v131;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v132 = __riscv_vmv_v_x_i32m2(0, 8);
    v131 = v132;
    const uint8_t* v133 = v17 + 4;
    const int8_t* v134 = (const int8_t*) v133;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v135 = 0; v135 < 16; v135 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=signed_scale_load
      const int8_t v136 = v130[v135];
      int v137 = (int) v136;
      int v138 = v137 - 32;
      size_t v139 = v135 * 16;
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_half
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v140 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v141 = v134 + v139;
      const int8_t* v142 = v8 + v139;
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v143 = __riscv_vle8_v_i8mf2(v141, v140);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v144 = __riscv_vle8_v_i8mf2(v142, v140);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v145 = __riscv_vwmul_vv_i16m1(v143, v144, v140);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v146 = v131;
      vint32m2_t v147 = __riscv_vwmacc_vx_i32m2(v146, v138, v145, v140);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v131 = v147;
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_half
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v148 = __riscv_vsetvl_e8mf2(8);
      size_t v149 = v139 + 8;
      const int8_t* v150 = v134 + v149;
      const int8_t* v151 = v8 + v149;
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v152 = __riscv_vle8_v_i8mf2(v150, v148);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v153 = __riscv_vle8_v_i8mf2(v151, v148);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v154 = __riscv_vwmul_vv_i16m1(v152, v153, v148);
      // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v155 = v131;
      vint32m2_t v156 = __riscv_vwmacc_vx_i32m2(v155, v138, v154, v148);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v131 = v156;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    const uint8_t* v157 = v15 + 108;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v158 = (float)*(const _Float16 *)(v157);
    const float* v159 = (const float*) v17;
    const float v160 = v159[0];
    float v161 = v158 * v160;
    vint32m2_t v162 = v131;
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v163 = __riscv_vfcvt_f_x_v_f32m2(v162, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v164 = __riscv_vfmul_vf_f32m2(v163, v161, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v165 = v11;
    vfloat32m2_t v166 = __riscv_vfadd_vv_f32m2(v165, v164, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v166;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v167 = &v10[0];
  vfloat32m2_t v168 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v167, v168, 8);
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v169 = v10[0];
  float v170 = 0.0f + v169;
  float v171 = v10[1];
  float v172 = v170 + v171;
  float v173 = v10[2];
  float v174 = v172 + v173;
  float v175 = v10[3];
  float v176 = v174 + v175;
  float v177 = v10[4];
  float v178 = v176 + v177;
  float v179 = v10[5];
  float v180 = v178 + v179;
  float v181 = v10[6];
  float v182 = v180 + v181;
  float v183 = v10[7];
  float v184 = v182 + v183;
  // tcrv_emitc.source_op=tcrv_rvv.q3_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v184;
  return;
}


