#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=sums8 source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v9[8];
  // tcrv_emitc.local_variable=sums source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat32m2_t v10;
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v11 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v10 = v11;
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v12 = 0; v12 < v6; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v13 = v12 * 210;
    const uint8_t* v14 = v3 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v15 = v12 * 292;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_6bit
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v17 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v18 = (const uint8_t*) v14;
    const uint8_t* v19 = v14 + 32;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t* v21 = v14 + 128;
    const uint8_t* v22 = (const uint8_t*) v21;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v23 = __riscv_vle8_v_u8m2(v18, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v24 = __riscv_vle8_v_u8m2(v20, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v25 = __riscv_vle8_v_u8m2(v22, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v26 = __riscv_vand_vx_u8m2(v23, 0x0F, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v27 = __riscv_vand_vx_u8m2(v25, 0x03, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v28 = __riscv_vsll_vx_u8m2(v27, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v29 = __riscv_vor_vv_u8m2(v26, v28, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v30 = __riscv_vreinterpret_v_u8m2_i8m2(v29);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v31 = __riscv_vsub_vx_i8m2(v30, 32, v17);
    int8_t* v32 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v32, v31, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v33 = __riscv_vand_vx_u8m2(v24, 0x0F, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v34 = __riscv_vsrl_vx_u8m2(v25, 0x02, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v35 = __riscv_vand_vx_u8m2(v34, 0x03, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v36 = __riscv_vsll_vx_u8m2(v35, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v37 = __riscv_vor_vv_u8m2(v33, v36, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v38 = __riscv_vreinterpret_v_u8m2_i8m2(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v39 = __riscv_vsub_vx_i8m2(v38, 32, v17);
    int8_t* v40 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v40, v39, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v41 = __riscv_vsrl_vx_u8m2(v23, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v42 = __riscv_vsrl_vx_u8m2(v25, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v43 = __riscv_vand_vx_u8m2(v42, 0x03, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v44 = __riscv_vsll_vx_u8m2(v43, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v45 = __riscv_vor_vv_u8m2(v41, v44, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v46 = __riscv_vreinterpret_v_u8m2_i8m2(v45);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v47 = __riscv_vsub_vx_i8m2(v46, 32, v17);
    int8_t* v48 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v48, v47, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v49 = __riscv_vsrl_vx_u8m2(v24, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v50 = __riscv_vsrl_vx_u8m2(v25, 0x06, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v51 = __riscv_vand_vx_u8m2(v50, 0x03, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v52 = __riscv_vsll_vx_u8m2(v51, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v53 = __riscv_vor_vv_u8m2(v49, v52, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v54 = __riscv_vreinterpret_v_u8m2_i8m2(v53);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v55 = __riscv_vsub_vx_i8m2(v54, 32, v17);
    int8_t* v56 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v56, v55, v17);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v57 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v58 = v14 + 64;
    const uint8_t* v59 = (const uint8_t*) v58;
    const uint8_t* v60 = v14 + 96;
    const uint8_t* v61 = (const uint8_t*) v60;
    const uint8_t* v62 = v14 + 160;
    const uint8_t* v63 = (const uint8_t*) v62;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v64 = __riscv_vle8_v_u8m2(v59, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v65 = __riscv_vle8_v_u8m2(v61, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v66 = __riscv_vle8_v_u8m2(v63, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v67 = __riscv_vand_vx_u8m2(v64, 0x0F, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v68 = __riscv_vand_vx_u8m2(v66, 0x03, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v69 = __riscv_vsll_vx_u8m2(v68, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v70 = __riscv_vor_vv_u8m2(v67, v69, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v71 = __riscv_vreinterpret_v_u8m2_i8m2(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v72 = __riscv_vsub_vx_i8m2(v71, 32, v57);
    int8_t* v73 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v73, v72, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v74 = __riscv_vand_vx_u8m2(v65, 0x0F, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v75 = __riscv_vsrl_vx_u8m2(v66, 0x02, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v76 = __riscv_vand_vx_u8m2(v75, 0x03, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v77 = __riscv_vsll_vx_u8m2(v76, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v78 = __riscv_vor_vv_u8m2(v74, v77, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v79 = __riscv_vreinterpret_v_u8m2_i8m2(v78);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v80 = __riscv_vsub_vx_i8m2(v79, 32, v57);
    int8_t* v81 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v81, v80, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v82 = __riscv_vsrl_vx_u8m2(v64, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v83 = __riscv_vsrl_vx_u8m2(v66, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v84 = __riscv_vand_vx_u8m2(v83, 0x03, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v85 = __riscv_vsll_vx_u8m2(v84, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v86 = __riscv_vor_vv_u8m2(v82, v85, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v87 = __riscv_vreinterpret_v_u8m2_i8m2(v86);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v88 = __riscv_vsub_vx_i8m2(v87, 32, v57);
    int8_t* v89 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v89, v88, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v90 = __riscv_vsrl_vx_u8m2(v65, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v91 = __riscv_vsrl_vx_u8m2(v66, 0x06, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v92 = __riscv_vand_vx_u8m2(v91, 0x03, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v93 = __riscv_vsll_vx_u8m2(v92, 0x04, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v94 = __riscv_vor_vv_u8m2(v90, v93, v57);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v95 = __riscv_vreinterpret_v_u8m2_i8m2(v94);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v96 = __riscv_vsub_vx_i8m2(v95, 32, v57);
    int8_t* v97 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v97, v96, v57);
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m4_t v98;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
    vint32m4_t v99 = __riscv_vmv_v_x_i32m4(0, 16);
    v98 = v99;
    const uint8_t* v100 = v16 + 4;
    const int8_t* v101 = (const int8_t*) v100;
    const uint8_t* v102 = v14 + 192;
    const int8_t* v103 = (const int8_t*) v102;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v104 = 0; v104 < 16; v104 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const int8_t v105 = v103[v104];
      int v106 = (int) v105;
      size_t v107 = v104 * 16;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_half
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v108 = __riscv_vsetvl_e8m1(16);
      const int8_t* v109 = v101 + v107;
      const int8_t* v110 = v8 + v107;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v111 = __riscv_vle8_v_i8m1(v109, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v112 = __riscv_vle8_v_i8m1(v110, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v113 = __riscv_vwmul_vv_i16m2(v111, v112, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m4
      vint32m4_t v114 = v98;
      vint32m4_t v115 = __riscv_vwmacc_vx_i32m4(v114, v106, v113, v108);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v98 = v115;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=aux32_fold_back
    vint32m4_t v116 = v98;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i32m4
    vint32m4_t v117 = __riscv_vslidedown_vx_i32m4(v116, 8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m4
    vint32m4_t v118 = __riscv_vadd_vv_i32m4(v116, v117, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m4_i32m2
    vint32m2_t v119 = __riscv_vget_v_i32m4_i32m2(v118, 0);
    // tcrv_emitc.local_variable=aux32_fold source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v120;
    v120 = v119;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    const uint8_t* v121 = v14 + 208;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v122 = (float)*(const _Float16 *)(v121);
    const float* v123 = (const float*) v16;
    const float v124 = v123[0];
    float v125 = v122 * v124;
    vint32m2_t v126 = v120;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v127 = __riscv_vfcvt_f_x_v_f32m2(v126, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v128 = __riscv_vfmul_vf_f32m2(v127, v125, 8);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v129 = v10;
    vfloat32m2_t v130 = __riscv_vfadd_vv_f32m2(v129, v128, 8);
    // tcrv_emitc.assign target=sums source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v130;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_sums_lanes
  float* v131 = &v9[0];
  vfloat32m2_t v132 = v10;
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v131, v132, 8);
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=horizontal_sum
  float v133 = v9[0];
  float v134 = 0.0f + v133;
  float v135 = v9[1];
  float v136 = v134 + v135;
  float v137 = v9[2];
  float v138 = v136 + v137;
  float v139 = v9[3];
  float v140 = v138 + v139;
  float v141 = v9[4];
  float v142 = v140 + v141;
  float v143 = v9[5];
  float v144 = v142 + v143;
  float v145 = v9[6];
  float v146 = v144 + v145;
  float v147 = v9[7];
  float v148 = v146 + v147;
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v148;
  return;
}


