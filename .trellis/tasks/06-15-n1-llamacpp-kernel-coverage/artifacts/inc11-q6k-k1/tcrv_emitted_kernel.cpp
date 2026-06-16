#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q6_K_q8_K_aux32_kernel_ggml_vec_dot_q6_K_q8_K_aux32(size_t v1, int32_t* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v6; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 210;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_6bit
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v14 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v15 = (const uint8_t*) v11;
    const uint8_t* v16 = v11 + 32;
    const uint8_t* v17 = (const uint8_t*) v16;
    const uint8_t* v18 = v11 + 128;
    const uint8_t* v19 = (const uint8_t*) v18;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v20 = __riscv_vle8_v_u8m2(v15, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v21 = __riscv_vle8_v_u8m2(v17, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v22 = __riscv_vle8_v_u8m2(v19, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v23 = __riscv_vand_vx_u8m2(v20, 0x0F, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v24 = __riscv_vand_vx_u8m2(v22, 0x03, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v25 = __riscv_vsll_vx_u8m2(v24, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v26 = __riscv_vor_vv_u8m2(v23, v25, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v27 = __riscv_vreinterpret_v_u8m2_i8m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v28 = __riscv_vsub_vx_i8m2(v27, 32, v14);
    int8_t* v29 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v29, v28, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v30 = __riscv_vand_vx_u8m2(v21, 0x0F, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v31 = __riscv_vsrl_vx_u8m2(v22, 0x02, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v32 = __riscv_vand_vx_u8m2(v31, 0x03, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v33 = __riscv_vsll_vx_u8m2(v32, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v34 = __riscv_vor_vv_u8m2(v30, v33, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v35 = __riscv_vreinterpret_v_u8m2_i8m2(v34);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v36 = __riscv_vsub_vx_i8m2(v35, 32, v14);
    int8_t* v37 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v37, v36, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v38 = __riscv_vsrl_vx_u8m2(v20, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v39 = __riscv_vsrl_vx_u8m2(v22, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v40 = __riscv_vand_vx_u8m2(v39, 0x03, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v41 = __riscv_vsll_vx_u8m2(v40, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v42 = __riscv_vor_vv_u8m2(v38, v41, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v43 = __riscv_vreinterpret_v_u8m2_i8m2(v42);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v44 = __riscv_vsub_vx_i8m2(v43, 32, v14);
    int8_t* v45 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v45, v44, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v46 = __riscv_vsrl_vx_u8m2(v21, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v47 = __riscv_vsrl_vx_u8m2(v22, 0x06, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v48 = __riscv_vand_vx_u8m2(v47, 0x03, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v49 = __riscv_vsll_vx_u8m2(v48, 0x04, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v50 = __riscv_vor_vv_u8m2(v46, v49, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v51 = __riscv_vreinterpret_v_u8m2_i8m2(v50);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v52 = __riscv_vsub_vx_i8m2(v51, 32, v14);
    int8_t* v53 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v53, v52, v14);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v54 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v55 = v11 + 64;
    const uint8_t* v56 = (const uint8_t*) v55;
    const uint8_t* v57 = v11 + 96;
    const uint8_t* v58 = (const uint8_t*) v57;
    const uint8_t* v59 = v11 + 160;
    const uint8_t* v60 = (const uint8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v61 = __riscv_vle8_v_u8m2(v56, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v62 = __riscv_vle8_v_u8m2(v58, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v63 = __riscv_vle8_v_u8m2(v60, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v64 = __riscv_vand_vx_u8m2(v61, 0x0F, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v65 = __riscv_vand_vx_u8m2(v63, 0x03, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v66 = __riscv_vsll_vx_u8m2(v65, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v67 = __riscv_vor_vv_u8m2(v64, v66, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v68 = __riscv_vreinterpret_v_u8m2_i8m2(v67);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v69 = __riscv_vsub_vx_i8m2(v68, 32, v54);
    int8_t* v70 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v70, v69, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v71 = __riscv_vand_vx_u8m2(v62, 0x0F, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v72 = __riscv_vsrl_vx_u8m2(v63, 0x02, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v73 = __riscv_vand_vx_u8m2(v72, 0x03, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v74 = __riscv_vsll_vx_u8m2(v73, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v75 = __riscv_vor_vv_u8m2(v71, v74, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v76 = __riscv_vreinterpret_v_u8m2_i8m2(v75);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v77 = __riscv_vsub_vx_i8m2(v76, 32, v54);
    int8_t* v78 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v78, v77, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v79 = __riscv_vsrl_vx_u8m2(v61, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v80 = __riscv_vsrl_vx_u8m2(v63, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v81 = __riscv_vand_vx_u8m2(v80, 0x03, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v82 = __riscv_vsll_vx_u8m2(v81, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v83 = __riscv_vor_vv_u8m2(v79, v82, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v84 = __riscv_vreinterpret_v_u8m2_i8m2(v83);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v85 = __riscv_vsub_vx_i8m2(v84, 32, v54);
    int8_t* v86 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v86, v85, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v87 = __riscv_vsrl_vx_u8m2(v62, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v88 = __riscv_vsrl_vx_u8m2(v63, 0x06, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v89 = __riscv_vand_vx_u8m2(v88, 0x03, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u8m2
    vuint8m2_t v90 = __riscv_vsll_vx_u8m2(v89, 0x04, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m2
    vuint8m2_t v91 = __riscv_vor_vv_u8m2(v87, v90, v54);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v92 = __riscv_vreinterpret_v_u8m2_i8m2(v91);
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m2
    vint8m2_t v93 = __riscv_vsub_vx_i8m2(v92, 32, v54);
    int8_t* v94 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v94, v93, v54);
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v95;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v96 = __riscv_vmv_v_x_i32m2(0, 8);
    v95 = v96;
    const uint8_t* v97 = v13 + 4;
    const int8_t* v98 = (const int8_t*) v97;
    const uint8_t* v99 = v11 + 192;
    const int8_t* v100 = (const int8_t*) v99;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v101 = 0; v101 < 16; v101 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const int8_t v102 = v100[v101];
      int v103 = (int) v102;
      size_t v104 = v101 * 16;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_half
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v105 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v106 = v98 + v104;
      const int8_t* v107 = v8 + v104;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v108 = __riscv_vle8_v_i8mf2(v106, v105);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v109 = __riscv_vle8_v_i8mf2(v107, v105);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v110 = __riscv_vwmul_vv_i16m1(v108, v109, v105);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v111 = v95;
      vint32m2_t v112 = __riscv_vwmacc_vx_i32m2(v111, v103, v110, v105);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v95 = v112;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_half
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v113 = __riscv_vsetvl_e8mf2(8);
      size_t v114 = v104 + 8;
      const int8_t* v115 = v98 + v114;
      const int8_t* v116 = v8 + v114;
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v117 = __riscv_vle8_v_i8mf2(v115, v113);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v118 = __riscv_vle8_v_i8mf2(v116, v113);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v119 = __riscv_vwmul_vv_i16m1(v117, v118, v113);
      // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v120 = v95;
      vint32m2_t v121 = __riscv_vwmacc_vx_i32m2(v120, v103, v119, v113);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v95 = v121;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_aux32
    size_t v122 = v9 * 8;
    int32_t* v123 = v2 + v122;
    vint32m2_t v124 = v95;
    // tcrv_emitc.source_op=tcrv_rvv.q6_k_q8_k_aux32_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
    __riscv_vse32_v_i32m2(v123, v124, 8);
  }
  return;
}


