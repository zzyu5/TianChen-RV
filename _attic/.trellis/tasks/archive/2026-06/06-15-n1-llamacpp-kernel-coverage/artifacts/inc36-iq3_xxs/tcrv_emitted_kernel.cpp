#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_ggml_vec_dot_iq3_xxs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const uint32_t tcrv_iq3xxs_grid[256] = {0x04040404U, 0x04040414U, 0x04040424U, 0x04040c0cU, 0x04040c1cU, 0x04040c3eU, 0x04041404U, 0x04041414U, 0x04041c0cU, 0x04042414U, 0x04043e1cU, 0x04043e2cU, 0x040c040cU, 0x040c041cU, 0x040c0c04U, 0x040c0c14U, 0x040c140cU, 0x040c142cU, 0x040c1c04U, 0x040c1c14U, 0x040c240cU, 0x040c2c24U, 0x040c3e04U, 0x04140404U, 0x04140414U, 0x04140424U, 0x04140c0cU, 0x04141404U, 0x04141414U, 0x04141c0cU, 0x04141c1cU, 0x04141c3eU, 0x04142c0cU, 0x04142c3eU, 0x04143e2cU, 0x041c040cU, 0x041c043eU, 0x041c0c04U, 0x041c0c14U, 0x041c142cU, 0x041c3e04U, 0x04240c1cU, 0x04241c3eU, 0x04242424U, 0x04242c3eU, 0x04243e1cU, 0x04243e2cU, 0x042c040cU, 0x042c043eU, 0x042c1c14U, 0x042c2c14U, 0x04341c2cU, 0x04343424U, 0x043e0c04U, 0x043e0c24U, 0x043e0c34U, 0x043e241cU, 0x043e340cU, 0x0c04040cU, 0x0c04041cU, 0x0c040c04U, 0x0c040c14U, 0x0c04140cU, 0x0c04141cU, 0x0c041c04U, 0x0c041c14U, 0x0c041c24U, 0x0c04243eU, 0x0c042c04U, 0x0c0c0404U, 0x0c0c0414U, 0x0c0c0c0cU, 0x0c0c1404U, 0x0c0c1414U, 0x0c14040cU, 0x0c14041cU, 0x0c140c04U, 0x0c140c14U, 0x0c14140cU, 0x0c141c04U, 0x0c143e14U, 0x0c1c0404U, 0x0c1c0414U, 0x0c1c1404U, 0x0c1c1c0cU, 0x0c1c2434U, 0x0c1c3434U, 0x0c24040cU, 0x0c24042cU, 0x0c242c04U, 0x0c2c1404U, 0x0c2c1424U, 0x0c2c2434U, 0x0c2c3e0cU, 0x0c34042cU, 0x0c3e1414U, 0x0c3e2404U, 0x14040404U, 0x14040414U, 0x14040c0cU, 0x14040c1cU, 0x14041404U, 0x14041414U, 0x14041434U, 0x14041c0cU, 0x14042414U, 0x140c040cU, 0x140c041cU, 0x140c042cU, 0x140c0c04U, 0x140c0c14U, 0x140c140cU, 0x140c1c04U, 0x140c341cU, 0x140c343eU, 0x140c3e04U, 0x14140404U, 0x14140414U, 0x14140c0cU, 0x14140c3eU, 0x14141404U, 0x14141414U, 0x14141c3eU, 0x14142404U, 0x14142c2cU, 0x141c040cU, 0x141c0c04U, 0x141c0c24U, 0x141c3e04U, 0x141c3e24U, 0x14241c2cU, 0x14242c1cU, 0x142c041cU, 0x142c143eU, 0x142c240cU, 0x142c3e24U, 0x143e040cU, 0x143e041cU, 0x143e0c34U, 0x143e242cU, 0x1c04040cU, 0x1c040c04U, 0x1c040c14U, 0x1c04140cU, 0x1c04141cU, 0x1c042c04U, 0x1c04342cU, 0x1c043e14U, 0x1c0c0404U, 0x1c0c0414U, 0x1c0c1404U, 0x1c0c1c0cU, 0x1c0c2424U, 0x1c0c2434U, 0x1c14040cU, 0x1c14041cU, 0x1c140c04U, 0x1c14142cU, 0x1c142c14U, 0x1c143e14U, 0x1c1c0c0cU, 0x1c1c1c1cU, 0x1c241c04U, 0x1c24243eU, 0x1c243e14U, 0x1c2c0404U, 0x1c2c0434U, 0x1c2c1414U, 0x1c2c2c2cU, 0x1c340c24U, 0x1c341c34U, 0x1c34341cU, 0x1c3e1c1cU, 0x1c3e3404U, 0x24040424U, 0x24040c3eU, 0x24041c2cU, 0x24041c3eU, 0x24042c1cU, 0x24042c3eU, 0x240c3e24U, 0x24141404U, 0x24141c3eU, 0x24142404U, 0x24143404U, 0x24143434U, 0x241c043eU, 0x241c242cU, 0x24240424U, 0x24242c0cU, 0x24243424U, 0x242c142cU, 0x242c241cU, 0x242c3e04U, 0x243e042cU, 0x243e0c04U, 0x243e0c14U, 0x243e1c04U, 0x2c040c14U, 0x2c04240cU, 0x2c043e04U, 0x2c0c0404U, 0x2c0c0434U, 0x2c0c1434U, 0x2c0c2c2cU, 0x2c140c24U, 0x2c141c14U, 0x2c143e14U, 0x2c1c0414U, 0x2c1c2c1cU, 0x2c240c04U, 0x2c24141cU, 0x2c24143eU, 0x2c243e14U, 0x2c2c0414U, 0x2c2c1c0cU, 0x2c342c04U, 0x2c3e1424U, 0x2c3e2414U, 0x34041424U, 0x34042424U, 0x34042434U, 0x34043424U, 0x340c140cU, 0x340c340cU, 0x34140c3eU, 0x34143424U, 0x341c1c04U, 0x341c1c34U, 0x34242424U, 0x342c042cU, 0x342c2c14U, 0x34341c1cU, 0x343e041cU, 0x343e140cU, 0x3e04041cU, 0x3e04042cU, 0x3e04043eU, 0x3e040c04U, 0x3e041c14U, 0x3e042c14U, 0x3e0c1434U, 0x3e0c2404U, 0x3e140c14U, 0x3e14242cU, 0x3e142c14U, 0x3e1c0404U, 0x3e1c0c2cU, 0x3e1c1c1cU, 0x3e1c3404U, 0x3e24140cU, 0x3e24240cU, 0x3e2c0404U, 0x3e2c0414U, 0x3e2c1424U, 0x3e341c04U};
  static const uint8_t tcrv_iq3xxs_ksigns[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
  static const uint8_t tcrv_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq3xxs_kmask, 4);
  const uint8_t* v9 = tcrv_iq3xxs_kmask + 4;
  vuint8m1_t v10 = __riscv_vle8_v_u8m1(v9, 4);
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_byte_view
  const int8_t* v11 = (const int8_t*) tcrv_iq3xxs_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v12 = 0; v12 < v7; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v13 = v12 * 98;
    const uint8_t* v14 = v3 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v15 = v12 * 292;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v17 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v18 = (const float*) v16;
    const float v19 = v18[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v20 = v17 * v19;
    const uint8_t* v21 = v14 + 2;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t* v23 = v14 + 66;
    const uint8_t* v24 = (const uint8_t*) v23;
    const uint8_t* v25 = v16 + 4;
    const int8_t* v26 = (const int8_t*) v25;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v27;
    v27 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v28 = v24[0];
    uint32_t v29 = (uint32_t) v28;
    const uint8_t v30 = v24[1];
    uint32_t v31 = (uint32_t) v30;
    uint32_t v32 = v31 << 8u;
    uint32_t v33 = v29 | v32;
    const uint8_t v34 = v24[2];
    uint32_t v35 = (uint32_t) v34;
    uint32_t v36 = v35 << 16u;
    uint32_t v37 = v33 | v36;
    const uint8_t v38 = v24[3];
    uint32_t v39 = (uint32_t) v38;
    uint32_t v40 = v39 << 24u;
    uint32_t v41 = v37 | v40;
    uint32_t v42 = v41 >> 28u;
    int v43 = (int) v42;
    int v44 = v43 * 2;
    int v45 = v44 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v46 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v47 = v41 >> 0u;
    uint32_t v48 = v47 & 127u;
    int v49 = (int) v48;
    const uint8_t v50 = tcrv_iq3xxs_ksigns[v49];
    int v51 = (int) v50;
    const int8_t* v52 = v26 + 4;
    const uint8_t v53 = v22[0];
    int v54 = (int) v53;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v55 = __riscv_vsetvl_e8m1(4);
    size_t v56 = v54 * 4;
    const int8_t* v57 = v11 + v56;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v58 = __riscv_vle8_v_i8m1(v57, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v59 = __riscv_vle8_v_i8m1(v26, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v60 = __riscv_vmv_v_x_u8m1(v51, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v61 = __riscv_vand_vv_u8m1(v60, v8, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v62 = __riscv_vmsne_vx_u8m1_b8(v61, 0, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v63 = __riscv_vneg_v_i8m1(v58, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v64 = __riscv_vmerge_vvm_i8m1(v58, v63, v62, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v65 = __riscv_vwmul_vv_i16m2(v64, v59, v55);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v66 = __riscv_vwredsum_vs_i16m2_i32m1(v65, v46, v55);
    const uint8_t v67 = v22[1];
    int v68 = (int) v67;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v69 = __riscv_vsetvl_e8m1(4);
    size_t v70 = v68 * 4;
    const int8_t* v71 = v11 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v72 = __riscv_vle8_v_i8m1(v71, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v73 = __riscv_vle8_v_i8m1(v52, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v74 = __riscv_vmv_v_x_u8m1(v51, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v75 = __riscv_vand_vv_u8m1(v74, v10, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v76 = __riscv_vmsne_vx_u8m1_b8(v75, 0, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v77 = __riscv_vneg_v_i8m1(v72, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v78 = __riscv_vmerge_vvm_i8m1(v72, v77, v76, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v79 = __riscv_vwmul_vv_i16m2(v78, v73, v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v80 = __riscv_vwredsum_vs_i16m2_i32m1(v79, v66, v69);
    const int8_t* v81 = v26 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v82 = v41 >> 7u;
    uint32_t v83 = v82 & 127u;
    int v84 = (int) v83;
    const uint8_t v85 = tcrv_iq3xxs_ksigns[v84];
    int v86 = (int) v85;
    const int8_t* v87 = v81 + 4;
    const uint8_t v88 = v22[2];
    int v89 = (int) v88;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v90 = __riscv_vsetvl_e8m1(4);
    size_t v91 = v89 * 4;
    const int8_t* v92 = v11 + v91;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v92, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v94 = __riscv_vle8_v_i8m1(v81, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v95 = __riscv_vmv_v_x_u8m1(v86, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v96 = __riscv_vand_vv_u8m1(v95, v8, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v97 = __riscv_vmsne_vx_u8m1_b8(v96, 0, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v98 = __riscv_vneg_v_i8m1(v93, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v99 = __riscv_vmerge_vvm_i8m1(v93, v98, v97, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v100 = __riscv_vwmul_vv_i16m2(v99, v94, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v101 = __riscv_vwredsum_vs_i16m2_i32m1(v100, v80, v90);
    const uint8_t v102 = v22[3];
    int v103 = (int) v102;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v104 = __riscv_vsetvl_e8m1(4);
    size_t v105 = v103 * 4;
    const int8_t* v106 = v11 + v105;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v107 = __riscv_vle8_v_i8m1(v106, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v108 = __riscv_vle8_v_i8m1(v87, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v109 = __riscv_vmv_v_x_u8m1(v86, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v110 = __riscv_vand_vv_u8m1(v109, v10, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v111 = __riscv_vmsne_vx_u8m1_b8(v110, 0, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v112 = __riscv_vneg_v_i8m1(v107, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v113 = __riscv_vmerge_vvm_i8m1(v107, v112, v111, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v114 = __riscv_vwmul_vv_i16m2(v113, v108, v104);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v115 = __riscv_vwredsum_vs_i16m2_i32m1(v114, v101, v104);
    const int8_t* v116 = v81 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v117 = v41 >> 14u;
    uint32_t v118 = v117 & 127u;
    int v119 = (int) v118;
    const uint8_t v120 = tcrv_iq3xxs_ksigns[v119];
    int v121 = (int) v120;
    const int8_t* v122 = v116 + 4;
    const uint8_t v123 = v22[4];
    int v124 = (int) v123;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v125 = __riscv_vsetvl_e8m1(4);
    size_t v126 = v124 * 4;
    const int8_t* v127 = v11 + v126;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v128 = __riscv_vle8_v_i8m1(v127, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v129 = __riscv_vle8_v_i8m1(v116, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v130 = __riscv_vmv_v_x_u8m1(v121, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v131 = __riscv_vand_vv_u8m1(v130, v8, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v132 = __riscv_vmsne_vx_u8m1_b8(v131, 0, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v133 = __riscv_vneg_v_i8m1(v128, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v134 = __riscv_vmerge_vvm_i8m1(v128, v133, v132, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v135 = __riscv_vwmul_vv_i16m2(v134, v129, v125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v136 = __riscv_vwredsum_vs_i16m2_i32m1(v135, v115, v125);
    const uint8_t v137 = v22[5];
    int v138 = (int) v137;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v139 = __riscv_vsetvl_e8m1(4);
    size_t v140 = v138 * 4;
    const int8_t* v141 = v11 + v140;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v142 = __riscv_vle8_v_i8m1(v141, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v143 = __riscv_vle8_v_i8m1(v122, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v144 = __riscv_vmv_v_x_u8m1(v121, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v145 = __riscv_vand_vv_u8m1(v144, v10, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v146 = __riscv_vmsne_vx_u8m1_b8(v145, 0, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v147 = __riscv_vneg_v_i8m1(v142, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v148 = __riscv_vmerge_vvm_i8m1(v142, v147, v146, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v149 = __riscv_vwmul_vv_i16m2(v148, v143, v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v150 = __riscv_vwredsum_vs_i16m2_i32m1(v149, v136, v139);
    const int8_t* v151 = v116 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v152 = v41 >> 21u;
    uint32_t v153 = v152 & 127u;
    int v154 = (int) v153;
    const uint8_t v155 = tcrv_iq3xxs_ksigns[v154];
    int v156 = (int) v155;
    const int8_t* v157 = v151 + 4;
    const uint8_t v158 = v22[6];
    int v159 = (int) v158;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v160 = __riscv_vsetvl_e8m1(4);
    size_t v161 = v159 * 4;
    const int8_t* v162 = v11 + v161;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v163 = __riscv_vle8_v_i8m1(v162, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v164 = __riscv_vle8_v_i8m1(v151, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v165 = __riscv_vmv_v_x_u8m1(v156, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v166 = __riscv_vand_vv_u8m1(v165, v8, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v167 = __riscv_vmsne_vx_u8m1_b8(v166, 0, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v168 = __riscv_vneg_v_i8m1(v163, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v169 = __riscv_vmerge_vvm_i8m1(v163, v168, v167, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v170 = __riscv_vwmul_vv_i16m2(v169, v164, v160);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v171 = __riscv_vwredsum_vs_i16m2_i32m1(v170, v150, v160);
    const uint8_t v172 = v22[7];
    int v173 = (int) v172;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v174 = __riscv_vsetvl_e8m1(4);
    size_t v175 = v173 * 4;
    const int8_t* v176 = v11 + v175;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v177 = __riscv_vle8_v_i8m1(v176, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v178 = __riscv_vle8_v_i8m1(v157, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v179 = __riscv_vmv_v_x_u8m1(v156, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v180 = __riscv_vand_vv_u8m1(v179, v10, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v181 = __riscv_vmsne_vx_u8m1_b8(v180, 0, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v182 = __riscv_vneg_v_i8m1(v177, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v183 = __riscv_vmerge_vvm_i8m1(v177, v182, v181, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v184 = __riscv_vwmul_vv_i16m2(v183, v178, v174);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v185 = __riscv_vwredsum_vs_i16m2_i32m1(v184, v171, v174);
    const int8_t* v186 = v151 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v187 = __riscv_vmv_x_s_i32m1_i32(v185);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v188 = v27;
    int32_t v189 = (int32_t) v45;
    int32_t v190 = v187 * v189;
    int32_t v191 = v188 + v190;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v191;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v192 = v24 + 4;
    const uint8_t v193 = v192[0];
    uint32_t v194 = (uint32_t) v193;
    const uint8_t v195 = v192[1];
    uint32_t v196 = (uint32_t) v195;
    uint32_t v197 = v196 << 8u;
    uint32_t v198 = v194 | v197;
    const uint8_t v199 = v192[2];
    uint32_t v200 = (uint32_t) v199;
    uint32_t v201 = v200 << 16u;
    uint32_t v202 = v198 | v201;
    const uint8_t v203 = v192[3];
    uint32_t v204 = (uint32_t) v203;
    uint32_t v205 = v204 << 24u;
    uint32_t v206 = v202 | v205;
    uint32_t v207 = v206 >> 28u;
    int v208 = (int) v207;
    int v209 = v208 * 2;
    int v210 = v209 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v211 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v212 = v22 + 8;
    const int8_t* v213 = v26 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v214 = v206 >> 0u;
    uint32_t v215 = v214 & 127u;
    int v216 = (int) v215;
    const uint8_t v217 = tcrv_iq3xxs_ksigns[v216];
    int v218 = (int) v217;
    const int8_t* v219 = v213 + 4;
    const uint8_t v220 = v212[0];
    int v221 = (int) v220;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v222 = __riscv_vsetvl_e8m1(4);
    size_t v223 = v221 * 4;
    const int8_t* v224 = v11 + v223;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v225 = __riscv_vle8_v_i8m1(v224, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v226 = __riscv_vle8_v_i8m1(v213, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v227 = __riscv_vmv_v_x_u8m1(v218, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v228 = __riscv_vand_vv_u8m1(v227, v8, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v229 = __riscv_vmsne_vx_u8m1_b8(v228, 0, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v230 = __riscv_vneg_v_i8m1(v225, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v231 = __riscv_vmerge_vvm_i8m1(v225, v230, v229, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v232 = __riscv_vwmul_vv_i16m2(v231, v226, v222);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v233 = __riscv_vwredsum_vs_i16m2_i32m1(v232, v211, v222);
    const uint8_t v234 = v212[1];
    int v235 = (int) v234;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v236 = __riscv_vsetvl_e8m1(4);
    size_t v237 = v235 * 4;
    const int8_t* v238 = v11 + v237;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v239 = __riscv_vle8_v_i8m1(v238, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v240 = __riscv_vle8_v_i8m1(v219, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v241 = __riscv_vmv_v_x_u8m1(v218, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v242 = __riscv_vand_vv_u8m1(v241, v10, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v243 = __riscv_vmsne_vx_u8m1_b8(v242, 0, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v244 = __riscv_vneg_v_i8m1(v239, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v245 = __riscv_vmerge_vvm_i8m1(v239, v244, v243, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v246 = __riscv_vwmul_vv_i16m2(v245, v240, v236);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v247 = __riscv_vwredsum_vs_i16m2_i32m1(v246, v233, v236);
    const int8_t* v248 = v213 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v249 = v206 >> 7u;
    uint32_t v250 = v249 & 127u;
    int v251 = (int) v250;
    const uint8_t v252 = tcrv_iq3xxs_ksigns[v251];
    int v253 = (int) v252;
    const int8_t* v254 = v248 + 4;
    const uint8_t v255 = v212[2];
    int v256 = (int) v255;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v257 = __riscv_vsetvl_e8m1(4);
    size_t v258 = v256 * 4;
    const int8_t* v259 = v11 + v258;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v260 = __riscv_vle8_v_i8m1(v259, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v261 = __riscv_vle8_v_i8m1(v248, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v262 = __riscv_vmv_v_x_u8m1(v253, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v263 = __riscv_vand_vv_u8m1(v262, v8, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v264 = __riscv_vmsne_vx_u8m1_b8(v263, 0, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v265 = __riscv_vneg_v_i8m1(v260, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v266 = __riscv_vmerge_vvm_i8m1(v260, v265, v264, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v267 = __riscv_vwmul_vv_i16m2(v266, v261, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v268 = __riscv_vwredsum_vs_i16m2_i32m1(v267, v247, v257);
    const uint8_t v269 = v212[3];
    int v270 = (int) v269;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v271 = __riscv_vsetvl_e8m1(4);
    size_t v272 = v270 * 4;
    const int8_t* v273 = v11 + v272;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v274 = __riscv_vle8_v_i8m1(v273, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v275 = __riscv_vle8_v_i8m1(v254, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v276 = __riscv_vmv_v_x_u8m1(v253, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v277 = __riscv_vand_vv_u8m1(v276, v10, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v278 = __riscv_vmsne_vx_u8m1_b8(v277, 0, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v279 = __riscv_vneg_v_i8m1(v274, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v280 = __riscv_vmerge_vvm_i8m1(v274, v279, v278, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v281 = __riscv_vwmul_vv_i16m2(v280, v275, v271);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v282 = __riscv_vwredsum_vs_i16m2_i32m1(v281, v268, v271);
    const int8_t* v283 = v248 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v284 = v206 >> 14u;
    uint32_t v285 = v284 & 127u;
    int v286 = (int) v285;
    const uint8_t v287 = tcrv_iq3xxs_ksigns[v286];
    int v288 = (int) v287;
    const int8_t* v289 = v283 + 4;
    const uint8_t v290 = v212[4];
    int v291 = (int) v290;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v292 = __riscv_vsetvl_e8m1(4);
    size_t v293 = v291 * 4;
    const int8_t* v294 = v11 + v293;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v295 = __riscv_vle8_v_i8m1(v294, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v296 = __riscv_vle8_v_i8m1(v283, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v297 = __riscv_vmv_v_x_u8m1(v288, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v298 = __riscv_vand_vv_u8m1(v297, v8, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v299 = __riscv_vmsne_vx_u8m1_b8(v298, 0, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v300 = __riscv_vneg_v_i8m1(v295, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v301 = __riscv_vmerge_vvm_i8m1(v295, v300, v299, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v302 = __riscv_vwmul_vv_i16m2(v301, v296, v292);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v303 = __riscv_vwredsum_vs_i16m2_i32m1(v302, v282, v292);
    const uint8_t v304 = v212[5];
    int v305 = (int) v304;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v306 = __riscv_vsetvl_e8m1(4);
    size_t v307 = v305 * 4;
    const int8_t* v308 = v11 + v307;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v309 = __riscv_vle8_v_i8m1(v308, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v310 = __riscv_vle8_v_i8m1(v289, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v311 = __riscv_vmv_v_x_u8m1(v288, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v312 = __riscv_vand_vv_u8m1(v311, v10, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v313 = __riscv_vmsne_vx_u8m1_b8(v312, 0, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v314 = __riscv_vneg_v_i8m1(v309, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v315 = __riscv_vmerge_vvm_i8m1(v309, v314, v313, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v316 = __riscv_vwmul_vv_i16m2(v315, v310, v306);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v317 = __riscv_vwredsum_vs_i16m2_i32m1(v316, v303, v306);
    const int8_t* v318 = v283 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v319 = v206 >> 21u;
    uint32_t v320 = v319 & 127u;
    int v321 = (int) v320;
    const uint8_t v322 = tcrv_iq3xxs_ksigns[v321];
    int v323 = (int) v322;
    const int8_t* v324 = v318 + 4;
    const uint8_t v325 = v212[6];
    int v326 = (int) v325;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v327 = __riscv_vsetvl_e8m1(4);
    size_t v328 = v326 * 4;
    const int8_t* v329 = v11 + v328;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v330 = __riscv_vle8_v_i8m1(v329, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v331 = __riscv_vle8_v_i8m1(v318, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v332 = __riscv_vmv_v_x_u8m1(v323, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v333 = __riscv_vand_vv_u8m1(v332, v8, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v334 = __riscv_vmsne_vx_u8m1_b8(v333, 0, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v335 = __riscv_vneg_v_i8m1(v330, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v336 = __riscv_vmerge_vvm_i8m1(v330, v335, v334, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v337 = __riscv_vwmul_vv_i16m2(v336, v331, v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v338 = __riscv_vwredsum_vs_i16m2_i32m1(v337, v317, v327);
    const uint8_t v339 = v212[7];
    int v340 = (int) v339;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v341 = __riscv_vsetvl_e8m1(4);
    size_t v342 = v340 * 4;
    const int8_t* v343 = v11 + v342;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v344 = __riscv_vle8_v_i8m1(v343, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v345 = __riscv_vle8_v_i8m1(v324, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v346 = __riscv_vmv_v_x_u8m1(v323, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v347 = __riscv_vand_vv_u8m1(v346, v10, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v348 = __riscv_vmsne_vx_u8m1_b8(v347, 0, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v349 = __riscv_vneg_v_i8m1(v344, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v350 = __riscv_vmerge_vvm_i8m1(v344, v349, v348, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v351 = __riscv_vwmul_vv_i16m2(v350, v345, v341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v352 = __riscv_vwredsum_vs_i16m2_i32m1(v351, v338, v341);
    const int8_t* v353 = v318 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v354 = __riscv_vmv_x_s_i32m1_i32(v352);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v355 = v27;
    int32_t v356 = (int32_t) v210;
    int32_t v357 = v354 * v356;
    int32_t v358 = v355 + v357;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v358;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v359 = v24 + 8;
    const uint8_t v360 = v359[0];
    uint32_t v361 = (uint32_t) v360;
    const uint8_t v362 = v359[1];
    uint32_t v363 = (uint32_t) v362;
    uint32_t v364 = v363 << 8u;
    uint32_t v365 = v361 | v364;
    const uint8_t v366 = v359[2];
    uint32_t v367 = (uint32_t) v366;
    uint32_t v368 = v367 << 16u;
    uint32_t v369 = v365 | v368;
    const uint8_t v370 = v359[3];
    uint32_t v371 = (uint32_t) v370;
    uint32_t v372 = v371 << 24u;
    uint32_t v373 = v369 | v372;
    uint32_t v374 = v373 >> 28u;
    int v375 = (int) v374;
    int v376 = v375 * 2;
    int v377 = v376 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v378 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v379 = v22 + 16;
    const int8_t* v380 = v26 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v381 = v373 >> 0u;
    uint32_t v382 = v381 & 127u;
    int v383 = (int) v382;
    const uint8_t v384 = tcrv_iq3xxs_ksigns[v383];
    int v385 = (int) v384;
    const int8_t* v386 = v380 + 4;
    const uint8_t v387 = v379[0];
    int v388 = (int) v387;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v389 = __riscv_vsetvl_e8m1(4);
    size_t v390 = v388 * 4;
    const int8_t* v391 = v11 + v390;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v392 = __riscv_vle8_v_i8m1(v391, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v393 = __riscv_vle8_v_i8m1(v380, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v394 = __riscv_vmv_v_x_u8m1(v385, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v395 = __riscv_vand_vv_u8m1(v394, v8, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v396 = __riscv_vmsne_vx_u8m1_b8(v395, 0, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v397 = __riscv_vneg_v_i8m1(v392, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v398 = __riscv_vmerge_vvm_i8m1(v392, v397, v396, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v399 = __riscv_vwmul_vv_i16m2(v398, v393, v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v400 = __riscv_vwredsum_vs_i16m2_i32m1(v399, v378, v389);
    const uint8_t v401 = v379[1];
    int v402 = (int) v401;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v403 = __riscv_vsetvl_e8m1(4);
    size_t v404 = v402 * 4;
    const int8_t* v405 = v11 + v404;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v406 = __riscv_vle8_v_i8m1(v405, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v407 = __riscv_vle8_v_i8m1(v386, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v408 = __riscv_vmv_v_x_u8m1(v385, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v409 = __riscv_vand_vv_u8m1(v408, v10, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v410 = __riscv_vmsne_vx_u8m1_b8(v409, 0, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v411 = __riscv_vneg_v_i8m1(v406, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v412 = __riscv_vmerge_vvm_i8m1(v406, v411, v410, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v413 = __riscv_vwmul_vv_i16m2(v412, v407, v403);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v414 = __riscv_vwredsum_vs_i16m2_i32m1(v413, v400, v403);
    const int8_t* v415 = v380 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v416 = v373 >> 7u;
    uint32_t v417 = v416 & 127u;
    int v418 = (int) v417;
    const uint8_t v419 = tcrv_iq3xxs_ksigns[v418];
    int v420 = (int) v419;
    const int8_t* v421 = v415 + 4;
    const uint8_t v422 = v379[2];
    int v423 = (int) v422;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v424 = __riscv_vsetvl_e8m1(4);
    size_t v425 = v423 * 4;
    const int8_t* v426 = v11 + v425;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v427 = __riscv_vle8_v_i8m1(v426, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v428 = __riscv_vle8_v_i8m1(v415, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v429 = __riscv_vmv_v_x_u8m1(v420, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v430 = __riscv_vand_vv_u8m1(v429, v8, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v431 = __riscv_vmsne_vx_u8m1_b8(v430, 0, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v432 = __riscv_vneg_v_i8m1(v427, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v433 = __riscv_vmerge_vvm_i8m1(v427, v432, v431, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v434 = __riscv_vwmul_vv_i16m2(v433, v428, v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v435 = __riscv_vwredsum_vs_i16m2_i32m1(v434, v414, v424);
    const uint8_t v436 = v379[3];
    int v437 = (int) v436;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v438 = __riscv_vsetvl_e8m1(4);
    size_t v439 = v437 * 4;
    const int8_t* v440 = v11 + v439;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v441 = __riscv_vle8_v_i8m1(v440, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v442 = __riscv_vle8_v_i8m1(v421, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v443 = __riscv_vmv_v_x_u8m1(v420, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v444 = __riscv_vand_vv_u8m1(v443, v10, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v445 = __riscv_vmsne_vx_u8m1_b8(v444, 0, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v446 = __riscv_vneg_v_i8m1(v441, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v447 = __riscv_vmerge_vvm_i8m1(v441, v446, v445, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v448 = __riscv_vwmul_vv_i16m2(v447, v442, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v449 = __riscv_vwredsum_vs_i16m2_i32m1(v448, v435, v438);
    const int8_t* v450 = v415 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v451 = v373 >> 14u;
    uint32_t v452 = v451 & 127u;
    int v453 = (int) v452;
    const uint8_t v454 = tcrv_iq3xxs_ksigns[v453];
    int v455 = (int) v454;
    const int8_t* v456 = v450 + 4;
    const uint8_t v457 = v379[4];
    int v458 = (int) v457;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v459 = __riscv_vsetvl_e8m1(4);
    size_t v460 = v458 * 4;
    const int8_t* v461 = v11 + v460;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v462 = __riscv_vle8_v_i8m1(v461, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v463 = __riscv_vle8_v_i8m1(v450, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v464 = __riscv_vmv_v_x_u8m1(v455, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v465 = __riscv_vand_vv_u8m1(v464, v8, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v466 = __riscv_vmsne_vx_u8m1_b8(v465, 0, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v467 = __riscv_vneg_v_i8m1(v462, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v468 = __riscv_vmerge_vvm_i8m1(v462, v467, v466, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v469 = __riscv_vwmul_vv_i16m2(v468, v463, v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v470 = __riscv_vwredsum_vs_i16m2_i32m1(v469, v449, v459);
    const uint8_t v471 = v379[5];
    int v472 = (int) v471;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v473 = __riscv_vsetvl_e8m1(4);
    size_t v474 = v472 * 4;
    const int8_t* v475 = v11 + v474;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v476 = __riscv_vle8_v_i8m1(v475, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v477 = __riscv_vle8_v_i8m1(v456, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v478 = __riscv_vmv_v_x_u8m1(v455, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v479 = __riscv_vand_vv_u8m1(v478, v10, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v480 = __riscv_vmsne_vx_u8m1_b8(v479, 0, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v481 = __riscv_vneg_v_i8m1(v476, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v482 = __riscv_vmerge_vvm_i8m1(v476, v481, v480, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v483 = __riscv_vwmul_vv_i16m2(v482, v477, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v484 = __riscv_vwredsum_vs_i16m2_i32m1(v483, v470, v473);
    const int8_t* v485 = v450 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v486 = v373 >> 21u;
    uint32_t v487 = v486 & 127u;
    int v488 = (int) v487;
    const uint8_t v489 = tcrv_iq3xxs_ksigns[v488];
    int v490 = (int) v489;
    const int8_t* v491 = v485 + 4;
    const uint8_t v492 = v379[6];
    int v493 = (int) v492;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v494 = __riscv_vsetvl_e8m1(4);
    size_t v495 = v493 * 4;
    const int8_t* v496 = v11 + v495;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v497 = __riscv_vle8_v_i8m1(v496, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v498 = __riscv_vle8_v_i8m1(v485, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v499 = __riscv_vmv_v_x_u8m1(v490, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v500 = __riscv_vand_vv_u8m1(v499, v8, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v501 = __riscv_vmsne_vx_u8m1_b8(v500, 0, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v502 = __riscv_vneg_v_i8m1(v497, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v503 = __riscv_vmerge_vvm_i8m1(v497, v502, v501, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v504 = __riscv_vwmul_vv_i16m2(v503, v498, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v505 = __riscv_vwredsum_vs_i16m2_i32m1(v504, v484, v494);
    const uint8_t v506 = v379[7];
    int v507 = (int) v506;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v508 = __riscv_vsetvl_e8m1(4);
    size_t v509 = v507 * 4;
    const int8_t* v510 = v11 + v509;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v511 = __riscv_vle8_v_i8m1(v510, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v512 = __riscv_vle8_v_i8m1(v491, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v513 = __riscv_vmv_v_x_u8m1(v490, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v514 = __riscv_vand_vv_u8m1(v513, v10, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v515 = __riscv_vmsne_vx_u8m1_b8(v514, 0, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v516 = __riscv_vneg_v_i8m1(v511, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v517 = __riscv_vmerge_vvm_i8m1(v511, v516, v515, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v518 = __riscv_vwmul_vv_i16m2(v517, v512, v508);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v519 = __riscv_vwredsum_vs_i16m2_i32m1(v518, v505, v508);
    const int8_t* v520 = v485 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v521 = __riscv_vmv_x_s_i32m1_i32(v519);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v522 = v27;
    int32_t v523 = (int32_t) v377;
    int32_t v524 = v521 * v523;
    int32_t v525 = v522 + v524;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v525;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v526 = v24 + 12;
    const uint8_t v527 = v526[0];
    uint32_t v528 = (uint32_t) v527;
    const uint8_t v529 = v526[1];
    uint32_t v530 = (uint32_t) v529;
    uint32_t v531 = v530 << 8u;
    uint32_t v532 = v528 | v531;
    const uint8_t v533 = v526[2];
    uint32_t v534 = (uint32_t) v533;
    uint32_t v535 = v534 << 16u;
    uint32_t v536 = v532 | v535;
    const uint8_t v537 = v526[3];
    uint32_t v538 = (uint32_t) v537;
    uint32_t v539 = v538 << 24u;
    uint32_t v540 = v536 | v539;
    uint32_t v541 = v540 >> 28u;
    int v542 = (int) v541;
    int v543 = v542 * 2;
    int v544 = v543 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v545 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v546 = v22 + 24;
    const int8_t* v547 = v26 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v548 = v540 >> 0u;
    uint32_t v549 = v548 & 127u;
    int v550 = (int) v549;
    const uint8_t v551 = tcrv_iq3xxs_ksigns[v550];
    int v552 = (int) v551;
    const int8_t* v553 = v547 + 4;
    const uint8_t v554 = v546[0];
    int v555 = (int) v554;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v556 = __riscv_vsetvl_e8m1(4);
    size_t v557 = v555 * 4;
    const int8_t* v558 = v11 + v557;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v559 = __riscv_vle8_v_i8m1(v558, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v560 = __riscv_vle8_v_i8m1(v547, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v561 = __riscv_vmv_v_x_u8m1(v552, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v562 = __riscv_vand_vv_u8m1(v561, v8, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v563 = __riscv_vmsne_vx_u8m1_b8(v562, 0, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v564 = __riscv_vneg_v_i8m1(v559, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v565 = __riscv_vmerge_vvm_i8m1(v559, v564, v563, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v566 = __riscv_vwmul_vv_i16m2(v565, v560, v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v567 = __riscv_vwredsum_vs_i16m2_i32m1(v566, v545, v556);
    const uint8_t v568 = v546[1];
    int v569 = (int) v568;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v570 = __riscv_vsetvl_e8m1(4);
    size_t v571 = v569 * 4;
    const int8_t* v572 = v11 + v571;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v573 = __riscv_vle8_v_i8m1(v572, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v574 = __riscv_vle8_v_i8m1(v553, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v575 = __riscv_vmv_v_x_u8m1(v552, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v576 = __riscv_vand_vv_u8m1(v575, v10, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v577 = __riscv_vmsne_vx_u8m1_b8(v576, 0, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v578 = __riscv_vneg_v_i8m1(v573, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v579 = __riscv_vmerge_vvm_i8m1(v573, v578, v577, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v580 = __riscv_vwmul_vv_i16m2(v579, v574, v570);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v581 = __riscv_vwredsum_vs_i16m2_i32m1(v580, v567, v570);
    const int8_t* v582 = v547 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v583 = v540 >> 7u;
    uint32_t v584 = v583 & 127u;
    int v585 = (int) v584;
    const uint8_t v586 = tcrv_iq3xxs_ksigns[v585];
    int v587 = (int) v586;
    const int8_t* v588 = v582 + 4;
    const uint8_t v589 = v546[2];
    int v590 = (int) v589;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v591 = __riscv_vsetvl_e8m1(4);
    size_t v592 = v590 * 4;
    const int8_t* v593 = v11 + v592;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v594 = __riscv_vle8_v_i8m1(v593, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v595 = __riscv_vle8_v_i8m1(v582, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v596 = __riscv_vmv_v_x_u8m1(v587, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v597 = __riscv_vand_vv_u8m1(v596, v8, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v598 = __riscv_vmsne_vx_u8m1_b8(v597, 0, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v599 = __riscv_vneg_v_i8m1(v594, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v600 = __riscv_vmerge_vvm_i8m1(v594, v599, v598, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v601 = __riscv_vwmul_vv_i16m2(v600, v595, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v602 = __riscv_vwredsum_vs_i16m2_i32m1(v601, v581, v591);
    const uint8_t v603 = v546[3];
    int v604 = (int) v603;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v605 = __riscv_vsetvl_e8m1(4);
    size_t v606 = v604 * 4;
    const int8_t* v607 = v11 + v606;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v608 = __riscv_vle8_v_i8m1(v607, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v609 = __riscv_vle8_v_i8m1(v588, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v610 = __riscv_vmv_v_x_u8m1(v587, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v611 = __riscv_vand_vv_u8m1(v610, v10, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v612 = __riscv_vmsne_vx_u8m1_b8(v611, 0, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v613 = __riscv_vneg_v_i8m1(v608, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v614 = __riscv_vmerge_vvm_i8m1(v608, v613, v612, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v615 = __riscv_vwmul_vv_i16m2(v614, v609, v605);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v616 = __riscv_vwredsum_vs_i16m2_i32m1(v615, v602, v605);
    const int8_t* v617 = v582 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v618 = v540 >> 14u;
    uint32_t v619 = v618 & 127u;
    int v620 = (int) v619;
    const uint8_t v621 = tcrv_iq3xxs_ksigns[v620];
    int v622 = (int) v621;
    const int8_t* v623 = v617 + 4;
    const uint8_t v624 = v546[4];
    int v625 = (int) v624;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v626 = __riscv_vsetvl_e8m1(4);
    size_t v627 = v625 * 4;
    const int8_t* v628 = v11 + v627;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v629 = __riscv_vle8_v_i8m1(v628, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v630 = __riscv_vle8_v_i8m1(v617, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v631 = __riscv_vmv_v_x_u8m1(v622, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v632 = __riscv_vand_vv_u8m1(v631, v8, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v633 = __riscv_vmsne_vx_u8m1_b8(v632, 0, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v634 = __riscv_vneg_v_i8m1(v629, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v635 = __riscv_vmerge_vvm_i8m1(v629, v634, v633, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v636 = __riscv_vwmul_vv_i16m2(v635, v630, v626);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v637 = __riscv_vwredsum_vs_i16m2_i32m1(v636, v616, v626);
    const uint8_t v638 = v546[5];
    int v639 = (int) v638;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v640 = __riscv_vsetvl_e8m1(4);
    size_t v641 = v639 * 4;
    const int8_t* v642 = v11 + v641;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v643 = __riscv_vle8_v_i8m1(v642, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v644 = __riscv_vle8_v_i8m1(v623, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v645 = __riscv_vmv_v_x_u8m1(v622, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v646 = __riscv_vand_vv_u8m1(v645, v10, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v647 = __riscv_vmsne_vx_u8m1_b8(v646, 0, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v648 = __riscv_vneg_v_i8m1(v643, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v649 = __riscv_vmerge_vvm_i8m1(v643, v648, v647, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v650 = __riscv_vwmul_vv_i16m2(v649, v644, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v651 = __riscv_vwredsum_vs_i16m2_i32m1(v650, v637, v640);
    const int8_t* v652 = v617 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v653 = v540 >> 21u;
    uint32_t v654 = v653 & 127u;
    int v655 = (int) v654;
    const uint8_t v656 = tcrv_iq3xxs_ksigns[v655];
    int v657 = (int) v656;
    const int8_t* v658 = v652 + 4;
    const uint8_t v659 = v546[6];
    int v660 = (int) v659;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v661 = __riscv_vsetvl_e8m1(4);
    size_t v662 = v660 * 4;
    const int8_t* v663 = v11 + v662;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v664 = __riscv_vle8_v_i8m1(v663, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v665 = __riscv_vle8_v_i8m1(v652, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v666 = __riscv_vmv_v_x_u8m1(v657, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v667 = __riscv_vand_vv_u8m1(v666, v8, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v668 = __riscv_vmsne_vx_u8m1_b8(v667, 0, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v669 = __riscv_vneg_v_i8m1(v664, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v670 = __riscv_vmerge_vvm_i8m1(v664, v669, v668, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v671 = __riscv_vwmul_vv_i16m2(v670, v665, v661);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v672 = __riscv_vwredsum_vs_i16m2_i32m1(v671, v651, v661);
    const uint8_t v673 = v546[7];
    int v674 = (int) v673;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v675 = __riscv_vsetvl_e8m1(4);
    size_t v676 = v674 * 4;
    const int8_t* v677 = v11 + v676;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v678 = __riscv_vle8_v_i8m1(v677, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v679 = __riscv_vle8_v_i8m1(v658, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v680 = __riscv_vmv_v_x_u8m1(v657, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v681 = __riscv_vand_vv_u8m1(v680, v10, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v682 = __riscv_vmsne_vx_u8m1_b8(v681, 0, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v683 = __riscv_vneg_v_i8m1(v678, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v684 = __riscv_vmerge_vvm_i8m1(v678, v683, v682, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v685 = __riscv_vwmul_vv_i16m2(v684, v679, v675);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v686 = __riscv_vwredsum_vs_i16m2_i32m1(v685, v672, v675);
    const int8_t* v687 = v652 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v688 = __riscv_vmv_x_s_i32m1_i32(v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v689 = v27;
    int32_t v690 = (int32_t) v544;
    int32_t v691 = v688 * v690;
    int32_t v692 = v689 + v691;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v692;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v693 = v24 + 16;
    const uint8_t v694 = v693[0];
    uint32_t v695 = (uint32_t) v694;
    const uint8_t v696 = v693[1];
    uint32_t v697 = (uint32_t) v696;
    uint32_t v698 = v697 << 8u;
    uint32_t v699 = v695 | v698;
    const uint8_t v700 = v693[2];
    uint32_t v701 = (uint32_t) v700;
    uint32_t v702 = v701 << 16u;
    uint32_t v703 = v699 | v702;
    const uint8_t v704 = v693[3];
    uint32_t v705 = (uint32_t) v704;
    uint32_t v706 = v705 << 24u;
    uint32_t v707 = v703 | v706;
    uint32_t v708 = v707 >> 28u;
    int v709 = (int) v708;
    int v710 = v709 * 2;
    int v711 = v710 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v712 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v713 = v22 + 32;
    const int8_t* v714 = v26 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v715 = v707 >> 0u;
    uint32_t v716 = v715 & 127u;
    int v717 = (int) v716;
    const uint8_t v718 = tcrv_iq3xxs_ksigns[v717];
    int v719 = (int) v718;
    const int8_t* v720 = v714 + 4;
    const uint8_t v721 = v713[0];
    int v722 = (int) v721;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v723 = __riscv_vsetvl_e8m1(4);
    size_t v724 = v722 * 4;
    const int8_t* v725 = v11 + v724;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v726 = __riscv_vle8_v_i8m1(v725, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v727 = __riscv_vle8_v_i8m1(v714, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v728 = __riscv_vmv_v_x_u8m1(v719, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v729 = __riscv_vand_vv_u8m1(v728, v8, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v730 = __riscv_vmsne_vx_u8m1_b8(v729, 0, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v731 = __riscv_vneg_v_i8m1(v726, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v732 = __riscv_vmerge_vvm_i8m1(v726, v731, v730, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v733 = __riscv_vwmul_vv_i16m2(v732, v727, v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v734 = __riscv_vwredsum_vs_i16m2_i32m1(v733, v712, v723);
    const uint8_t v735 = v713[1];
    int v736 = (int) v735;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v737 = __riscv_vsetvl_e8m1(4);
    size_t v738 = v736 * 4;
    const int8_t* v739 = v11 + v738;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v740 = __riscv_vle8_v_i8m1(v739, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v741 = __riscv_vle8_v_i8m1(v720, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v742 = __riscv_vmv_v_x_u8m1(v719, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v743 = __riscv_vand_vv_u8m1(v742, v10, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v744 = __riscv_vmsne_vx_u8m1_b8(v743, 0, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v745 = __riscv_vneg_v_i8m1(v740, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v746 = __riscv_vmerge_vvm_i8m1(v740, v745, v744, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v747 = __riscv_vwmul_vv_i16m2(v746, v741, v737);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v748 = __riscv_vwredsum_vs_i16m2_i32m1(v747, v734, v737);
    const int8_t* v749 = v714 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v750 = v707 >> 7u;
    uint32_t v751 = v750 & 127u;
    int v752 = (int) v751;
    const uint8_t v753 = tcrv_iq3xxs_ksigns[v752];
    int v754 = (int) v753;
    const int8_t* v755 = v749 + 4;
    const uint8_t v756 = v713[2];
    int v757 = (int) v756;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v758 = __riscv_vsetvl_e8m1(4);
    size_t v759 = v757 * 4;
    const int8_t* v760 = v11 + v759;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v761 = __riscv_vle8_v_i8m1(v760, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v762 = __riscv_vle8_v_i8m1(v749, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v763 = __riscv_vmv_v_x_u8m1(v754, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v764 = __riscv_vand_vv_u8m1(v763, v8, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v765 = __riscv_vmsne_vx_u8m1_b8(v764, 0, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v766 = __riscv_vneg_v_i8m1(v761, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v767 = __riscv_vmerge_vvm_i8m1(v761, v766, v765, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v768 = __riscv_vwmul_vv_i16m2(v767, v762, v758);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v769 = __riscv_vwredsum_vs_i16m2_i32m1(v768, v748, v758);
    const uint8_t v770 = v713[3];
    int v771 = (int) v770;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v772 = __riscv_vsetvl_e8m1(4);
    size_t v773 = v771 * 4;
    const int8_t* v774 = v11 + v773;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v775 = __riscv_vle8_v_i8m1(v774, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v776 = __riscv_vle8_v_i8m1(v755, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v777 = __riscv_vmv_v_x_u8m1(v754, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v778 = __riscv_vand_vv_u8m1(v777, v10, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v779 = __riscv_vmsne_vx_u8m1_b8(v778, 0, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v780 = __riscv_vneg_v_i8m1(v775, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v781 = __riscv_vmerge_vvm_i8m1(v775, v780, v779, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v782 = __riscv_vwmul_vv_i16m2(v781, v776, v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v783 = __riscv_vwredsum_vs_i16m2_i32m1(v782, v769, v772);
    const int8_t* v784 = v749 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v785 = v707 >> 14u;
    uint32_t v786 = v785 & 127u;
    int v787 = (int) v786;
    const uint8_t v788 = tcrv_iq3xxs_ksigns[v787];
    int v789 = (int) v788;
    const int8_t* v790 = v784 + 4;
    const uint8_t v791 = v713[4];
    int v792 = (int) v791;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v793 = __riscv_vsetvl_e8m1(4);
    size_t v794 = v792 * 4;
    const int8_t* v795 = v11 + v794;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v796 = __riscv_vle8_v_i8m1(v795, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v797 = __riscv_vle8_v_i8m1(v784, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v798 = __riscv_vmv_v_x_u8m1(v789, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v799 = __riscv_vand_vv_u8m1(v798, v8, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v800 = __riscv_vmsne_vx_u8m1_b8(v799, 0, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v801 = __riscv_vneg_v_i8m1(v796, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v802 = __riscv_vmerge_vvm_i8m1(v796, v801, v800, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v803 = __riscv_vwmul_vv_i16m2(v802, v797, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v804 = __riscv_vwredsum_vs_i16m2_i32m1(v803, v783, v793);
    const uint8_t v805 = v713[5];
    int v806 = (int) v805;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v807 = __riscv_vsetvl_e8m1(4);
    size_t v808 = v806 * 4;
    const int8_t* v809 = v11 + v808;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v810 = __riscv_vle8_v_i8m1(v809, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v811 = __riscv_vle8_v_i8m1(v790, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v812 = __riscv_vmv_v_x_u8m1(v789, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v813 = __riscv_vand_vv_u8m1(v812, v10, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v814 = __riscv_vmsne_vx_u8m1_b8(v813, 0, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v815 = __riscv_vneg_v_i8m1(v810, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v816 = __riscv_vmerge_vvm_i8m1(v810, v815, v814, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v817 = __riscv_vwmul_vv_i16m2(v816, v811, v807);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v818 = __riscv_vwredsum_vs_i16m2_i32m1(v817, v804, v807);
    const int8_t* v819 = v784 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v820 = v707 >> 21u;
    uint32_t v821 = v820 & 127u;
    int v822 = (int) v821;
    const uint8_t v823 = tcrv_iq3xxs_ksigns[v822];
    int v824 = (int) v823;
    const int8_t* v825 = v819 + 4;
    const uint8_t v826 = v713[6];
    int v827 = (int) v826;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v828 = __riscv_vsetvl_e8m1(4);
    size_t v829 = v827 * 4;
    const int8_t* v830 = v11 + v829;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v831 = __riscv_vle8_v_i8m1(v830, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v832 = __riscv_vle8_v_i8m1(v819, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v833 = __riscv_vmv_v_x_u8m1(v824, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v834 = __riscv_vand_vv_u8m1(v833, v8, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v835 = __riscv_vmsne_vx_u8m1_b8(v834, 0, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v836 = __riscv_vneg_v_i8m1(v831, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v837 = __riscv_vmerge_vvm_i8m1(v831, v836, v835, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v838 = __riscv_vwmul_vv_i16m2(v837, v832, v828);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v839 = __riscv_vwredsum_vs_i16m2_i32m1(v838, v818, v828);
    const uint8_t v840 = v713[7];
    int v841 = (int) v840;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v842 = __riscv_vsetvl_e8m1(4);
    size_t v843 = v841 * 4;
    const int8_t* v844 = v11 + v843;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v845 = __riscv_vle8_v_i8m1(v844, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v846 = __riscv_vle8_v_i8m1(v825, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v847 = __riscv_vmv_v_x_u8m1(v824, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v848 = __riscv_vand_vv_u8m1(v847, v10, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v849 = __riscv_vmsne_vx_u8m1_b8(v848, 0, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v850 = __riscv_vneg_v_i8m1(v845, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v851 = __riscv_vmerge_vvm_i8m1(v845, v850, v849, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v852 = __riscv_vwmul_vv_i16m2(v851, v846, v842);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v853 = __riscv_vwredsum_vs_i16m2_i32m1(v852, v839, v842);
    const int8_t* v854 = v819 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v855 = __riscv_vmv_x_s_i32m1_i32(v853);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v856 = v27;
    int32_t v857 = (int32_t) v711;
    int32_t v858 = v855 * v857;
    int32_t v859 = v856 + v858;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v859;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v860 = v24 + 20;
    const uint8_t v861 = v860[0];
    uint32_t v862 = (uint32_t) v861;
    const uint8_t v863 = v860[1];
    uint32_t v864 = (uint32_t) v863;
    uint32_t v865 = v864 << 8u;
    uint32_t v866 = v862 | v865;
    const uint8_t v867 = v860[2];
    uint32_t v868 = (uint32_t) v867;
    uint32_t v869 = v868 << 16u;
    uint32_t v870 = v866 | v869;
    const uint8_t v871 = v860[3];
    uint32_t v872 = (uint32_t) v871;
    uint32_t v873 = v872 << 24u;
    uint32_t v874 = v870 | v873;
    uint32_t v875 = v874 >> 28u;
    int v876 = (int) v875;
    int v877 = v876 * 2;
    int v878 = v877 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v879 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v880 = v22 + 40;
    const int8_t* v881 = v26 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v882 = v874 >> 0u;
    uint32_t v883 = v882 & 127u;
    int v884 = (int) v883;
    const uint8_t v885 = tcrv_iq3xxs_ksigns[v884];
    int v886 = (int) v885;
    const int8_t* v887 = v881 + 4;
    const uint8_t v888 = v880[0];
    int v889 = (int) v888;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v890 = __riscv_vsetvl_e8m1(4);
    size_t v891 = v889 * 4;
    const int8_t* v892 = v11 + v891;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v893 = __riscv_vle8_v_i8m1(v892, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v894 = __riscv_vle8_v_i8m1(v881, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v895 = __riscv_vmv_v_x_u8m1(v886, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v896 = __riscv_vand_vv_u8m1(v895, v8, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v897 = __riscv_vmsne_vx_u8m1_b8(v896, 0, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v898 = __riscv_vneg_v_i8m1(v893, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v899 = __riscv_vmerge_vvm_i8m1(v893, v898, v897, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v900 = __riscv_vwmul_vv_i16m2(v899, v894, v890);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v901 = __riscv_vwredsum_vs_i16m2_i32m1(v900, v879, v890);
    const uint8_t v902 = v880[1];
    int v903 = (int) v902;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v904 = __riscv_vsetvl_e8m1(4);
    size_t v905 = v903 * 4;
    const int8_t* v906 = v11 + v905;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v907 = __riscv_vle8_v_i8m1(v906, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v908 = __riscv_vle8_v_i8m1(v887, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v909 = __riscv_vmv_v_x_u8m1(v886, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v910 = __riscv_vand_vv_u8m1(v909, v10, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v911 = __riscv_vmsne_vx_u8m1_b8(v910, 0, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v912 = __riscv_vneg_v_i8m1(v907, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v913 = __riscv_vmerge_vvm_i8m1(v907, v912, v911, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v914 = __riscv_vwmul_vv_i16m2(v913, v908, v904);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v915 = __riscv_vwredsum_vs_i16m2_i32m1(v914, v901, v904);
    const int8_t* v916 = v881 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v917 = v874 >> 7u;
    uint32_t v918 = v917 & 127u;
    int v919 = (int) v918;
    const uint8_t v920 = tcrv_iq3xxs_ksigns[v919];
    int v921 = (int) v920;
    const int8_t* v922 = v916 + 4;
    const uint8_t v923 = v880[2];
    int v924 = (int) v923;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v925 = __riscv_vsetvl_e8m1(4);
    size_t v926 = v924 * 4;
    const int8_t* v927 = v11 + v926;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v928 = __riscv_vle8_v_i8m1(v927, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v929 = __riscv_vle8_v_i8m1(v916, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v930 = __riscv_vmv_v_x_u8m1(v921, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v931 = __riscv_vand_vv_u8m1(v930, v8, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v932 = __riscv_vmsne_vx_u8m1_b8(v931, 0, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v933 = __riscv_vneg_v_i8m1(v928, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v934 = __riscv_vmerge_vvm_i8m1(v928, v933, v932, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v935 = __riscv_vwmul_vv_i16m2(v934, v929, v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v936 = __riscv_vwredsum_vs_i16m2_i32m1(v935, v915, v925);
    const uint8_t v937 = v880[3];
    int v938 = (int) v937;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v939 = __riscv_vsetvl_e8m1(4);
    size_t v940 = v938 * 4;
    const int8_t* v941 = v11 + v940;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v942 = __riscv_vle8_v_i8m1(v941, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v943 = __riscv_vle8_v_i8m1(v922, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v944 = __riscv_vmv_v_x_u8m1(v921, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v945 = __riscv_vand_vv_u8m1(v944, v10, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v946 = __riscv_vmsne_vx_u8m1_b8(v945, 0, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v947 = __riscv_vneg_v_i8m1(v942, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v948 = __riscv_vmerge_vvm_i8m1(v942, v947, v946, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v949 = __riscv_vwmul_vv_i16m2(v948, v943, v939);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v950 = __riscv_vwredsum_vs_i16m2_i32m1(v949, v936, v939);
    const int8_t* v951 = v916 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v952 = v874 >> 14u;
    uint32_t v953 = v952 & 127u;
    int v954 = (int) v953;
    const uint8_t v955 = tcrv_iq3xxs_ksigns[v954];
    int v956 = (int) v955;
    const int8_t* v957 = v951 + 4;
    const uint8_t v958 = v880[4];
    int v959 = (int) v958;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v960 = __riscv_vsetvl_e8m1(4);
    size_t v961 = v959 * 4;
    const int8_t* v962 = v11 + v961;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v963 = __riscv_vle8_v_i8m1(v962, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v964 = __riscv_vle8_v_i8m1(v951, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v965 = __riscv_vmv_v_x_u8m1(v956, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v966 = __riscv_vand_vv_u8m1(v965, v8, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v967 = __riscv_vmsne_vx_u8m1_b8(v966, 0, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v968 = __riscv_vneg_v_i8m1(v963, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v969 = __riscv_vmerge_vvm_i8m1(v963, v968, v967, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v970 = __riscv_vwmul_vv_i16m2(v969, v964, v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v971 = __riscv_vwredsum_vs_i16m2_i32m1(v970, v950, v960);
    const uint8_t v972 = v880[5];
    int v973 = (int) v972;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v974 = __riscv_vsetvl_e8m1(4);
    size_t v975 = v973 * 4;
    const int8_t* v976 = v11 + v975;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v977 = __riscv_vle8_v_i8m1(v976, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v978 = __riscv_vle8_v_i8m1(v957, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v979 = __riscv_vmv_v_x_u8m1(v956, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v980 = __riscv_vand_vv_u8m1(v979, v10, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v981 = __riscv_vmsne_vx_u8m1_b8(v980, 0, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v982 = __riscv_vneg_v_i8m1(v977, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v983 = __riscv_vmerge_vvm_i8m1(v977, v982, v981, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v984 = __riscv_vwmul_vv_i16m2(v983, v978, v974);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v985 = __riscv_vwredsum_vs_i16m2_i32m1(v984, v971, v974);
    const int8_t* v986 = v951 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v987 = v874 >> 21u;
    uint32_t v988 = v987 & 127u;
    int v989 = (int) v988;
    const uint8_t v990 = tcrv_iq3xxs_ksigns[v989];
    int v991 = (int) v990;
    const int8_t* v992 = v986 + 4;
    const uint8_t v993 = v880[6];
    int v994 = (int) v993;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v995 = __riscv_vsetvl_e8m1(4);
    size_t v996 = v994 * 4;
    const int8_t* v997 = v11 + v996;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v998 = __riscv_vle8_v_i8m1(v997, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v999 = __riscv_vle8_v_i8m1(v986, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1000 = __riscv_vmv_v_x_u8m1(v991, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1001 = __riscv_vand_vv_u8m1(v1000, v8, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1002 = __riscv_vmsne_vx_u8m1_b8(v1001, 0, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1003 = __riscv_vneg_v_i8m1(v998, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1004 = __riscv_vmerge_vvm_i8m1(v998, v1003, v1002, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1005 = __riscv_vwmul_vv_i16m2(v1004, v999, v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1006 = __riscv_vwredsum_vs_i16m2_i32m1(v1005, v985, v995);
    const uint8_t v1007 = v880[7];
    int v1008 = (int) v1007;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1009 = __riscv_vsetvl_e8m1(4);
    size_t v1010 = v1008 * 4;
    const int8_t* v1011 = v11 + v1010;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1012 = __riscv_vle8_v_i8m1(v1011, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1013 = __riscv_vle8_v_i8m1(v992, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1014 = __riscv_vmv_v_x_u8m1(v991, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1015 = __riscv_vand_vv_u8m1(v1014, v10, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1016 = __riscv_vmsne_vx_u8m1_b8(v1015, 0, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1017 = __riscv_vneg_v_i8m1(v1012, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1018 = __riscv_vmerge_vvm_i8m1(v1012, v1017, v1016, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1019 = __riscv_vwmul_vv_i16m2(v1018, v1013, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1020 = __riscv_vwredsum_vs_i16m2_i32m1(v1019, v1006, v1009);
    const int8_t* v1021 = v986 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1022 = __riscv_vmv_x_s_i32m1_i32(v1020);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1023 = v27;
    int32_t v1024 = (int32_t) v878;
    int32_t v1025 = v1022 * v1024;
    int32_t v1026 = v1023 + v1025;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v1026;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v1027 = v24 + 24;
    const uint8_t v1028 = v1027[0];
    uint32_t v1029 = (uint32_t) v1028;
    const uint8_t v1030 = v1027[1];
    uint32_t v1031 = (uint32_t) v1030;
    uint32_t v1032 = v1031 << 8u;
    uint32_t v1033 = v1029 | v1032;
    const uint8_t v1034 = v1027[2];
    uint32_t v1035 = (uint32_t) v1034;
    uint32_t v1036 = v1035 << 16u;
    uint32_t v1037 = v1033 | v1036;
    const uint8_t v1038 = v1027[3];
    uint32_t v1039 = (uint32_t) v1038;
    uint32_t v1040 = v1039 << 24u;
    uint32_t v1041 = v1037 | v1040;
    uint32_t v1042 = v1041 >> 28u;
    int v1043 = (int) v1042;
    int v1044 = v1043 * 2;
    int v1045 = v1044 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v1046 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v1047 = v22 + 48;
    const int8_t* v1048 = v26 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1049 = v1041 >> 0u;
    uint32_t v1050 = v1049 & 127u;
    int v1051 = (int) v1050;
    const uint8_t v1052 = tcrv_iq3xxs_ksigns[v1051];
    int v1053 = (int) v1052;
    const int8_t* v1054 = v1048 + 4;
    const uint8_t v1055 = v1047[0];
    int v1056 = (int) v1055;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1057 = __riscv_vsetvl_e8m1(4);
    size_t v1058 = v1056 * 4;
    const int8_t* v1059 = v11 + v1058;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1060 = __riscv_vle8_v_i8m1(v1059, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1061 = __riscv_vle8_v_i8m1(v1048, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1062 = __riscv_vmv_v_x_u8m1(v1053, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1063 = __riscv_vand_vv_u8m1(v1062, v8, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1064 = __riscv_vmsne_vx_u8m1_b8(v1063, 0, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1065 = __riscv_vneg_v_i8m1(v1060, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1066 = __riscv_vmerge_vvm_i8m1(v1060, v1065, v1064, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1067 = __riscv_vwmul_vv_i16m2(v1066, v1061, v1057);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1068 = __riscv_vwredsum_vs_i16m2_i32m1(v1067, v1046, v1057);
    const uint8_t v1069 = v1047[1];
    int v1070 = (int) v1069;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1071 = __riscv_vsetvl_e8m1(4);
    size_t v1072 = v1070 * 4;
    const int8_t* v1073 = v11 + v1072;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1074 = __riscv_vle8_v_i8m1(v1073, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1075 = __riscv_vle8_v_i8m1(v1054, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1076 = __riscv_vmv_v_x_u8m1(v1053, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1077 = __riscv_vand_vv_u8m1(v1076, v10, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1078 = __riscv_vmsne_vx_u8m1_b8(v1077, 0, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1079 = __riscv_vneg_v_i8m1(v1074, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1080 = __riscv_vmerge_vvm_i8m1(v1074, v1079, v1078, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1081 = __riscv_vwmul_vv_i16m2(v1080, v1075, v1071);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1082 = __riscv_vwredsum_vs_i16m2_i32m1(v1081, v1068, v1071);
    const int8_t* v1083 = v1048 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1084 = v1041 >> 7u;
    uint32_t v1085 = v1084 & 127u;
    int v1086 = (int) v1085;
    const uint8_t v1087 = tcrv_iq3xxs_ksigns[v1086];
    int v1088 = (int) v1087;
    const int8_t* v1089 = v1083 + 4;
    const uint8_t v1090 = v1047[2];
    int v1091 = (int) v1090;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1092 = __riscv_vsetvl_e8m1(4);
    size_t v1093 = v1091 * 4;
    const int8_t* v1094 = v11 + v1093;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1095 = __riscv_vle8_v_i8m1(v1094, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1096 = __riscv_vle8_v_i8m1(v1083, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1097 = __riscv_vmv_v_x_u8m1(v1088, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1098 = __riscv_vand_vv_u8m1(v1097, v8, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1099 = __riscv_vmsne_vx_u8m1_b8(v1098, 0, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1100 = __riscv_vneg_v_i8m1(v1095, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1101 = __riscv_vmerge_vvm_i8m1(v1095, v1100, v1099, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1102 = __riscv_vwmul_vv_i16m2(v1101, v1096, v1092);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1103 = __riscv_vwredsum_vs_i16m2_i32m1(v1102, v1082, v1092);
    const uint8_t v1104 = v1047[3];
    int v1105 = (int) v1104;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1106 = __riscv_vsetvl_e8m1(4);
    size_t v1107 = v1105 * 4;
    const int8_t* v1108 = v11 + v1107;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1109 = __riscv_vle8_v_i8m1(v1108, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1110 = __riscv_vle8_v_i8m1(v1089, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1111 = __riscv_vmv_v_x_u8m1(v1088, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1112 = __riscv_vand_vv_u8m1(v1111, v10, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1113 = __riscv_vmsne_vx_u8m1_b8(v1112, 0, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1114 = __riscv_vneg_v_i8m1(v1109, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1115 = __riscv_vmerge_vvm_i8m1(v1109, v1114, v1113, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1116 = __riscv_vwmul_vv_i16m2(v1115, v1110, v1106);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1117 = __riscv_vwredsum_vs_i16m2_i32m1(v1116, v1103, v1106);
    const int8_t* v1118 = v1083 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1119 = v1041 >> 14u;
    uint32_t v1120 = v1119 & 127u;
    int v1121 = (int) v1120;
    const uint8_t v1122 = tcrv_iq3xxs_ksigns[v1121];
    int v1123 = (int) v1122;
    const int8_t* v1124 = v1118 + 4;
    const uint8_t v1125 = v1047[4];
    int v1126 = (int) v1125;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1127 = __riscv_vsetvl_e8m1(4);
    size_t v1128 = v1126 * 4;
    const int8_t* v1129 = v11 + v1128;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1130 = __riscv_vle8_v_i8m1(v1129, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1131 = __riscv_vle8_v_i8m1(v1118, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1132 = __riscv_vmv_v_x_u8m1(v1123, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1133 = __riscv_vand_vv_u8m1(v1132, v8, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1134 = __riscv_vmsne_vx_u8m1_b8(v1133, 0, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1135 = __riscv_vneg_v_i8m1(v1130, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1136 = __riscv_vmerge_vvm_i8m1(v1130, v1135, v1134, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1137 = __riscv_vwmul_vv_i16m2(v1136, v1131, v1127);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1138 = __riscv_vwredsum_vs_i16m2_i32m1(v1137, v1117, v1127);
    const uint8_t v1139 = v1047[5];
    int v1140 = (int) v1139;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1141 = __riscv_vsetvl_e8m1(4);
    size_t v1142 = v1140 * 4;
    const int8_t* v1143 = v11 + v1142;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1144 = __riscv_vle8_v_i8m1(v1143, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1145 = __riscv_vle8_v_i8m1(v1124, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1146 = __riscv_vmv_v_x_u8m1(v1123, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1147 = __riscv_vand_vv_u8m1(v1146, v10, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1148 = __riscv_vmsne_vx_u8m1_b8(v1147, 0, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1149 = __riscv_vneg_v_i8m1(v1144, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1150 = __riscv_vmerge_vvm_i8m1(v1144, v1149, v1148, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1151 = __riscv_vwmul_vv_i16m2(v1150, v1145, v1141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1152 = __riscv_vwredsum_vs_i16m2_i32m1(v1151, v1138, v1141);
    const int8_t* v1153 = v1118 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1154 = v1041 >> 21u;
    uint32_t v1155 = v1154 & 127u;
    int v1156 = (int) v1155;
    const uint8_t v1157 = tcrv_iq3xxs_ksigns[v1156];
    int v1158 = (int) v1157;
    const int8_t* v1159 = v1153 + 4;
    const uint8_t v1160 = v1047[6];
    int v1161 = (int) v1160;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1162 = __riscv_vsetvl_e8m1(4);
    size_t v1163 = v1161 * 4;
    const int8_t* v1164 = v11 + v1163;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1165 = __riscv_vle8_v_i8m1(v1164, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1166 = __riscv_vle8_v_i8m1(v1153, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1167 = __riscv_vmv_v_x_u8m1(v1158, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1168 = __riscv_vand_vv_u8m1(v1167, v8, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1169 = __riscv_vmsne_vx_u8m1_b8(v1168, 0, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1170 = __riscv_vneg_v_i8m1(v1165, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1171 = __riscv_vmerge_vvm_i8m1(v1165, v1170, v1169, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1172 = __riscv_vwmul_vv_i16m2(v1171, v1166, v1162);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1173 = __riscv_vwredsum_vs_i16m2_i32m1(v1172, v1152, v1162);
    const uint8_t v1174 = v1047[7];
    int v1175 = (int) v1174;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1176 = __riscv_vsetvl_e8m1(4);
    size_t v1177 = v1175 * 4;
    const int8_t* v1178 = v11 + v1177;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1179 = __riscv_vle8_v_i8m1(v1178, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1180 = __riscv_vle8_v_i8m1(v1159, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1181 = __riscv_vmv_v_x_u8m1(v1158, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1182 = __riscv_vand_vv_u8m1(v1181, v10, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1183 = __riscv_vmsne_vx_u8m1_b8(v1182, 0, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1184 = __riscv_vneg_v_i8m1(v1179, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1185 = __riscv_vmerge_vvm_i8m1(v1179, v1184, v1183, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1186 = __riscv_vwmul_vv_i16m2(v1185, v1180, v1176);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1187 = __riscv_vwredsum_vs_i16m2_i32m1(v1186, v1173, v1176);
    const int8_t* v1188 = v1153 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1189 = __riscv_vmv_x_s_i32m1_i32(v1187);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1190 = v27;
    int32_t v1191 = (int32_t) v1045;
    int32_t v1192 = v1189 * v1191;
    int32_t v1193 = v1190 + v1192;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v1193;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v1194 = v24 + 28;
    const uint8_t v1195 = v1194[0];
    uint32_t v1196 = (uint32_t) v1195;
    const uint8_t v1197 = v1194[1];
    uint32_t v1198 = (uint32_t) v1197;
    uint32_t v1199 = v1198 << 8u;
    uint32_t v1200 = v1196 | v1199;
    const uint8_t v1201 = v1194[2];
    uint32_t v1202 = (uint32_t) v1201;
    uint32_t v1203 = v1202 << 16u;
    uint32_t v1204 = v1200 | v1203;
    const uint8_t v1205 = v1194[3];
    uint32_t v1206 = (uint32_t) v1205;
    uint32_t v1207 = v1206 << 24u;
    uint32_t v1208 = v1204 | v1207;
    uint32_t v1209 = v1208 >> 28u;
    int v1210 = (int) v1209;
    int v1211 = v1210 * 2;
    int v1212 = v1211 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v1213 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v1214 = v22 + 56;
    const int8_t* v1215 = v26 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1216 = v1208 >> 0u;
    uint32_t v1217 = v1216 & 127u;
    int v1218 = (int) v1217;
    const uint8_t v1219 = tcrv_iq3xxs_ksigns[v1218];
    int v1220 = (int) v1219;
    const int8_t* v1221 = v1215 + 4;
    const uint8_t v1222 = v1214[0];
    int v1223 = (int) v1222;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1224 = __riscv_vsetvl_e8m1(4);
    size_t v1225 = v1223 * 4;
    const int8_t* v1226 = v11 + v1225;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1227 = __riscv_vle8_v_i8m1(v1226, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1228 = __riscv_vle8_v_i8m1(v1215, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1229 = __riscv_vmv_v_x_u8m1(v1220, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1230 = __riscv_vand_vv_u8m1(v1229, v8, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1231 = __riscv_vmsne_vx_u8m1_b8(v1230, 0, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1232 = __riscv_vneg_v_i8m1(v1227, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1233 = __riscv_vmerge_vvm_i8m1(v1227, v1232, v1231, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1234 = __riscv_vwmul_vv_i16m2(v1233, v1228, v1224);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1235 = __riscv_vwredsum_vs_i16m2_i32m1(v1234, v1213, v1224);
    const uint8_t v1236 = v1214[1];
    int v1237 = (int) v1236;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1238 = __riscv_vsetvl_e8m1(4);
    size_t v1239 = v1237 * 4;
    const int8_t* v1240 = v11 + v1239;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1241 = __riscv_vle8_v_i8m1(v1240, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1242 = __riscv_vle8_v_i8m1(v1221, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1243 = __riscv_vmv_v_x_u8m1(v1220, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1244 = __riscv_vand_vv_u8m1(v1243, v10, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1245 = __riscv_vmsne_vx_u8m1_b8(v1244, 0, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1246 = __riscv_vneg_v_i8m1(v1241, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1247 = __riscv_vmerge_vvm_i8m1(v1241, v1246, v1245, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1248 = __riscv_vwmul_vv_i16m2(v1247, v1242, v1238);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1249 = __riscv_vwredsum_vs_i16m2_i32m1(v1248, v1235, v1238);
    const int8_t* v1250 = v1215 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1251 = v1208 >> 7u;
    uint32_t v1252 = v1251 & 127u;
    int v1253 = (int) v1252;
    const uint8_t v1254 = tcrv_iq3xxs_ksigns[v1253];
    int v1255 = (int) v1254;
    const int8_t* v1256 = v1250 + 4;
    const uint8_t v1257 = v1214[2];
    int v1258 = (int) v1257;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1259 = __riscv_vsetvl_e8m1(4);
    size_t v1260 = v1258 * 4;
    const int8_t* v1261 = v11 + v1260;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1262 = __riscv_vle8_v_i8m1(v1261, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1263 = __riscv_vle8_v_i8m1(v1250, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1264 = __riscv_vmv_v_x_u8m1(v1255, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1265 = __riscv_vand_vv_u8m1(v1264, v8, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1266 = __riscv_vmsne_vx_u8m1_b8(v1265, 0, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1267 = __riscv_vneg_v_i8m1(v1262, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1268 = __riscv_vmerge_vvm_i8m1(v1262, v1267, v1266, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1269 = __riscv_vwmul_vv_i16m2(v1268, v1263, v1259);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1270 = __riscv_vwredsum_vs_i16m2_i32m1(v1269, v1249, v1259);
    const uint8_t v1271 = v1214[3];
    int v1272 = (int) v1271;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1273 = __riscv_vsetvl_e8m1(4);
    size_t v1274 = v1272 * 4;
    const int8_t* v1275 = v11 + v1274;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1276 = __riscv_vle8_v_i8m1(v1275, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1277 = __riscv_vle8_v_i8m1(v1256, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1278 = __riscv_vmv_v_x_u8m1(v1255, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1279 = __riscv_vand_vv_u8m1(v1278, v10, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1280 = __riscv_vmsne_vx_u8m1_b8(v1279, 0, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1281 = __riscv_vneg_v_i8m1(v1276, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1282 = __riscv_vmerge_vvm_i8m1(v1276, v1281, v1280, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1283 = __riscv_vwmul_vv_i16m2(v1282, v1277, v1273);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1284 = __riscv_vwredsum_vs_i16m2_i32m1(v1283, v1270, v1273);
    const int8_t* v1285 = v1250 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1286 = v1208 >> 14u;
    uint32_t v1287 = v1286 & 127u;
    int v1288 = (int) v1287;
    const uint8_t v1289 = tcrv_iq3xxs_ksigns[v1288];
    int v1290 = (int) v1289;
    const int8_t* v1291 = v1285 + 4;
    const uint8_t v1292 = v1214[4];
    int v1293 = (int) v1292;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1294 = __riscv_vsetvl_e8m1(4);
    size_t v1295 = v1293 * 4;
    const int8_t* v1296 = v11 + v1295;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1297 = __riscv_vle8_v_i8m1(v1296, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1298 = __riscv_vle8_v_i8m1(v1285, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1299 = __riscv_vmv_v_x_u8m1(v1290, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1300 = __riscv_vand_vv_u8m1(v1299, v8, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1301 = __riscv_vmsne_vx_u8m1_b8(v1300, 0, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1302 = __riscv_vneg_v_i8m1(v1297, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1303 = __riscv_vmerge_vvm_i8m1(v1297, v1302, v1301, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1304 = __riscv_vwmul_vv_i16m2(v1303, v1298, v1294);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1305 = __riscv_vwredsum_vs_i16m2_i32m1(v1304, v1284, v1294);
    const uint8_t v1306 = v1214[5];
    int v1307 = (int) v1306;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1308 = __riscv_vsetvl_e8m1(4);
    size_t v1309 = v1307 * 4;
    const int8_t* v1310 = v11 + v1309;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1311 = __riscv_vle8_v_i8m1(v1310, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1312 = __riscv_vle8_v_i8m1(v1291, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1313 = __riscv_vmv_v_x_u8m1(v1290, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1314 = __riscv_vand_vv_u8m1(v1313, v10, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1315 = __riscv_vmsne_vx_u8m1_b8(v1314, 0, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1316 = __riscv_vneg_v_i8m1(v1311, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1317 = __riscv_vmerge_vvm_i8m1(v1311, v1316, v1315, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1318 = __riscv_vwmul_vv_i16m2(v1317, v1312, v1308);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1319 = __riscv_vwredsum_vs_i16m2_i32m1(v1318, v1305, v1308);
    const int8_t* v1320 = v1285 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1321 = v1208 >> 21u;
    uint32_t v1322 = v1321 & 127u;
    int v1323 = (int) v1322;
    const uint8_t v1324 = tcrv_iq3xxs_ksigns[v1323];
    int v1325 = (int) v1324;
    const int8_t* v1326 = v1320 + 4;
    const uint8_t v1327 = v1214[6];
    int v1328 = (int) v1327;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1329 = __riscv_vsetvl_e8m1(4);
    size_t v1330 = v1328 * 4;
    const int8_t* v1331 = v11 + v1330;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1332 = __riscv_vle8_v_i8m1(v1331, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1333 = __riscv_vle8_v_i8m1(v1320, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1334 = __riscv_vmv_v_x_u8m1(v1325, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1335 = __riscv_vand_vv_u8m1(v1334, v8, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1336 = __riscv_vmsne_vx_u8m1_b8(v1335, 0, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1337 = __riscv_vneg_v_i8m1(v1332, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1338 = __riscv_vmerge_vvm_i8m1(v1332, v1337, v1336, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1339 = __riscv_vwmul_vv_i16m2(v1338, v1333, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1340 = __riscv_vwredsum_vs_i16m2_i32m1(v1339, v1319, v1329);
    const uint8_t v1341 = v1214[7];
    int v1342 = (int) v1341;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1343 = __riscv_vsetvl_e8m1(4);
    size_t v1344 = v1342 * 4;
    const int8_t* v1345 = v11 + v1344;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1346 = __riscv_vle8_v_i8m1(v1345, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1347 = __riscv_vle8_v_i8m1(v1326, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1348 = __riscv_vmv_v_x_u8m1(v1325, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1349 = __riscv_vand_vv_u8m1(v1348, v10, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1350 = __riscv_vmsne_vx_u8m1_b8(v1349, 0, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1351 = __riscv_vneg_v_i8m1(v1346, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1352 = __riscv_vmerge_vvm_i8m1(v1346, v1351, v1350, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1353 = __riscv_vwmul_vv_i16m2(v1352, v1347, v1343);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1354 = __riscv_vwredsum_vs_i16m2_i32m1(v1353, v1340, v1343);
    const int8_t* v1355 = v1320 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1356 = __riscv_vmv_x_s_i32m1_i32(v1354);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1357 = v27;
    int32_t v1358 = (int32_t) v1212;
    int32_t v1359 = v1356 * v1358;
    int32_t v1360 = v1357 + v1359;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v27 = v1360;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v1361 = v27;
    float v1362 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v1362 + v20 * (float) v1361;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v1363 = v6;
  float v1364 = 0.25f * v1363;
  v2[0] = v1364;
  return;
}


