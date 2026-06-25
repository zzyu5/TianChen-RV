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
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq3xxs_kmask, 8);
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_i32_view
  const int32_t* v9 = (const int32_t*) tcrv_iq3xxs_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v7; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 98;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v16 = (const float*) v14;
    const float v17 = v16[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v18 = v15 * v17;
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t* v21 = v12 + 66;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t* v23 = v14 + 4;
    const int8_t* v24 = (const int8_t*) v23;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v26 = v22[0];
    uint32_t v27 = (uint32_t) v26;
    const uint8_t v28 = v22[1];
    uint32_t v29 = (uint32_t) v28;
    uint32_t v30 = v29 << 8u;
    uint32_t v31 = v27 | v30;
    const uint8_t v32 = v22[2];
    uint32_t v33 = (uint32_t) v32;
    uint32_t v34 = v33 << 16u;
    uint32_t v35 = v31 | v34;
    const uint8_t v36 = v22[3];
    uint32_t v37 = (uint32_t) v36;
    uint32_t v38 = v37 << 24u;
    uint32_t v39 = v35 | v38;
    uint32_t v40 = v39 >> 28u;
    int v41 = (int) v40;
    int v42 = v41 * 2;
    int v43 = v42 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v44 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v45 = v39 >> 0u;
    uint32_t v46 = v45 & 127u;
    int v47 = (int) v46;
    const uint8_t v48 = tcrv_iq3xxs_ksigns[v47];
    int v49 = (int) v48;
    const uint8_t v50 = v20[0];
    int v51 = (int) v50;
    const uint8_t v52 = v20[1];
    int v53 = (int) v52;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v54[2];
    int v55 = v51 << 2;
    uint16_t v56 = (uint16_t) v55;
    v54[0] = v56;
    int v57 = v53 << 2;
    uint16_t v58 = (uint16_t) v57;
    v54[1] = v58;
    uint16_t* v59 = &v54[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v60 = __riscv_vle16_v_u16mf2(v59, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v61 = __riscv_vluxei16_v_i32m1(v9, v60, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v62 = __riscv_vreinterpret_v_i32m1_i8m1(v61);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v63 = __riscv_vle8_v_i8m1(v24, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v64 = __riscv_vmv_v_x_u8m1(v49, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v65 = __riscv_vand_vv_u8m1(v64, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v66 = __riscv_vmsne_vx_u8m1_b8(v65, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v67 = __riscv_vneg_v_i8m1(v62, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v68 = __riscv_vmerge_vvm_i8m1(v62, v67, v66, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v69 = __riscv_vwmul_vv_i16m2(v68, v63, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v70 = __riscv_vwredsum_vs_i16m2_i32m1(v69, v44, 8);
    const int8_t* v71 = v24 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v72 = v39 >> 7u;
    uint32_t v73 = v72 & 127u;
    int v74 = (int) v73;
    const uint8_t v75 = tcrv_iq3xxs_ksigns[v74];
    int v76 = (int) v75;
    const uint8_t v77 = v20[2];
    int v78 = (int) v77;
    const uint8_t v79 = v20[3];
    int v80 = (int) v79;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v81[2];
    int v82 = v78 << 2;
    uint16_t v83 = (uint16_t) v82;
    v81[0] = v83;
    int v84 = v80 << 2;
    uint16_t v85 = (uint16_t) v84;
    v81[1] = v85;
    uint16_t* v86 = &v81[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v87 = __riscv_vle16_v_u16mf2(v86, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v88 = __riscv_vluxei16_v_i32m1(v9, v87, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v89 = __riscv_vreinterpret_v_i32m1_i8m1(v88);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v90 = __riscv_vle8_v_i8m1(v71, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v91 = __riscv_vmv_v_x_u8m1(v76, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v92 = __riscv_vand_vv_u8m1(v91, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v93 = __riscv_vmsne_vx_u8m1_b8(v92, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v94 = __riscv_vneg_v_i8m1(v89, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v95 = __riscv_vmerge_vvm_i8m1(v89, v94, v93, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v96 = __riscv_vwmul_vv_i16m2(v95, v90, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v97 = __riscv_vwredsum_vs_i16m2_i32m1(v96, v70, 8);
    const int8_t* v98 = v71 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v99 = v39 >> 14u;
    uint32_t v100 = v99 & 127u;
    int v101 = (int) v100;
    const uint8_t v102 = tcrv_iq3xxs_ksigns[v101];
    int v103 = (int) v102;
    const uint8_t v104 = v20[4];
    int v105 = (int) v104;
    const uint8_t v106 = v20[5];
    int v107 = (int) v106;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v108[2];
    int v109 = v105 << 2;
    uint16_t v110 = (uint16_t) v109;
    v108[0] = v110;
    int v111 = v107 << 2;
    uint16_t v112 = (uint16_t) v111;
    v108[1] = v112;
    uint16_t* v113 = &v108[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v114 = __riscv_vle16_v_u16mf2(v113, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v115 = __riscv_vluxei16_v_i32m1(v9, v114, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v116 = __riscv_vreinterpret_v_i32m1_i8m1(v115);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v117 = __riscv_vle8_v_i8m1(v98, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v118 = __riscv_vmv_v_x_u8m1(v103, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v119 = __riscv_vand_vv_u8m1(v118, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v120 = __riscv_vmsne_vx_u8m1_b8(v119, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v121 = __riscv_vneg_v_i8m1(v116, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v122 = __riscv_vmerge_vvm_i8m1(v116, v121, v120, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v123 = __riscv_vwmul_vv_i16m2(v122, v117, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v124 = __riscv_vwredsum_vs_i16m2_i32m1(v123, v97, 8);
    const int8_t* v125 = v98 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v126 = v39 >> 21u;
    uint32_t v127 = v126 & 127u;
    int v128 = (int) v127;
    const uint8_t v129 = tcrv_iq3xxs_ksigns[v128];
    int v130 = (int) v129;
    const uint8_t v131 = v20[6];
    int v132 = (int) v131;
    const uint8_t v133 = v20[7];
    int v134 = (int) v133;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v135[2];
    int v136 = v132 << 2;
    uint16_t v137 = (uint16_t) v136;
    v135[0] = v137;
    int v138 = v134 << 2;
    uint16_t v139 = (uint16_t) v138;
    v135[1] = v139;
    uint16_t* v140 = &v135[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v141 = __riscv_vle16_v_u16mf2(v140, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v142 = __riscv_vluxei16_v_i32m1(v9, v141, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v143 = __riscv_vreinterpret_v_i32m1_i8m1(v142);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v144 = __riscv_vle8_v_i8m1(v125, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v145 = __riscv_vmv_v_x_u8m1(v130, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v146 = __riscv_vand_vv_u8m1(v145, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v147 = __riscv_vmsne_vx_u8m1_b8(v146, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v148 = __riscv_vneg_v_i8m1(v143, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v149 = __riscv_vmerge_vvm_i8m1(v143, v148, v147, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v150 = __riscv_vwmul_vv_i16m2(v149, v144, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v151 = __riscv_vwredsum_vs_i16m2_i32m1(v150, v124, 8);
    const int8_t* v152 = v125 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v153 = __riscv_vmv_x_s_i32m1_i32(v151);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v154 = v25;
    int32_t v155 = (int32_t) v43;
    int32_t v156 = v153 * v155;
    int32_t v157 = v154 + v156;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v157;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v158 = v22 + 4;
    const uint8_t v159 = v158[0];
    uint32_t v160 = (uint32_t) v159;
    const uint8_t v161 = v158[1];
    uint32_t v162 = (uint32_t) v161;
    uint32_t v163 = v162 << 8u;
    uint32_t v164 = v160 | v163;
    const uint8_t v165 = v158[2];
    uint32_t v166 = (uint32_t) v165;
    uint32_t v167 = v166 << 16u;
    uint32_t v168 = v164 | v167;
    const uint8_t v169 = v158[3];
    uint32_t v170 = (uint32_t) v169;
    uint32_t v171 = v170 << 24u;
    uint32_t v172 = v168 | v171;
    uint32_t v173 = v172 >> 28u;
    int v174 = (int) v173;
    int v175 = v174 * 2;
    int v176 = v175 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v177 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v178 = v20 + 8;
    const int8_t* v179 = v24 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v180 = v172 >> 0u;
    uint32_t v181 = v180 & 127u;
    int v182 = (int) v181;
    const uint8_t v183 = tcrv_iq3xxs_ksigns[v182];
    int v184 = (int) v183;
    const uint8_t v185 = v178[0];
    int v186 = (int) v185;
    const uint8_t v187 = v178[1];
    int v188 = (int) v187;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v189[2];
    int v190 = v186 << 2;
    uint16_t v191 = (uint16_t) v190;
    v189[0] = v191;
    int v192 = v188 << 2;
    uint16_t v193 = (uint16_t) v192;
    v189[1] = v193;
    uint16_t* v194 = &v189[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v195 = __riscv_vle16_v_u16mf2(v194, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v196 = __riscv_vluxei16_v_i32m1(v9, v195, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v197 = __riscv_vreinterpret_v_i32m1_i8m1(v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v198 = __riscv_vle8_v_i8m1(v179, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v199 = __riscv_vmv_v_x_u8m1(v184, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v200 = __riscv_vand_vv_u8m1(v199, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v201 = __riscv_vmsne_vx_u8m1_b8(v200, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v202 = __riscv_vneg_v_i8m1(v197, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v203 = __riscv_vmerge_vvm_i8m1(v197, v202, v201, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v204 = __riscv_vwmul_vv_i16m2(v203, v198, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v205 = __riscv_vwredsum_vs_i16m2_i32m1(v204, v177, 8);
    const int8_t* v206 = v179 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v207 = v172 >> 7u;
    uint32_t v208 = v207 & 127u;
    int v209 = (int) v208;
    const uint8_t v210 = tcrv_iq3xxs_ksigns[v209];
    int v211 = (int) v210;
    const uint8_t v212 = v178[2];
    int v213 = (int) v212;
    const uint8_t v214 = v178[3];
    int v215 = (int) v214;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v216[2];
    int v217 = v213 << 2;
    uint16_t v218 = (uint16_t) v217;
    v216[0] = v218;
    int v219 = v215 << 2;
    uint16_t v220 = (uint16_t) v219;
    v216[1] = v220;
    uint16_t* v221 = &v216[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v222 = __riscv_vle16_v_u16mf2(v221, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v223 = __riscv_vluxei16_v_i32m1(v9, v222, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v224 = __riscv_vreinterpret_v_i32m1_i8m1(v223);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v225 = __riscv_vle8_v_i8m1(v206, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v226 = __riscv_vmv_v_x_u8m1(v211, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v227 = __riscv_vand_vv_u8m1(v226, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v228 = __riscv_vmsne_vx_u8m1_b8(v227, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v229 = __riscv_vneg_v_i8m1(v224, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v230 = __riscv_vmerge_vvm_i8m1(v224, v229, v228, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v231 = __riscv_vwmul_vv_i16m2(v230, v225, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v232 = __riscv_vwredsum_vs_i16m2_i32m1(v231, v205, 8);
    const int8_t* v233 = v206 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v234 = v172 >> 14u;
    uint32_t v235 = v234 & 127u;
    int v236 = (int) v235;
    const uint8_t v237 = tcrv_iq3xxs_ksigns[v236];
    int v238 = (int) v237;
    const uint8_t v239 = v178[4];
    int v240 = (int) v239;
    const uint8_t v241 = v178[5];
    int v242 = (int) v241;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v243[2];
    int v244 = v240 << 2;
    uint16_t v245 = (uint16_t) v244;
    v243[0] = v245;
    int v246 = v242 << 2;
    uint16_t v247 = (uint16_t) v246;
    v243[1] = v247;
    uint16_t* v248 = &v243[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v249 = __riscv_vle16_v_u16mf2(v248, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v250 = __riscv_vluxei16_v_i32m1(v9, v249, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v251 = __riscv_vreinterpret_v_i32m1_i8m1(v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v252 = __riscv_vle8_v_i8m1(v233, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v253 = __riscv_vmv_v_x_u8m1(v238, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v254 = __riscv_vand_vv_u8m1(v253, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v255 = __riscv_vmsne_vx_u8m1_b8(v254, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v256 = __riscv_vneg_v_i8m1(v251, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v257 = __riscv_vmerge_vvm_i8m1(v251, v256, v255, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v258 = __riscv_vwmul_vv_i16m2(v257, v252, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v259 = __riscv_vwredsum_vs_i16m2_i32m1(v258, v232, 8);
    const int8_t* v260 = v233 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v261 = v172 >> 21u;
    uint32_t v262 = v261 & 127u;
    int v263 = (int) v262;
    const uint8_t v264 = tcrv_iq3xxs_ksigns[v263];
    int v265 = (int) v264;
    const uint8_t v266 = v178[6];
    int v267 = (int) v266;
    const uint8_t v268 = v178[7];
    int v269 = (int) v268;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v270[2];
    int v271 = v267 << 2;
    uint16_t v272 = (uint16_t) v271;
    v270[0] = v272;
    int v273 = v269 << 2;
    uint16_t v274 = (uint16_t) v273;
    v270[1] = v274;
    uint16_t* v275 = &v270[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v276 = __riscv_vle16_v_u16mf2(v275, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v277 = __riscv_vluxei16_v_i32m1(v9, v276, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v278 = __riscv_vreinterpret_v_i32m1_i8m1(v277);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v279 = __riscv_vle8_v_i8m1(v260, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v280 = __riscv_vmv_v_x_u8m1(v265, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v281 = __riscv_vand_vv_u8m1(v280, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v282 = __riscv_vmsne_vx_u8m1_b8(v281, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v283 = __riscv_vneg_v_i8m1(v278, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v284 = __riscv_vmerge_vvm_i8m1(v278, v283, v282, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v285 = __riscv_vwmul_vv_i16m2(v284, v279, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v286 = __riscv_vwredsum_vs_i16m2_i32m1(v285, v259, 8);
    const int8_t* v287 = v260 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v288 = __riscv_vmv_x_s_i32m1_i32(v286);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v289 = v25;
    int32_t v290 = (int32_t) v176;
    int32_t v291 = v288 * v290;
    int32_t v292 = v289 + v291;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v292;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v293 = v22 + 8;
    const uint8_t v294 = v293[0];
    uint32_t v295 = (uint32_t) v294;
    const uint8_t v296 = v293[1];
    uint32_t v297 = (uint32_t) v296;
    uint32_t v298 = v297 << 8u;
    uint32_t v299 = v295 | v298;
    const uint8_t v300 = v293[2];
    uint32_t v301 = (uint32_t) v300;
    uint32_t v302 = v301 << 16u;
    uint32_t v303 = v299 | v302;
    const uint8_t v304 = v293[3];
    uint32_t v305 = (uint32_t) v304;
    uint32_t v306 = v305 << 24u;
    uint32_t v307 = v303 | v306;
    uint32_t v308 = v307 >> 28u;
    int v309 = (int) v308;
    int v310 = v309 * 2;
    int v311 = v310 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v312 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v313 = v20 + 16;
    const int8_t* v314 = v24 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v315 = v307 >> 0u;
    uint32_t v316 = v315 & 127u;
    int v317 = (int) v316;
    const uint8_t v318 = tcrv_iq3xxs_ksigns[v317];
    int v319 = (int) v318;
    const uint8_t v320 = v313[0];
    int v321 = (int) v320;
    const uint8_t v322 = v313[1];
    int v323 = (int) v322;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v324[2];
    int v325 = v321 << 2;
    uint16_t v326 = (uint16_t) v325;
    v324[0] = v326;
    int v327 = v323 << 2;
    uint16_t v328 = (uint16_t) v327;
    v324[1] = v328;
    uint16_t* v329 = &v324[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v330 = __riscv_vle16_v_u16mf2(v329, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v331 = __riscv_vluxei16_v_i32m1(v9, v330, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v332 = __riscv_vreinterpret_v_i32m1_i8m1(v331);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v333 = __riscv_vle8_v_i8m1(v314, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v334 = __riscv_vmv_v_x_u8m1(v319, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v335 = __riscv_vand_vv_u8m1(v334, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v336 = __riscv_vmsne_vx_u8m1_b8(v335, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v337 = __riscv_vneg_v_i8m1(v332, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v338 = __riscv_vmerge_vvm_i8m1(v332, v337, v336, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v339 = __riscv_vwmul_vv_i16m2(v338, v333, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v340 = __riscv_vwredsum_vs_i16m2_i32m1(v339, v312, 8);
    const int8_t* v341 = v314 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v342 = v307 >> 7u;
    uint32_t v343 = v342 & 127u;
    int v344 = (int) v343;
    const uint8_t v345 = tcrv_iq3xxs_ksigns[v344];
    int v346 = (int) v345;
    const uint8_t v347 = v313[2];
    int v348 = (int) v347;
    const uint8_t v349 = v313[3];
    int v350 = (int) v349;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v351[2];
    int v352 = v348 << 2;
    uint16_t v353 = (uint16_t) v352;
    v351[0] = v353;
    int v354 = v350 << 2;
    uint16_t v355 = (uint16_t) v354;
    v351[1] = v355;
    uint16_t* v356 = &v351[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v357 = __riscv_vle16_v_u16mf2(v356, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v358 = __riscv_vluxei16_v_i32m1(v9, v357, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v359 = __riscv_vreinterpret_v_i32m1_i8m1(v358);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v360 = __riscv_vle8_v_i8m1(v341, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v361 = __riscv_vmv_v_x_u8m1(v346, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v362 = __riscv_vand_vv_u8m1(v361, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v363 = __riscv_vmsne_vx_u8m1_b8(v362, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v364 = __riscv_vneg_v_i8m1(v359, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v365 = __riscv_vmerge_vvm_i8m1(v359, v364, v363, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v366 = __riscv_vwmul_vv_i16m2(v365, v360, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v367 = __riscv_vwredsum_vs_i16m2_i32m1(v366, v340, 8);
    const int8_t* v368 = v341 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v369 = v307 >> 14u;
    uint32_t v370 = v369 & 127u;
    int v371 = (int) v370;
    const uint8_t v372 = tcrv_iq3xxs_ksigns[v371];
    int v373 = (int) v372;
    const uint8_t v374 = v313[4];
    int v375 = (int) v374;
    const uint8_t v376 = v313[5];
    int v377 = (int) v376;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v378[2];
    int v379 = v375 << 2;
    uint16_t v380 = (uint16_t) v379;
    v378[0] = v380;
    int v381 = v377 << 2;
    uint16_t v382 = (uint16_t) v381;
    v378[1] = v382;
    uint16_t* v383 = &v378[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v384 = __riscv_vle16_v_u16mf2(v383, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v385 = __riscv_vluxei16_v_i32m1(v9, v384, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v386 = __riscv_vreinterpret_v_i32m1_i8m1(v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v387 = __riscv_vle8_v_i8m1(v368, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v388 = __riscv_vmv_v_x_u8m1(v373, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v389 = __riscv_vand_vv_u8m1(v388, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v390 = __riscv_vmsne_vx_u8m1_b8(v389, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v391 = __riscv_vneg_v_i8m1(v386, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v392 = __riscv_vmerge_vvm_i8m1(v386, v391, v390, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v393 = __riscv_vwmul_vv_i16m2(v392, v387, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v394 = __riscv_vwredsum_vs_i16m2_i32m1(v393, v367, 8);
    const int8_t* v395 = v368 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v396 = v307 >> 21u;
    uint32_t v397 = v396 & 127u;
    int v398 = (int) v397;
    const uint8_t v399 = tcrv_iq3xxs_ksigns[v398];
    int v400 = (int) v399;
    const uint8_t v401 = v313[6];
    int v402 = (int) v401;
    const uint8_t v403 = v313[7];
    int v404 = (int) v403;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v405[2];
    int v406 = v402 << 2;
    uint16_t v407 = (uint16_t) v406;
    v405[0] = v407;
    int v408 = v404 << 2;
    uint16_t v409 = (uint16_t) v408;
    v405[1] = v409;
    uint16_t* v410 = &v405[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v411 = __riscv_vle16_v_u16mf2(v410, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v412 = __riscv_vluxei16_v_i32m1(v9, v411, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v413 = __riscv_vreinterpret_v_i32m1_i8m1(v412);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v414 = __riscv_vle8_v_i8m1(v395, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v415 = __riscv_vmv_v_x_u8m1(v400, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v416 = __riscv_vand_vv_u8m1(v415, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v417 = __riscv_vmsne_vx_u8m1_b8(v416, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v418 = __riscv_vneg_v_i8m1(v413, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v419 = __riscv_vmerge_vvm_i8m1(v413, v418, v417, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v420 = __riscv_vwmul_vv_i16m2(v419, v414, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v421 = __riscv_vwredsum_vs_i16m2_i32m1(v420, v394, 8);
    const int8_t* v422 = v395 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v423 = __riscv_vmv_x_s_i32m1_i32(v421);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v424 = v25;
    int32_t v425 = (int32_t) v311;
    int32_t v426 = v423 * v425;
    int32_t v427 = v424 + v426;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v427;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v428 = v22 + 12;
    const uint8_t v429 = v428[0];
    uint32_t v430 = (uint32_t) v429;
    const uint8_t v431 = v428[1];
    uint32_t v432 = (uint32_t) v431;
    uint32_t v433 = v432 << 8u;
    uint32_t v434 = v430 | v433;
    const uint8_t v435 = v428[2];
    uint32_t v436 = (uint32_t) v435;
    uint32_t v437 = v436 << 16u;
    uint32_t v438 = v434 | v437;
    const uint8_t v439 = v428[3];
    uint32_t v440 = (uint32_t) v439;
    uint32_t v441 = v440 << 24u;
    uint32_t v442 = v438 | v441;
    uint32_t v443 = v442 >> 28u;
    int v444 = (int) v443;
    int v445 = v444 * 2;
    int v446 = v445 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v447 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v448 = v20 + 24;
    const int8_t* v449 = v24 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v450 = v442 >> 0u;
    uint32_t v451 = v450 & 127u;
    int v452 = (int) v451;
    const uint8_t v453 = tcrv_iq3xxs_ksigns[v452];
    int v454 = (int) v453;
    const uint8_t v455 = v448[0];
    int v456 = (int) v455;
    const uint8_t v457 = v448[1];
    int v458 = (int) v457;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v459[2];
    int v460 = v456 << 2;
    uint16_t v461 = (uint16_t) v460;
    v459[0] = v461;
    int v462 = v458 << 2;
    uint16_t v463 = (uint16_t) v462;
    v459[1] = v463;
    uint16_t* v464 = &v459[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v465 = __riscv_vle16_v_u16mf2(v464, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v466 = __riscv_vluxei16_v_i32m1(v9, v465, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v467 = __riscv_vreinterpret_v_i32m1_i8m1(v466);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v468 = __riscv_vle8_v_i8m1(v449, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v469 = __riscv_vmv_v_x_u8m1(v454, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v470 = __riscv_vand_vv_u8m1(v469, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v471 = __riscv_vmsne_vx_u8m1_b8(v470, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v472 = __riscv_vneg_v_i8m1(v467, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v473 = __riscv_vmerge_vvm_i8m1(v467, v472, v471, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v474 = __riscv_vwmul_vv_i16m2(v473, v468, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v475 = __riscv_vwredsum_vs_i16m2_i32m1(v474, v447, 8);
    const int8_t* v476 = v449 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v477 = v442 >> 7u;
    uint32_t v478 = v477 & 127u;
    int v479 = (int) v478;
    const uint8_t v480 = tcrv_iq3xxs_ksigns[v479];
    int v481 = (int) v480;
    const uint8_t v482 = v448[2];
    int v483 = (int) v482;
    const uint8_t v484 = v448[3];
    int v485 = (int) v484;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v486[2];
    int v487 = v483 << 2;
    uint16_t v488 = (uint16_t) v487;
    v486[0] = v488;
    int v489 = v485 << 2;
    uint16_t v490 = (uint16_t) v489;
    v486[1] = v490;
    uint16_t* v491 = &v486[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v492 = __riscv_vle16_v_u16mf2(v491, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v493 = __riscv_vluxei16_v_i32m1(v9, v492, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v494 = __riscv_vreinterpret_v_i32m1_i8m1(v493);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v495 = __riscv_vle8_v_i8m1(v476, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v496 = __riscv_vmv_v_x_u8m1(v481, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v497 = __riscv_vand_vv_u8m1(v496, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v498 = __riscv_vmsne_vx_u8m1_b8(v497, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v499 = __riscv_vneg_v_i8m1(v494, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v500 = __riscv_vmerge_vvm_i8m1(v494, v499, v498, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v501 = __riscv_vwmul_vv_i16m2(v500, v495, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v502 = __riscv_vwredsum_vs_i16m2_i32m1(v501, v475, 8);
    const int8_t* v503 = v476 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v504 = v442 >> 14u;
    uint32_t v505 = v504 & 127u;
    int v506 = (int) v505;
    const uint8_t v507 = tcrv_iq3xxs_ksigns[v506];
    int v508 = (int) v507;
    const uint8_t v509 = v448[4];
    int v510 = (int) v509;
    const uint8_t v511 = v448[5];
    int v512 = (int) v511;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v513[2];
    int v514 = v510 << 2;
    uint16_t v515 = (uint16_t) v514;
    v513[0] = v515;
    int v516 = v512 << 2;
    uint16_t v517 = (uint16_t) v516;
    v513[1] = v517;
    uint16_t* v518 = &v513[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v519 = __riscv_vle16_v_u16mf2(v518, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v520 = __riscv_vluxei16_v_i32m1(v9, v519, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v521 = __riscv_vreinterpret_v_i32m1_i8m1(v520);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v522 = __riscv_vle8_v_i8m1(v503, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v523 = __riscv_vmv_v_x_u8m1(v508, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v524 = __riscv_vand_vv_u8m1(v523, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v525 = __riscv_vmsne_vx_u8m1_b8(v524, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v526 = __riscv_vneg_v_i8m1(v521, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v527 = __riscv_vmerge_vvm_i8m1(v521, v526, v525, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v528 = __riscv_vwmul_vv_i16m2(v527, v522, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v529 = __riscv_vwredsum_vs_i16m2_i32m1(v528, v502, 8);
    const int8_t* v530 = v503 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v531 = v442 >> 21u;
    uint32_t v532 = v531 & 127u;
    int v533 = (int) v532;
    const uint8_t v534 = tcrv_iq3xxs_ksigns[v533];
    int v535 = (int) v534;
    const uint8_t v536 = v448[6];
    int v537 = (int) v536;
    const uint8_t v538 = v448[7];
    int v539 = (int) v538;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v540[2];
    int v541 = v537 << 2;
    uint16_t v542 = (uint16_t) v541;
    v540[0] = v542;
    int v543 = v539 << 2;
    uint16_t v544 = (uint16_t) v543;
    v540[1] = v544;
    uint16_t* v545 = &v540[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v546 = __riscv_vle16_v_u16mf2(v545, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v547 = __riscv_vluxei16_v_i32m1(v9, v546, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v548 = __riscv_vreinterpret_v_i32m1_i8m1(v547);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v549 = __riscv_vle8_v_i8m1(v530, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v550 = __riscv_vmv_v_x_u8m1(v535, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v551 = __riscv_vand_vv_u8m1(v550, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v552 = __riscv_vmsne_vx_u8m1_b8(v551, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v553 = __riscv_vneg_v_i8m1(v548, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v554 = __riscv_vmerge_vvm_i8m1(v548, v553, v552, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v555 = __riscv_vwmul_vv_i16m2(v554, v549, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v556 = __riscv_vwredsum_vs_i16m2_i32m1(v555, v529, 8);
    const int8_t* v557 = v530 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v558 = __riscv_vmv_x_s_i32m1_i32(v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v559 = v25;
    int32_t v560 = (int32_t) v446;
    int32_t v561 = v558 * v560;
    int32_t v562 = v559 + v561;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v562;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v563 = v22 + 16;
    const uint8_t v564 = v563[0];
    uint32_t v565 = (uint32_t) v564;
    const uint8_t v566 = v563[1];
    uint32_t v567 = (uint32_t) v566;
    uint32_t v568 = v567 << 8u;
    uint32_t v569 = v565 | v568;
    const uint8_t v570 = v563[2];
    uint32_t v571 = (uint32_t) v570;
    uint32_t v572 = v571 << 16u;
    uint32_t v573 = v569 | v572;
    const uint8_t v574 = v563[3];
    uint32_t v575 = (uint32_t) v574;
    uint32_t v576 = v575 << 24u;
    uint32_t v577 = v573 | v576;
    uint32_t v578 = v577 >> 28u;
    int v579 = (int) v578;
    int v580 = v579 * 2;
    int v581 = v580 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v582 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v583 = v20 + 32;
    const int8_t* v584 = v24 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v585 = v577 >> 0u;
    uint32_t v586 = v585 & 127u;
    int v587 = (int) v586;
    const uint8_t v588 = tcrv_iq3xxs_ksigns[v587];
    int v589 = (int) v588;
    const uint8_t v590 = v583[0];
    int v591 = (int) v590;
    const uint8_t v592 = v583[1];
    int v593 = (int) v592;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v594[2];
    int v595 = v591 << 2;
    uint16_t v596 = (uint16_t) v595;
    v594[0] = v596;
    int v597 = v593 << 2;
    uint16_t v598 = (uint16_t) v597;
    v594[1] = v598;
    uint16_t* v599 = &v594[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v600 = __riscv_vle16_v_u16mf2(v599, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v601 = __riscv_vluxei16_v_i32m1(v9, v600, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v602 = __riscv_vreinterpret_v_i32m1_i8m1(v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v603 = __riscv_vle8_v_i8m1(v584, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v604 = __riscv_vmv_v_x_u8m1(v589, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v605 = __riscv_vand_vv_u8m1(v604, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v606 = __riscv_vmsne_vx_u8m1_b8(v605, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v607 = __riscv_vneg_v_i8m1(v602, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v608 = __riscv_vmerge_vvm_i8m1(v602, v607, v606, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v609 = __riscv_vwmul_vv_i16m2(v608, v603, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v610 = __riscv_vwredsum_vs_i16m2_i32m1(v609, v582, 8);
    const int8_t* v611 = v584 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v612 = v577 >> 7u;
    uint32_t v613 = v612 & 127u;
    int v614 = (int) v613;
    const uint8_t v615 = tcrv_iq3xxs_ksigns[v614];
    int v616 = (int) v615;
    const uint8_t v617 = v583[2];
    int v618 = (int) v617;
    const uint8_t v619 = v583[3];
    int v620 = (int) v619;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v621[2];
    int v622 = v618 << 2;
    uint16_t v623 = (uint16_t) v622;
    v621[0] = v623;
    int v624 = v620 << 2;
    uint16_t v625 = (uint16_t) v624;
    v621[1] = v625;
    uint16_t* v626 = &v621[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v627 = __riscv_vle16_v_u16mf2(v626, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v628 = __riscv_vluxei16_v_i32m1(v9, v627, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v629 = __riscv_vreinterpret_v_i32m1_i8m1(v628);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v630 = __riscv_vle8_v_i8m1(v611, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v631 = __riscv_vmv_v_x_u8m1(v616, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v632 = __riscv_vand_vv_u8m1(v631, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v633 = __riscv_vmsne_vx_u8m1_b8(v632, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v634 = __riscv_vneg_v_i8m1(v629, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v635 = __riscv_vmerge_vvm_i8m1(v629, v634, v633, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v636 = __riscv_vwmul_vv_i16m2(v635, v630, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v637 = __riscv_vwredsum_vs_i16m2_i32m1(v636, v610, 8);
    const int8_t* v638 = v611 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v639 = v577 >> 14u;
    uint32_t v640 = v639 & 127u;
    int v641 = (int) v640;
    const uint8_t v642 = tcrv_iq3xxs_ksigns[v641];
    int v643 = (int) v642;
    const uint8_t v644 = v583[4];
    int v645 = (int) v644;
    const uint8_t v646 = v583[5];
    int v647 = (int) v646;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v648[2];
    int v649 = v645 << 2;
    uint16_t v650 = (uint16_t) v649;
    v648[0] = v650;
    int v651 = v647 << 2;
    uint16_t v652 = (uint16_t) v651;
    v648[1] = v652;
    uint16_t* v653 = &v648[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v654 = __riscv_vle16_v_u16mf2(v653, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v655 = __riscv_vluxei16_v_i32m1(v9, v654, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v656 = __riscv_vreinterpret_v_i32m1_i8m1(v655);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v657 = __riscv_vle8_v_i8m1(v638, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v658 = __riscv_vmv_v_x_u8m1(v643, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v659 = __riscv_vand_vv_u8m1(v658, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v660 = __riscv_vmsne_vx_u8m1_b8(v659, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v661 = __riscv_vneg_v_i8m1(v656, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v662 = __riscv_vmerge_vvm_i8m1(v656, v661, v660, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v663 = __riscv_vwmul_vv_i16m2(v662, v657, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v664 = __riscv_vwredsum_vs_i16m2_i32m1(v663, v637, 8);
    const int8_t* v665 = v638 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v666 = v577 >> 21u;
    uint32_t v667 = v666 & 127u;
    int v668 = (int) v667;
    const uint8_t v669 = tcrv_iq3xxs_ksigns[v668];
    int v670 = (int) v669;
    const uint8_t v671 = v583[6];
    int v672 = (int) v671;
    const uint8_t v673 = v583[7];
    int v674 = (int) v673;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v675[2];
    int v676 = v672 << 2;
    uint16_t v677 = (uint16_t) v676;
    v675[0] = v677;
    int v678 = v674 << 2;
    uint16_t v679 = (uint16_t) v678;
    v675[1] = v679;
    uint16_t* v680 = &v675[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v681 = __riscv_vle16_v_u16mf2(v680, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v682 = __riscv_vluxei16_v_i32m1(v9, v681, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v683 = __riscv_vreinterpret_v_i32m1_i8m1(v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v684 = __riscv_vle8_v_i8m1(v665, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v685 = __riscv_vmv_v_x_u8m1(v670, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v686 = __riscv_vand_vv_u8m1(v685, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v687 = __riscv_vmsne_vx_u8m1_b8(v686, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v688 = __riscv_vneg_v_i8m1(v683, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v689 = __riscv_vmerge_vvm_i8m1(v683, v688, v687, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v690 = __riscv_vwmul_vv_i16m2(v689, v684, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v691 = __riscv_vwredsum_vs_i16m2_i32m1(v690, v664, 8);
    const int8_t* v692 = v665 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v693 = __riscv_vmv_x_s_i32m1_i32(v691);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v694 = v25;
    int32_t v695 = (int32_t) v581;
    int32_t v696 = v693 * v695;
    int32_t v697 = v694 + v696;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v697;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v698 = v22 + 20;
    const uint8_t v699 = v698[0];
    uint32_t v700 = (uint32_t) v699;
    const uint8_t v701 = v698[1];
    uint32_t v702 = (uint32_t) v701;
    uint32_t v703 = v702 << 8u;
    uint32_t v704 = v700 | v703;
    const uint8_t v705 = v698[2];
    uint32_t v706 = (uint32_t) v705;
    uint32_t v707 = v706 << 16u;
    uint32_t v708 = v704 | v707;
    const uint8_t v709 = v698[3];
    uint32_t v710 = (uint32_t) v709;
    uint32_t v711 = v710 << 24u;
    uint32_t v712 = v708 | v711;
    uint32_t v713 = v712 >> 28u;
    int v714 = (int) v713;
    int v715 = v714 * 2;
    int v716 = v715 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v717 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v718 = v20 + 40;
    const int8_t* v719 = v24 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v720 = v712 >> 0u;
    uint32_t v721 = v720 & 127u;
    int v722 = (int) v721;
    const uint8_t v723 = tcrv_iq3xxs_ksigns[v722];
    int v724 = (int) v723;
    const uint8_t v725 = v718[0];
    int v726 = (int) v725;
    const uint8_t v727 = v718[1];
    int v728 = (int) v727;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v729[2];
    int v730 = v726 << 2;
    uint16_t v731 = (uint16_t) v730;
    v729[0] = v731;
    int v732 = v728 << 2;
    uint16_t v733 = (uint16_t) v732;
    v729[1] = v733;
    uint16_t* v734 = &v729[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v735 = __riscv_vle16_v_u16mf2(v734, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v736 = __riscv_vluxei16_v_i32m1(v9, v735, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v737 = __riscv_vreinterpret_v_i32m1_i8m1(v736);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v738 = __riscv_vle8_v_i8m1(v719, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v739 = __riscv_vmv_v_x_u8m1(v724, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v740 = __riscv_vand_vv_u8m1(v739, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v741 = __riscv_vmsne_vx_u8m1_b8(v740, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v742 = __riscv_vneg_v_i8m1(v737, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v743 = __riscv_vmerge_vvm_i8m1(v737, v742, v741, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v744 = __riscv_vwmul_vv_i16m2(v743, v738, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v745 = __riscv_vwredsum_vs_i16m2_i32m1(v744, v717, 8);
    const int8_t* v746 = v719 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v747 = v712 >> 7u;
    uint32_t v748 = v747 & 127u;
    int v749 = (int) v748;
    const uint8_t v750 = tcrv_iq3xxs_ksigns[v749];
    int v751 = (int) v750;
    const uint8_t v752 = v718[2];
    int v753 = (int) v752;
    const uint8_t v754 = v718[3];
    int v755 = (int) v754;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v756[2];
    int v757 = v753 << 2;
    uint16_t v758 = (uint16_t) v757;
    v756[0] = v758;
    int v759 = v755 << 2;
    uint16_t v760 = (uint16_t) v759;
    v756[1] = v760;
    uint16_t* v761 = &v756[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v762 = __riscv_vle16_v_u16mf2(v761, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v763 = __riscv_vluxei16_v_i32m1(v9, v762, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v764 = __riscv_vreinterpret_v_i32m1_i8m1(v763);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v765 = __riscv_vle8_v_i8m1(v746, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v766 = __riscv_vmv_v_x_u8m1(v751, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v767 = __riscv_vand_vv_u8m1(v766, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v768 = __riscv_vmsne_vx_u8m1_b8(v767, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v769 = __riscv_vneg_v_i8m1(v764, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v770 = __riscv_vmerge_vvm_i8m1(v764, v769, v768, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v771 = __riscv_vwmul_vv_i16m2(v770, v765, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v772 = __riscv_vwredsum_vs_i16m2_i32m1(v771, v745, 8);
    const int8_t* v773 = v746 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v774 = v712 >> 14u;
    uint32_t v775 = v774 & 127u;
    int v776 = (int) v775;
    const uint8_t v777 = tcrv_iq3xxs_ksigns[v776];
    int v778 = (int) v777;
    const uint8_t v779 = v718[4];
    int v780 = (int) v779;
    const uint8_t v781 = v718[5];
    int v782 = (int) v781;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v783[2];
    int v784 = v780 << 2;
    uint16_t v785 = (uint16_t) v784;
    v783[0] = v785;
    int v786 = v782 << 2;
    uint16_t v787 = (uint16_t) v786;
    v783[1] = v787;
    uint16_t* v788 = &v783[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v789 = __riscv_vle16_v_u16mf2(v788, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v790 = __riscv_vluxei16_v_i32m1(v9, v789, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v791 = __riscv_vreinterpret_v_i32m1_i8m1(v790);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v792 = __riscv_vle8_v_i8m1(v773, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v793 = __riscv_vmv_v_x_u8m1(v778, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v794 = __riscv_vand_vv_u8m1(v793, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v795 = __riscv_vmsne_vx_u8m1_b8(v794, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v796 = __riscv_vneg_v_i8m1(v791, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v797 = __riscv_vmerge_vvm_i8m1(v791, v796, v795, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v798 = __riscv_vwmul_vv_i16m2(v797, v792, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v799 = __riscv_vwredsum_vs_i16m2_i32m1(v798, v772, 8);
    const int8_t* v800 = v773 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v801 = v712 >> 21u;
    uint32_t v802 = v801 & 127u;
    int v803 = (int) v802;
    const uint8_t v804 = tcrv_iq3xxs_ksigns[v803];
    int v805 = (int) v804;
    const uint8_t v806 = v718[6];
    int v807 = (int) v806;
    const uint8_t v808 = v718[7];
    int v809 = (int) v808;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v810[2];
    int v811 = v807 << 2;
    uint16_t v812 = (uint16_t) v811;
    v810[0] = v812;
    int v813 = v809 << 2;
    uint16_t v814 = (uint16_t) v813;
    v810[1] = v814;
    uint16_t* v815 = &v810[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v816 = __riscv_vle16_v_u16mf2(v815, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v817 = __riscv_vluxei16_v_i32m1(v9, v816, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v818 = __riscv_vreinterpret_v_i32m1_i8m1(v817);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v819 = __riscv_vle8_v_i8m1(v800, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v820 = __riscv_vmv_v_x_u8m1(v805, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v821 = __riscv_vand_vv_u8m1(v820, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v822 = __riscv_vmsne_vx_u8m1_b8(v821, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v823 = __riscv_vneg_v_i8m1(v818, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v824 = __riscv_vmerge_vvm_i8m1(v818, v823, v822, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v825 = __riscv_vwmul_vv_i16m2(v824, v819, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v826 = __riscv_vwredsum_vs_i16m2_i32m1(v825, v799, 8);
    const int8_t* v827 = v800 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v828 = __riscv_vmv_x_s_i32m1_i32(v826);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v829 = v25;
    int32_t v830 = (int32_t) v716;
    int32_t v831 = v828 * v830;
    int32_t v832 = v829 + v831;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v832;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v833 = v22 + 24;
    const uint8_t v834 = v833[0];
    uint32_t v835 = (uint32_t) v834;
    const uint8_t v836 = v833[1];
    uint32_t v837 = (uint32_t) v836;
    uint32_t v838 = v837 << 8u;
    uint32_t v839 = v835 | v838;
    const uint8_t v840 = v833[2];
    uint32_t v841 = (uint32_t) v840;
    uint32_t v842 = v841 << 16u;
    uint32_t v843 = v839 | v842;
    const uint8_t v844 = v833[3];
    uint32_t v845 = (uint32_t) v844;
    uint32_t v846 = v845 << 24u;
    uint32_t v847 = v843 | v846;
    uint32_t v848 = v847 >> 28u;
    int v849 = (int) v848;
    int v850 = v849 * 2;
    int v851 = v850 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v852 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v853 = v20 + 48;
    const int8_t* v854 = v24 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v855 = v847 >> 0u;
    uint32_t v856 = v855 & 127u;
    int v857 = (int) v856;
    const uint8_t v858 = tcrv_iq3xxs_ksigns[v857];
    int v859 = (int) v858;
    const uint8_t v860 = v853[0];
    int v861 = (int) v860;
    const uint8_t v862 = v853[1];
    int v863 = (int) v862;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v864[2];
    int v865 = v861 << 2;
    uint16_t v866 = (uint16_t) v865;
    v864[0] = v866;
    int v867 = v863 << 2;
    uint16_t v868 = (uint16_t) v867;
    v864[1] = v868;
    uint16_t* v869 = &v864[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v870 = __riscv_vle16_v_u16mf2(v869, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v871 = __riscv_vluxei16_v_i32m1(v9, v870, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v872 = __riscv_vreinterpret_v_i32m1_i8m1(v871);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v873 = __riscv_vle8_v_i8m1(v854, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v874 = __riscv_vmv_v_x_u8m1(v859, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v875 = __riscv_vand_vv_u8m1(v874, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v876 = __riscv_vmsne_vx_u8m1_b8(v875, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v877 = __riscv_vneg_v_i8m1(v872, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v878 = __riscv_vmerge_vvm_i8m1(v872, v877, v876, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v879 = __riscv_vwmul_vv_i16m2(v878, v873, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v880 = __riscv_vwredsum_vs_i16m2_i32m1(v879, v852, 8);
    const int8_t* v881 = v854 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v882 = v847 >> 7u;
    uint32_t v883 = v882 & 127u;
    int v884 = (int) v883;
    const uint8_t v885 = tcrv_iq3xxs_ksigns[v884];
    int v886 = (int) v885;
    const uint8_t v887 = v853[2];
    int v888 = (int) v887;
    const uint8_t v889 = v853[3];
    int v890 = (int) v889;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v891[2];
    int v892 = v888 << 2;
    uint16_t v893 = (uint16_t) v892;
    v891[0] = v893;
    int v894 = v890 << 2;
    uint16_t v895 = (uint16_t) v894;
    v891[1] = v895;
    uint16_t* v896 = &v891[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v897 = __riscv_vle16_v_u16mf2(v896, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v898 = __riscv_vluxei16_v_i32m1(v9, v897, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v899 = __riscv_vreinterpret_v_i32m1_i8m1(v898);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v900 = __riscv_vle8_v_i8m1(v881, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v901 = __riscv_vmv_v_x_u8m1(v886, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v902 = __riscv_vand_vv_u8m1(v901, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v903 = __riscv_vmsne_vx_u8m1_b8(v902, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v904 = __riscv_vneg_v_i8m1(v899, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v905 = __riscv_vmerge_vvm_i8m1(v899, v904, v903, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v906 = __riscv_vwmul_vv_i16m2(v905, v900, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v907 = __riscv_vwredsum_vs_i16m2_i32m1(v906, v880, 8);
    const int8_t* v908 = v881 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v909 = v847 >> 14u;
    uint32_t v910 = v909 & 127u;
    int v911 = (int) v910;
    const uint8_t v912 = tcrv_iq3xxs_ksigns[v911];
    int v913 = (int) v912;
    const uint8_t v914 = v853[4];
    int v915 = (int) v914;
    const uint8_t v916 = v853[5];
    int v917 = (int) v916;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v918[2];
    int v919 = v915 << 2;
    uint16_t v920 = (uint16_t) v919;
    v918[0] = v920;
    int v921 = v917 << 2;
    uint16_t v922 = (uint16_t) v921;
    v918[1] = v922;
    uint16_t* v923 = &v918[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v924 = __riscv_vle16_v_u16mf2(v923, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v925 = __riscv_vluxei16_v_i32m1(v9, v924, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v926 = __riscv_vreinterpret_v_i32m1_i8m1(v925);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v927 = __riscv_vle8_v_i8m1(v908, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v928 = __riscv_vmv_v_x_u8m1(v913, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v929 = __riscv_vand_vv_u8m1(v928, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v930 = __riscv_vmsne_vx_u8m1_b8(v929, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v931 = __riscv_vneg_v_i8m1(v926, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v932 = __riscv_vmerge_vvm_i8m1(v926, v931, v930, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v933 = __riscv_vwmul_vv_i16m2(v932, v927, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v934 = __riscv_vwredsum_vs_i16m2_i32m1(v933, v907, 8);
    const int8_t* v935 = v908 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v936 = v847 >> 21u;
    uint32_t v937 = v936 & 127u;
    int v938 = (int) v937;
    const uint8_t v939 = tcrv_iq3xxs_ksigns[v938];
    int v940 = (int) v939;
    const uint8_t v941 = v853[6];
    int v942 = (int) v941;
    const uint8_t v943 = v853[7];
    int v944 = (int) v943;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v945[2];
    int v946 = v942 << 2;
    uint16_t v947 = (uint16_t) v946;
    v945[0] = v947;
    int v948 = v944 << 2;
    uint16_t v949 = (uint16_t) v948;
    v945[1] = v949;
    uint16_t* v950 = &v945[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v951 = __riscv_vle16_v_u16mf2(v950, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v952 = __riscv_vluxei16_v_i32m1(v9, v951, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v953 = __riscv_vreinterpret_v_i32m1_i8m1(v952);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v954 = __riscv_vle8_v_i8m1(v935, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v955 = __riscv_vmv_v_x_u8m1(v940, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v956 = __riscv_vand_vv_u8m1(v955, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v957 = __riscv_vmsne_vx_u8m1_b8(v956, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v958 = __riscv_vneg_v_i8m1(v953, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v959 = __riscv_vmerge_vvm_i8m1(v953, v958, v957, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v960 = __riscv_vwmul_vv_i16m2(v959, v954, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v961 = __riscv_vwredsum_vs_i16m2_i32m1(v960, v934, 8);
    const int8_t* v962 = v935 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v963 = __riscv_vmv_x_s_i32m1_i32(v961);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v964 = v25;
    int32_t v965 = (int32_t) v851;
    int32_t v966 = v963 * v965;
    int32_t v967 = v964 + v966;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v967;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v968 = v22 + 28;
    const uint8_t v969 = v968[0];
    uint32_t v970 = (uint32_t) v969;
    const uint8_t v971 = v968[1];
    uint32_t v972 = (uint32_t) v971;
    uint32_t v973 = v972 << 8u;
    uint32_t v974 = v970 | v973;
    const uint8_t v975 = v968[2];
    uint32_t v976 = (uint32_t) v975;
    uint32_t v977 = v976 << 16u;
    uint32_t v978 = v974 | v977;
    const uint8_t v979 = v968[3];
    uint32_t v980 = (uint32_t) v979;
    uint32_t v981 = v980 << 24u;
    uint32_t v982 = v978 | v981;
    uint32_t v983 = v982 >> 28u;
    int v984 = (int) v983;
    int v985 = v984 * 2;
    int v986 = v985 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v987 = __riscv_vmv_v_x_i32m1(0, 1);
    const uint8_t* v988 = v20 + 56;
    const int8_t* v989 = v24 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v990 = v982 >> 0u;
    uint32_t v991 = v990 & 127u;
    int v992 = (int) v991;
    const uint8_t v993 = tcrv_iq3xxs_ksigns[v992];
    int v994 = (int) v993;
    const uint8_t v995 = v988[0];
    int v996 = (int) v995;
    const uint8_t v997 = v988[1];
    int v998 = (int) v997;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v999[2];
    int v1000 = v996 << 2;
    uint16_t v1001 = (uint16_t) v1000;
    v999[0] = v1001;
    int v1002 = v998 << 2;
    uint16_t v1003 = (uint16_t) v1002;
    v999[1] = v1003;
    uint16_t* v1004 = &v999[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1005 = __riscv_vle16_v_u16mf2(v1004, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1006 = __riscv_vluxei16_v_i32m1(v9, v1005, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1007 = __riscv_vreinterpret_v_i32m1_i8m1(v1006);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1008 = __riscv_vle8_v_i8m1(v989, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1009 = __riscv_vmv_v_x_u8m1(v994, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1010 = __riscv_vand_vv_u8m1(v1009, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1011 = __riscv_vmsne_vx_u8m1_b8(v1010, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1012 = __riscv_vneg_v_i8m1(v1007, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1013 = __riscv_vmerge_vvm_i8m1(v1007, v1012, v1011, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1014 = __riscv_vwmul_vv_i16m2(v1013, v1008, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1015 = __riscv_vwredsum_vs_i16m2_i32m1(v1014, v987, 8);
    const int8_t* v1016 = v989 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1017 = v982 >> 7u;
    uint32_t v1018 = v1017 & 127u;
    int v1019 = (int) v1018;
    const uint8_t v1020 = tcrv_iq3xxs_ksigns[v1019];
    int v1021 = (int) v1020;
    const uint8_t v1022 = v988[2];
    int v1023 = (int) v1022;
    const uint8_t v1024 = v988[3];
    int v1025 = (int) v1024;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1026[2];
    int v1027 = v1023 << 2;
    uint16_t v1028 = (uint16_t) v1027;
    v1026[0] = v1028;
    int v1029 = v1025 << 2;
    uint16_t v1030 = (uint16_t) v1029;
    v1026[1] = v1030;
    uint16_t* v1031 = &v1026[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1032 = __riscv_vle16_v_u16mf2(v1031, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1033 = __riscv_vluxei16_v_i32m1(v9, v1032, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1034 = __riscv_vreinterpret_v_i32m1_i8m1(v1033);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1035 = __riscv_vle8_v_i8m1(v1016, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1036 = __riscv_vmv_v_x_u8m1(v1021, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1037 = __riscv_vand_vv_u8m1(v1036, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1038 = __riscv_vmsne_vx_u8m1_b8(v1037, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1039 = __riscv_vneg_v_i8m1(v1034, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1040 = __riscv_vmerge_vvm_i8m1(v1034, v1039, v1038, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1041 = __riscv_vwmul_vv_i16m2(v1040, v1035, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1042 = __riscv_vwredsum_vs_i16m2_i32m1(v1041, v1015, 8);
    const int8_t* v1043 = v1016 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1044 = v982 >> 14u;
    uint32_t v1045 = v1044 & 127u;
    int v1046 = (int) v1045;
    const uint8_t v1047 = tcrv_iq3xxs_ksigns[v1046];
    int v1048 = (int) v1047;
    const uint8_t v1049 = v988[4];
    int v1050 = (int) v1049;
    const uint8_t v1051 = v988[5];
    int v1052 = (int) v1051;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1053[2];
    int v1054 = v1050 << 2;
    uint16_t v1055 = (uint16_t) v1054;
    v1053[0] = v1055;
    int v1056 = v1052 << 2;
    uint16_t v1057 = (uint16_t) v1056;
    v1053[1] = v1057;
    uint16_t* v1058 = &v1053[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1059 = __riscv_vle16_v_u16mf2(v1058, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1060 = __riscv_vluxei16_v_i32m1(v9, v1059, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1061 = __riscv_vreinterpret_v_i32m1_i8m1(v1060);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1062 = __riscv_vle8_v_i8m1(v1043, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1063 = __riscv_vmv_v_x_u8m1(v1048, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1064 = __riscv_vand_vv_u8m1(v1063, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1065 = __riscv_vmsne_vx_u8m1_b8(v1064, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1066 = __riscv_vneg_v_i8m1(v1061, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1067 = __riscv_vmerge_vvm_i8m1(v1061, v1066, v1065, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1068 = __riscv_vwmul_vv_i16m2(v1067, v1062, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1069 = __riscv_vwredsum_vs_i16m2_i32m1(v1068, v1042, 8);
    const int8_t* v1070 = v1043 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    uint32_t v1071 = v982 >> 21u;
    uint32_t v1072 = v1071 & 127u;
    int v1073 = (int) v1072;
    const uint8_t v1074 = tcrv_iq3xxs_ksigns[v1073];
    int v1075 = (int) v1074;
    const uint8_t v1076 = v988[6];
    int v1077 = (int) v1076;
    const uint8_t v1078 = v988[7];
    int v1079 = (int) v1078;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1080[2];
    int v1081 = v1077 << 2;
    uint16_t v1082 = (uint16_t) v1081;
    v1080[0] = v1082;
    int v1083 = v1079 << 2;
    uint16_t v1084 = (uint16_t) v1083;
    v1080[1] = v1084;
    uint16_t* v1085 = &v1080[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1086 = __riscv_vle16_v_u16mf2(v1085, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1087 = __riscv_vluxei16_v_i32m1(v9, v1086, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1088 = __riscv_vreinterpret_v_i32m1_i8m1(v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1089 = __riscv_vle8_v_i8m1(v1070, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1090 = __riscv_vmv_v_x_u8m1(v1075, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1091 = __riscv_vand_vv_u8m1(v1090, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1092 = __riscv_vmsne_vx_u8m1_b8(v1091, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1093 = __riscv_vneg_v_i8m1(v1088, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1094 = __riscv_vmerge_vvm_i8m1(v1088, v1093, v1092, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1095 = __riscv_vwmul_vv_i16m2(v1094, v1089, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1096 = __riscv_vwredsum_vs_i16m2_i32m1(v1095, v1069, 8);
    const int8_t* v1097 = v1070 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1098 = __riscv_vmv_x_s_i32m1_i32(v1096);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1099 = v25;
    int32_t v1100 = (int32_t) v986;
    int32_t v1101 = v1098 * v1100;
    int32_t v1102 = v1099 + v1101;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v1102;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v1103 = v25;
    float v1104 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v1104 + v18 * (float) v1103;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v1105 = v6;
  float v1106 = 0.25f * v1105;
  v2[0] = v1106;
  return;
}


