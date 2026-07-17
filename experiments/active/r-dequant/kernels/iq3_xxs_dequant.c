#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq3_xxs_kernel_dequant_iq3_xxs(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const uint32_t weft_iq3xxs_grid[256] = {0x04040404U, 0x04040414U, 0x04040424U, 0x04040c0cU, 0x04040c1cU, 0x04040c3eU, 0x04041404U, 0x04041414U, 0x04041c0cU, 0x04042414U, 0x04043e1cU, 0x04043e2cU, 0x040c040cU, 0x040c041cU, 0x040c0c04U, 0x040c0c14U, 0x040c140cU, 0x040c142cU, 0x040c1c04U, 0x040c1c14U, 0x040c240cU, 0x040c2c24U, 0x040c3e04U, 0x04140404U, 0x04140414U, 0x04140424U, 0x04140c0cU, 0x04141404U, 0x04141414U, 0x04141c0cU, 0x04141c1cU, 0x04141c3eU, 0x04142c0cU, 0x04142c3eU, 0x04143e2cU, 0x041c040cU, 0x041c043eU, 0x041c0c04U, 0x041c0c14U, 0x041c142cU, 0x041c3e04U, 0x04240c1cU, 0x04241c3eU, 0x04242424U, 0x04242c3eU, 0x04243e1cU, 0x04243e2cU, 0x042c040cU, 0x042c043eU, 0x042c1c14U, 0x042c2c14U, 0x04341c2cU, 0x04343424U, 0x043e0c04U, 0x043e0c24U, 0x043e0c34U, 0x043e241cU, 0x043e340cU, 0x0c04040cU, 0x0c04041cU, 0x0c040c04U, 0x0c040c14U, 0x0c04140cU, 0x0c04141cU, 0x0c041c04U, 0x0c041c14U, 0x0c041c24U, 0x0c04243eU, 0x0c042c04U, 0x0c0c0404U, 0x0c0c0414U, 0x0c0c0c0cU, 0x0c0c1404U, 0x0c0c1414U, 0x0c14040cU, 0x0c14041cU, 0x0c140c04U, 0x0c140c14U, 0x0c14140cU, 0x0c141c04U, 0x0c143e14U, 0x0c1c0404U, 0x0c1c0414U, 0x0c1c1404U, 0x0c1c1c0cU, 0x0c1c2434U, 0x0c1c3434U, 0x0c24040cU, 0x0c24042cU, 0x0c242c04U, 0x0c2c1404U, 0x0c2c1424U, 0x0c2c2434U, 0x0c2c3e0cU, 0x0c34042cU, 0x0c3e1414U, 0x0c3e2404U, 0x14040404U, 0x14040414U, 0x14040c0cU, 0x14040c1cU, 0x14041404U, 0x14041414U, 0x14041434U, 0x14041c0cU, 0x14042414U, 0x140c040cU, 0x140c041cU, 0x140c042cU, 0x140c0c04U, 0x140c0c14U, 0x140c140cU, 0x140c1c04U, 0x140c341cU, 0x140c343eU, 0x140c3e04U, 0x14140404U, 0x14140414U, 0x14140c0cU, 0x14140c3eU, 0x14141404U, 0x14141414U, 0x14141c3eU, 0x14142404U, 0x14142c2cU, 0x141c040cU, 0x141c0c04U, 0x141c0c24U, 0x141c3e04U, 0x141c3e24U, 0x14241c2cU, 0x14242c1cU, 0x142c041cU, 0x142c143eU, 0x142c240cU, 0x142c3e24U, 0x143e040cU, 0x143e041cU, 0x143e0c34U, 0x143e242cU, 0x1c04040cU, 0x1c040c04U, 0x1c040c14U, 0x1c04140cU, 0x1c04141cU, 0x1c042c04U, 0x1c04342cU, 0x1c043e14U, 0x1c0c0404U, 0x1c0c0414U, 0x1c0c1404U, 0x1c0c1c0cU, 0x1c0c2424U, 0x1c0c2434U, 0x1c14040cU, 0x1c14041cU, 0x1c140c04U, 0x1c14142cU, 0x1c142c14U, 0x1c143e14U, 0x1c1c0c0cU, 0x1c1c1c1cU, 0x1c241c04U, 0x1c24243eU, 0x1c243e14U, 0x1c2c0404U, 0x1c2c0434U, 0x1c2c1414U, 0x1c2c2c2cU, 0x1c340c24U, 0x1c341c34U, 0x1c34341cU, 0x1c3e1c1cU, 0x1c3e3404U, 0x24040424U, 0x24040c3eU, 0x24041c2cU, 0x24041c3eU, 0x24042c1cU, 0x24042c3eU, 0x240c3e24U, 0x24141404U, 0x24141c3eU, 0x24142404U, 0x24143404U, 0x24143434U, 0x241c043eU, 0x241c242cU, 0x24240424U, 0x24242c0cU, 0x24243424U, 0x242c142cU, 0x242c241cU, 0x242c3e04U, 0x243e042cU, 0x243e0c04U, 0x243e0c14U, 0x243e1c04U, 0x2c040c14U, 0x2c04240cU, 0x2c043e04U, 0x2c0c0404U, 0x2c0c0434U, 0x2c0c1434U, 0x2c0c2c2cU, 0x2c140c24U, 0x2c141c14U, 0x2c143e14U, 0x2c1c0414U, 0x2c1c2c1cU, 0x2c240c04U, 0x2c24141cU, 0x2c24143eU, 0x2c243e14U, 0x2c2c0414U, 0x2c2c1c0cU, 0x2c342c04U, 0x2c3e1424U, 0x2c3e2414U, 0x34041424U, 0x34042424U, 0x34042434U, 0x34043424U, 0x340c140cU, 0x340c340cU, 0x34140c3eU, 0x34143424U, 0x341c1c04U, 0x341c1c34U, 0x34242424U, 0x342c042cU, 0x342c2c14U, 0x34341c1cU, 0x343e041cU, 0x343e140cU, 0x3e04041cU, 0x3e04042cU, 0x3e04043eU, 0x3e040c04U, 0x3e041c14U, 0x3e042c14U, 0x3e0c1434U, 0x3e0c2404U, 0x3e140c14U, 0x3e14242cU, 0x3e142c14U, 0x3e1c0404U, 0x3e1c0c2cU, 0x3e1c1c1cU, 0x3e1c3404U, 0x3e24140cU, 0x3e24240cU, 0x3e2c0404U, 0x3e2c0414U, 0x3e2c1424U, 0x3e341c04U};
  static const uint8_t weft_iq3xxs_ksigns[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
  static const uint8_t weft_iq3xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v6 = __riscv_vle8_v_u8m1(weft_iq3xxs_kmask, 8);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_i32_view
  const int32_t* v7 = (const int32_t*) weft_iq3xxs_grid;
  for (size_t v8 = 0; v8 < v5; v8 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v9 = v8 * 98;
    const uint8_t* v10 = v2 + v9;
    size_t v11 = v8 * 256;
    float* v12 = v3 + v11;
    float* v13 = (float*) v12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v10);
    const uint8_t* v15 = v10 + 2;
    const uint8_t* v16 = (const uint8_t*) v15;
    const uint8_t* v17 = v10 + 66;
    const uint8_t* v18 = (const uint8_t*) v17;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v19 = v18[0];
    uint32_t v20 = (uint32_t) v19;
    const uint8_t v21 = v18[1];
    uint32_t v22 = (uint32_t) v21;
    uint32_t v23 = v22 << 8u;
    uint32_t v24 = v20 | v23;
    const uint8_t v25 = v18[2];
    uint32_t v26 = (uint32_t) v25;
    uint32_t v27 = v26 << 16u;
    uint32_t v28 = v24 | v27;
    const uint8_t v29 = v18[3];
    uint32_t v30 = (uint32_t) v29;
    uint32_t v31 = v30 << 24u;
    uint32_t v32 = v28 | v31;
    uint32_t v33 = v32 >> 28u;
    int v34 = (int) v33;
    float v35 = (float) v34;
    float v36 = 0.5f + v35;
    float v37 = v14 * v36;
    float v38 = v37 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v39[2];
    const uint8_t v40 = v16[0];
    int v41 = (int) v40;
    int v42 = v41 << 2;
    uint16_t v43 = (uint16_t) v42;
    v39[0] = v43;
    const uint8_t v44 = v16[1];
    int v45 = (int) v44;
    int v46 = v45 << 2;
    uint16_t v47 = (uint16_t) v46;
    v39[1] = v47;
    uint16_t* v48 = &v39[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v49 = __riscv_vle16_v_u16mf2(v48, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v50 = __riscv_vluxei16_v_i32m1(v7, v49, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v51 = __riscv_vreinterpret_v_i32m1_i8m1(v50);
    uint32_t v52 = v32 >> 0u;
    uint32_t v53 = v52 & 127u;
    int v54 = (int) v53;
    const uint8_t v55 = weft_iq3xxs_ksigns[v54];
    int v56 = (int) v55;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v57 = __riscv_vmv_v_x_u8m1(v56, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v58 = __riscv_vand_vv_u8m1(v57, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v59 = __riscv_vmsne_vx_u8m1_b8(v58, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v60 = __riscv_vneg_v_i8m1(v51, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v61 = __riscv_vmerge_vvm_i8m1(v51, v60, v59, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v62 = __riscv_vsext_vf4_i32m4(v61, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v63 = __riscv_vfcvt_f_x_v_f32m4(v62, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v64 = __riscv_vfmul_vf_f32m4(v63, v38, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v13, v64, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v65[2];
    const uint8_t v66 = v16[2];
    int v67 = (int) v66;
    int v68 = v67 << 2;
    uint16_t v69 = (uint16_t) v68;
    v65[0] = v69;
    const uint8_t v70 = v16[3];
    int v71 = (int) v70;
    int v72 = v71 << 2;
    uint16_t v73 = (uint16_t) v72;
    v65[1] = v73;
    uint16_t* v74 = &v65[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v75 = __riscv_vle16_v_u16mf2(v74, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v76 = __riscv_vluxei16_v_i32m1(v7, v75, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v77 = __riscv_vreinterpret_v_i32m1_i8m1(v76);
    uint32_t v78 = v32 >> 7u;
    uint32_t v79 = v78 & 127u;
    int v80 = (int) v79;
    const uint8_t v81 = weft_iq3xxs_ksigns[v80];
    int v82 = (int) v81;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v83 = __riscv_vmv_v_x_u8m1(v82, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v84 = __riscv_vand_vv_u8m1(v83, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v85 = __riscv_vmsne_vx_u8m1_b8(v84, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v86 = __riscv_vneg_v_i8m1(v77, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v87 = __riscv_vmerge_vvm_i8m1(v77, v86, v85, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v88 = __riscv_vsext_vf4_i32m4(v87, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v89 = __riscv_vfcvt_f_x_v_f32m4(v88, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v90 = __riscv_vfmul_vf_f32m4(v89, v38, 8);
    float* v91 = v13 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v91, v90, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v92[2];
    const uint8_t v93 = v16[4];
    int v94 = (int) v93;
    int v95 = v94 << 2;
    uint16_t v96 = (uint16_t) v95;
    v92[0] = v96;
    const uint8_t v97 = v16[5];
    int v98 = (int) v97;
    int v99 = v98 << 2;
    uint16_t v100 = (uint16_t) v99;
    v92[1] = v100;
    uint16_t* v101 = &v92[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v102 = __riscv_vle16_v_u16mf2(v101, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v103 = __riscv_vluxei16_v_i32m1(v7, v102, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v104 = __riscv_vreinterpret_v_i32m1_i8m1(v103);
    uint32_t v105 = v32 >> 14u;
    uint32_t v106 = v105 & 127u;
    int v107 = (int) v106;
    const uint8_t v108 = weft_iq3xxs_ksigns[v107];
    int v109 = (int) v108;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v110 = __riscv_vmv_v_x_u8m1(v109, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v111 = __riscv_vand_vv_u8m1(v110, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v112 = __riscv_vmsne_vx_u8m1_b8(v111, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v113 = __riscv_vneg_v_i8m1(v104, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v114 = __riscv_vmerge_vvm_i8m1(v104, v113, v112, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v115 = __riscv_vsext_vf4_i32m4(v114, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v116 = __riscv_vfcvt_f_x_v_f32m4(v115, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v117 = __riscv_vfmul_vf_f32m4(v116, v38, 8);
    float* v118 = v13 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v118, v117, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v119[2];
    const uint8_t v120 = v16[6];
    int v121 = (int) v120;
    int v122 = v121 << 2;
    uint16_t v123 = (uint16_t) v122;
    v119[0] = v123;
    const uint8_t v124 = v16[7];
    int v125 = (int) v124;
    int v126 = v125 << 2;
    uint16_t v127 = (uint16_t) v126;
    v119[1] = v127;
    uint16_t* v128 = &v119[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v129 = __riscv_vle16_v_u16mf2(v128, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v130 = __riscv_vluxei16_v_i32m1(v7, v129, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v131 = __riscv_vreinterpret_v_i32m1_i8m1(v130);
    uint32_t v132 = v32 >> 21u;
    uint32_t v133 = v132 & 127u;
    int v134 = (int) v133;
    const uint8_t v135 = weft_iq3xxs_ksigns[v134];
    int v136 = (int) v135;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v137 = __riscv_vmv_v_x_u8m1(v136, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v138 = __riscv_vand_vv_u8m1(v137, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v139 = __riscv_vmsne_vx_u8m1_b8(v138, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v140 = __riscv_vneg_v_i8m1(v131, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v141 = __riscv_vmerge_vvm_i8m1(v131, v140, v139, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v142 = __riscv_vsext_vf4_i32m4(v141, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v143 = __riscv_vfcvt_f_x_v_f32m4(v142, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v144 = __riscv_vfmul_vf_f32m4(v143, v38, 8);
    float* v145 = v13 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v145, v144, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v146 = v18 + 4;
    const uint8_t v147 = v146[0];
    uint32_t v148 = (uint32_t) v147;
    const uint8_t v149 = v146[1];
    uint32_t v150 = (uint32_t) v149;
    uint32_t v151 = v150 << 8u;
    uint32_t v152 = v148 | v151;
    const uint8_t v153 = v146[2];
    uint32_t v154 = (uint32_t) v153;
    uint32_t v155 = v154 << 16u;
    uint32_t v156 = v152 | v155;
    const uint8_t v157 = v146[3];
    uint32_t v158 = (uint32_t) v157;
    uint32_t v159 = v158 << 24u;
    uint32_t v160 = v156 | v159;
    uint32_t v161 = v160 >> 28u;
    int v162 = (int) v161;
    float v163 = (float) v162;
    float v164 = 0.5f + v163;
    float v165 = v14 * v164;
    float v166 = v165 * 0.5f;
    const uint8_t* v167 = v16 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v168[2];
    const uint8_t v169 = v167[0];
    int v170 = (int) v169;
    int v171 = v170 << 2;
    uint16_t v172 = (uint16_t) v171;
    v168[0] = v172;
    const uint8_t v173 = v167[1];
    int v174 = (int) v173;
    int v175 = v174 << 2;
    uint16_t v176 = (uint16_t) v175;
    v168[1] = v176;
    uint16_t* v177 = &v168[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v178 = __riscv_vle16_v_u16mf2(v177, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v179 = __riscv_vluxei16_v_i32m1(v7, v178, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v180 = __riscv_vreinterpret_v_i32m1_i8m1(v179);
    uint32_t v181 = v160 >> 0u;
    uint32_t v182 = v181 & 127u;
    int v183 = (int) v182;
    const uint8_t v184 = weft_iq3xxs_ksigns[v183];
    int v185 = (int) v184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v186 = __riscv_vmv_v_x_u8m1(v185, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v187 = __riscv_vand_vv_u8m1(v186, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v188 = __riscv_vmsne_vx_u8m1_b8(v187, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v189 = __riscv_vneg_v_i8m1(v180, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v190 = __riscv_vmerge_vvm_i8m1(v180, v189, v188, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v191 = __riscv_vsext_vf4_i32m4(v190, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v192 = __riscv_vfcvt_f_x_v_f32m4(v191, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v193 = __riscv_vfmul_vf_f32m4(v192, v166, 8);
    float* v194 = v13 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v194, v193, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v195[2];
    const uint8_t v196 = v167[2];
    int v197 = (int) v196;
    int v198 = v197 << 2;
    uint16_t v199 = (uint16_t) v198;
    v195[0] = v199;
    const uint8_t v200 = v167[3];
    int v201 = (int) v200;
    int v202 = v201 << 2;
    uint16_t v203 = (uint16_t) v202;
    v195[1] = v203;
    uint16_t* v204 = &v195[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v205 = __riscv_vle16_v_u16mf2(v204, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v206 = __riscv_vluxei16_v_i32m1(v7, v205, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v207 = __riscv_vreinterpret_v_i32m1_i8m1(v206);
    uint32_t v208 = v160 >> 7u;
    uint32_t v209 = v208 & 127u;
    int v210 = (int) v209;
    const uint8_t v211 = weft_iq3xxs_ksigns[v210];
    int v212 = (int) v211;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v213 = __riscv_vmv_v_x_u8m1(v212, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v214 = __riscv_vand_vv_u8m1(v213, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v215 = __riscv_vmsne_vx_u8m1_b8(v214, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v216 = __riscv_vneg_v_i8m1(v207, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v217 = __riscv_vmerge_vvm_i8m1(v207, v216, v215, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v218 = __riscv_vsext_vf4_i32m4(v217, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v219 = __riscv_vfcvt_f_x_v_f32m4(v218, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v220 = __riscv_vfmul_vf_f32m4(v219, v166, 8);
    float* v221 = v13 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v221, v220, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v222[2];
    const uint8_t v223 = v167[4];
    int v224 = (int) v223;
    int v225 = v224 << 2;
    uint16_t v226 = (uint16_t) v225;
    v222[0] = v226;
    const uint8_t v227 = v167[5];
    int v228 = (int) v227;
    int v229 = v228 << 2;
    uint16_t v230 = (uint16_t) v229;
    v222[1] = v230;
    uint16_t* v231 = &v222[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v232 = __riscv_vle16_v_u16mf2(v231, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v233 = __riscv_vluxei16_v_i32m1(v7, v232, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v234 = __riscv_vreinterpret_v_i32m1_i8m1(v233);
    uint32_t v235 = v160 >> 14u;
    uint32_t v236 = v235 & 127u;
    int v237 = (int) v236;
    const uint8_t v238 = weft_iq3xxs_ksigns[v237];
    int v239 = (int) v238;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v240 = __riscv_vmv_v_x_u8m1(v239, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v241 = __riscv_vand_vv_u8m1(v240, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v242 = __riscv_vmsne_vx_u8m1_b8(v241, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v243 = __riscv_vneg_v_i8m1(v234, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v244 = __riscv_vmerge_vvm_i8m1(v234, v243, v242, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v245 = __riscv_vsext_vf4_i32m4(v244, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v246 = __riscv_vfcvt_f_x_v_f32m4(v245, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v247 = __riscv_vfmul_vf_f32m4(v246, v166, 8);
    float* v248 = v13 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v248, v247, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v249[2];
    const uint8_t v250 = v167[6];
    int v251 = (int) v250;
    int v252 = v251 << 2;
    uint16_t v253 = (uint16_t) v252;
    v249[0] = v253;
    const uint8_t v254 = v167[7];
    int v255 = (int) v254;
    int v256 = v255 << 2;
    uint16_t v257 = (uint16_t) v256;
    v249[1] = v257;
    uint16_t* v258 = &v249[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v259 = __riscv_vle16_v_u16mf2(v258, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v260 = __riscv_vluxei16_v_i32m1(v7, v259, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v261 = __riscv_vreinterpret_v_i32m1_i8m1(v260);
    uint32_t v262 = v160 >> 21u;
    uint32_t v263 = v262 & 127u;
    int v264 = (int) v263;
    const uint8_t v265 = weft_iq3xxs_ksigns[v264];
    int v266 = (int) v265;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v267 = __riscv_vmv_v_x_u8m1(v266, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v268 = __riscv_vand_vv_u8m1(v267, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v269 = __riscv_vmsne_vx_u8m1_b8(v268, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v270 = __riscv_vneg_v_i8m1(v261, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v271 = __riscv_vmerge_vvm_i8m1(v261, v270, v269, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v272 = __riscv_vsext_vf4_i32m4(v271, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v273 = __riscv_vfcvt_f_x_v_f32m4(v272, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v274 = __riscv_vfmul_vf_f32m4(v273, v166, 8);
    float* v275 = v13 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v275, v274, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v276 = v18 + 8;
    const uint8_t v277 = v276[0];
    uint32_t v278 = (uint32_t) v277;
    const uint8_t v279 = v276[1];
    uint32_t v280 = (uint32_t) v279;
    uint32_t v281 = v280 << 8u;
    uint32_t v282 = v278 | v281;
    const uint8_t v283 = v276[2];
    uint32_t v284 = (uint32_t) v283;
    uint32_t v285 = v284 << 16u;
    uint32_t v286 = v282 | v285;
    const uint8_t v287 = v276[3];
    uint32_t v288 = (uint32_t) v287;
    uint32_t v289 = v288 << 24u;
    uint32_t v290 = v286 | v289;
    uint32_t v291 = v290 >> 28u;
    int v292 = (int) v291;
    float v293 = (float) v292;
    float v294 = 0.5f + v293;
    float v295 = v14 * v294;
    float v296 = v295 * 0.5f;
    const uint8_t* v297 = v16 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v298[2];
    const uint8_t v299 = v297[0];
    int v300 = (int) v299;
    int v301 = v300 << 2;
    uint16_t v302 = (uint16_t) v301;
    v298[0] = v302;
    const uint8_t v303 = v297[1];
    int v304 = (int) v303;
    int v305 = v304 << 2;
    uint16_t v306 = (uint16_t) v305;
    v298[1] = v306;
    uint16_t* v307 = &v298[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v308 = __riscv_vle16_v_u16mf2(v307, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v309 = __riscv_vluxei16_v_i32m1(v7, v308, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v310 = __riscv_vreinterpret_v_i32m1_i8m1(v309);
    uint32_t v311 = v290 >> 0u;
    uint32_t v312 = v311 & 127u;
    int v313 = (int) v312;
    const uint8_t v314 = weft_iq3xxs_ksigns[v313];
    int v315 = (int) v314;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v316 = __riscv_vmv_v_x_u8m1(v315, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v317 = __riscv_vand_vv_u8m1(v316, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v318 = __riscv_vmsne_vx_u8m1_b8(v317, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v319 = __riscv_vneg_v_i8m1(v310, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v320 = __riscv_vmerge_vvm_i8m1(v310, v319, v318, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v321 = __riscv_vsext_vf4_i32m4(v320, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v322 = __riscv_vfcvt_f_x_v_f32m4(v321, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v323 = __riscv_vfmul_vf_f32m4(v322, v296, 8);
    float* v324 = v13 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v324, v323, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v325[2];
    const uint8_t v326 = v297[2];
    int v327 = (int) v326;
    int v328 = v327 << 2;
    uint16_t v329 = (uint16_t) v328;
    v325[0] = v329;
    const uint8_t v330 = v297[3];
    int v331 = (int) v330;
    int v332 = v331 << 2;
    uint16_t v333 = (uint16_t) v332;
    v325[1] = v333;
    uint16_t* v334 = &v325[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v335 = __riscv_vle16_v_u16mf2(v334, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v336 = __riscv_vluxei16_v_i32m1(v7, v335, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v337 = __riscv_vreinterpret_v_i32m1_i8m1(v336);
    uint32_t v338 = v290 >> 7u;
    uint32_t v339 = v338 & 127u;
    int v340 = (int) v339;
    const uint8_t v341 = weft_iq3xxs_ksigns[v340];
    int v342 = (int) v341;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v343 = __riscv_vmv_v_x_u8m1(v342, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v344 = __riscv_vand_vv_u8m1(v343, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v345 = __riscv_vmsne_vx_u8m1_b8(v344, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v346 = __riscv_vneg_v_i8m1(v337, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v347 = __riscv_vmerge_vvm_i8m1(v337, v346, v345, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v348 = __riscv_vsext_vf4_i32m4(v347, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v349 = __riscv_vfcvt_f_x_v_f32m4(v348, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v350 = __riscv_vfmul_vf_f32m4(v349, v296, 8);
    float* v351 = v13 + 72;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v351, v350, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v352[2];
    const uint8_t v353 = v297[4];
    int v354 = (int) v353;
    int v355 = v354 << 2;
    uint16_t v356 = (uint16_t) v355;
    v352[0] = v356;
    const uint8_t v357 = v297[5];
    int v358 = (int) v357;
    int v359 = v358 << 2;
    uint16_t v360 = (uint16_t) v359;
    v352[1] = v360;
    uint16_t* v361 = &v352[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v362 = __riscv_vle16_v_u16mf2(v361, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v363 = __riscv_vluxei16_v_i32m1(v7, v362, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v364 = __riscv_vreinterpret_v_i32m1_i8m1(v363);
    uint32_t v365 = v290 >> 14u;
    uint32_t v366 = v365 & 127u;
    int v367 = (int) v366;
    const uint8_t v368 = weft_iq3xxs_ksigns[v367];
    int v369 = (int) v368;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v370 = __riscv_vmv_v_x_u8m1(v369, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v371 = __riscv_vand_vv_u8m1(v370, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v372 = __riscv_vmsne_vx_u8m1_b8(v371, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v373 = __riscv_vneg_v_i8m1(v364, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v374 = __riscv_vmerge_vvm_i8m1(v364, v373, v372, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v375 = __riscv_vsext_vf4_i32m4(v374, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v376 = __riscv_vfcvt_f_x_v_f32m4(v375, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v377 = __riscv_vfmul_vf_f32m4(v376, v296, 8);
    float* v378 = v13 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v378, v377, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v379[2];
    const uint8_t v380 = v297[6];
    int v381 = (int) v380;
    int v382 = v381 << 2;
    uint16_t v383 = (uint16_t) v382;
    v379[0] = v383;
    const uint8_t v384 = v297[7];
    int v385 = (int) v384;
    int v386 = v385 << 2;
    uint16_t v387 = (uint16_t) v386;
    v379[1] = v387;
    uint16_t* v388 = &v379[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v389 = __riscv_vle16_v_u16mf2(v388, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v390 = __riscv_vluxei16_v_i32m1(v7, v389, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v391 = __riscv_vreinterpret_v_i32m1_i8m1(v390);
    uint32_t v392 = v290 >> 21u;
    uint32_t v393 = v392 & 127u;
    int v394 = (int) v393;
    const uint8_t v395 = weft_iq3xxs_ksigns[v394];
    int v396 = (int) v395;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v397 = __riscv_vmv_v_x_u8m1(v396, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v398 = __riscv_vand_vv_u8m1(v397, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v399 = __riscv_vmsne_vx_u8m1_b8(v398, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v400 = __riscv_vneg_v_i8m1(v391, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v401 = __riscv_vmerge_vvm_i8m1(v391, v400, v399, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v402 = __riscv_vsext_vf4_i32m4(v401, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v403 = __riscv_vfcvt_f_x_v_f32m4(v402, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v404 = __riscv_vfmul_vf_f32m4(v403, v296, 8);
    float* v405 = v13 + 88;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v405, v404, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v406 = v18 + 12;
    const uint8_t v407 = v406[0];
    uint32_t v408 = (uint32_t) v407;
    const uint8_t v409 = v406[1];
    uint32_t v410 = (uint32_t) v409;
    uint32_t v411 = v410 << 8u;
    uint32_t v412 = v408 | v411;
    const uint8_t v413 = v406[2];
    uint32_t v414 = (uint32_t) v413;
    uint32_t v415 = v414 << 16u;
    uint32_t v416 = v412 | v415;
    const uint8_t v417 = v406[3];
    uint32_t v418 = (uint32_t) v417;
    uint32_t v419 = v418 << 24u;
    uint32_t v420 = v416 | v419;
    uint32_t v421 = v420 >> 28u;
    int v422 = (int) v421;
    float v423 = (float) v422;
    float v424 = 0.5f + v423;
    float v425 = v14 * v424;
    float v426 = v425 * 0.5f;
    const uint8_t* v427 = v16 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v428[2];
    const uint8_t v429 = v427[0];
    int v430 = (int) v429;
    int v431 = v430 << 2;
    uint16_t v432 = (uint16_t) v431;
    v428[0] = v432;
    const uint8_t v433 = v427[1];
    int v434 = (int) v433;
    int v435 = v434 << 2;
    uint16_t v436 = (uint16_t) v435;
    v428[1] = v436;
    uint16_t* v437 = &v428[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v438 = __riscv_vle16_v_u16mf2(v437, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v439 = __riscv_vluxei16_v_i32m1(v7, v438, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v440 = __riscv_vreinterpret_v_i32m1_i8m1(v439);
    uint32_t v441 = v420 >> 0u;
    uint32_t v442 = v441 & 127u;
    int v443 = (int) v442;
    const uint8_t v444 = weft_iq3xxs_ksigns[v443];
    int v445 = (int) v444;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v446 = __riscv_vmv_v_x_u8m1(v445, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v447 = __riscv_vand_vv_u8m1(v446, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v448 = __riscv_vmsne_vx_u8m1_b8(v447, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v449 = __riscv_vneg_v_i8m1(v440, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v450 = __riscv_vmerge_vvm_i8m1(v440, v449, v448, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v451 = __riscv_vsext_vf4_i32m4(v450, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v452 = __riscv_vfcvt_f_x_v_f32m4(v451, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v453 = __riscv_vfmul_vf_f32m4(v452, v426, 8);
    float* v454 = v13 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v454, v453, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v455[2];
    const uint8_t v456 = v427[2];
    int v457 = (int) v456;
    int v458 = v457 << 2;
    uint16_t v459 = (uint16_t) v458;
    v455[0] = v459;
    const uint8_t v460 = v427[3];
    int v461 = (int) v460;
    int v462 = v461 << 2;
    uint16_t v463 = (uint16_t) v462;
    v455[1] = v463;
    uint16_t* v464 = &v455[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v465 = __riscv_vle16_v_u16mf2(v464, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v466 = __riscv_vluxei16_v_i32m1(v7, v465, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v467 = __riscv_vreinterpret_v_i32m1_i8m1(v466);
    uint32_t v468 = v420 >> 7u;
    uint32_t v469 = v468 & 127u;
    int v470 = (int) v469;
    const uint8_t v471 = weft_iq3xxs_ksigns[v470];
    int v472 = (int) v471;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v473 = __riscv_vmv_v_x_u8m1(v472, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v474 = __riscv_vand_vv_u8m1(v473, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v475 = __riscv_vmsne_vx_u8m1_b8(v474, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v476 = __riscv_vneg_v_i8m1(v467, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v477 = __riscv_vmerge_vvm_i8m1(v467, v476, v475, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v478 = __riscv_vsext_vf4_i32m4(v477, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v479 = __riscv_vfcvt_f_x_v_f32m4(v478, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v480 = __riscv_vfmul_vf_f32m4(v479, v426, 8);
    float* v481 = v13 + 104;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v481, v480, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v482[2];
    const uint8_t v483 = v427[4];
    int v484 = (int) v483;
    int v485 = v484 << 2;
    uint16_t v486 = (uint16_t) v485;
    v482[0] = v486;
    const uint8_t v487 = v427[5];
    int v488 = (int) v487;
    int v489 = v488 << 2;
    uint16_t v490 = (uint16_t) v489;
    v482[1] = v490;
    uint16_t* v491 = &v482[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v492 = __riscv_vle16_v_u16mf2(v491, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v493 = __riscv_vluxei16_v_i32m1(v7, v492, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v494 = __riscv_vreinterpret_v_i32m1_i8m1(v493);
    uint32_t v495 = v420 >> 14u;
    uint32_t v496 = v495 & 127u;
    int v497 = (int) v496;
    const uint8_t v498 = weft_iq3xxs_ksigns[v497];
    int v499 = (int) v498;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v500 = __riscv_vmv_v_x_u8m1(v499, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v501 = __riscv_vand_vv_u8m1(v500, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v502 = __riscv_vmsne_vx_u8m1_b8(v501, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v503 = __riscv_vneg_v_i8m1(v494, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v504 = __riscv_vmerge_vvm_i8m1(v494, v503, v502, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v505 = __riscv_vsext_vf4_i32m4(v504, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v506 = __riscv_vfcvt_f_x_v_f32m4(v505, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v507 = __riscv_vfmul_vf_f32m4(v506, v426, 8);
    float* v508 = v13 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v508, v507, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v509[2];
    const uint8_t v510 = v427[6];
    int v511 = (int) v510;
    int v512 = v511 << 2;
    uint16_t v513 = (uint16_t) v512;
    v509[0] = v513;
    const uint8_t v514 = v427[7];
    int v515 = (int) v514;
    int v516 = v515 << 2;
    uint16_t v517 = (uint16_t) v516;
    v509[1] = v517;
    uint16_t* v518 = &v509[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v519 = __riscv_vle16_v_u16mf2(v518, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v520 = __riscv_vluxei16_v_i32m1(v7, v519, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v521 = __riscv_vreinterpret_v_i32m1_i8m1(v520);
    uint32_t v522 = v420 >> 21u;
    uint32_t v523 = v522 & 127u;
    int v524 = (int) v523;
    const uint8_t v525 = weft_iq3xxs_ksigns[v524];
    int v526 = (int) v525;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v527 = __riscv_vmv_v_x_u8m1(v526, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v528 = __riscv_vand_vv_u8m1(v527, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v529 = __riscv_vmsne_vx_u8m1_b8(v528, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v530 = __riscv_vneg_v_i8m1(v521, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v531 = __riscv_vmerge_vvm_i8m1(v521, v530, v529, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v532 = __riscv_vsext_vf4_i32m4(v531, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v533 = __riscv_vfcvt_f_x_v_f32m4(v532, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v534 = __riscv_vfmul_vf_f32m4(v533, v426, 8);
    float* v535 = v13 + 120;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v535, v534, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v536 = v18 + 16;
    const uint8_t v537 = v536[0];
    uint32_t v538 = (uint32_t) v537;
    const uint8_t v539 = v536[1];
    uint32_t v540 = (uint32_t) v539;
    uint32_t v541 = v540 << 8u;
    uint32_t v542 = v538 | v541;
    const uint8_t v543 = v536[2];
    uint32_t v544 = (uint32_t) v543;
    uint32_t v545 = v544 << 16u;
    uint32_t v546 = v542 | v545;
    const uint8_t v547 = v536[3];
    uint32_t v548 = (uint32_t) v547;
    uint32_t v549 = v548 << 24u;
    uint32_t v550 = v546 | v549;
    uint32_t v551 = v550 >> 28u;
    int v552 = (int) v551;
    float v553 = (float) v552;
    float v554 = 0.5f + v553;
    float v555 = v14 * v554;
    float v556 = v555 * 0.5f;
    const uint8_t* v557 = v16 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v558[2];
    const uint8_t v559 = v557[0];
    int v560 = (int) v559;
    int v561 = v560 << 2;
    uint16_t v562 = (uint16_t) v561;
    v558[0] = v562;
    const uint8_t v563 = v557[1];
    int v564 = (int) v563;
    int v565 = v564 << 2;
    uint16_t v566 = (uint16_t) v565;
    v558[1] = v566;
    uint16_t* v567 = &v558[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v568 = __riscv_vle16_v_u16mf2(v567, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v569 = __riscv_vluxei16_v_i32m1(v7, v568, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v570 = __riscv_vreinterpret_v_i32m1_i8m1(v569);
    uint32_t v571 = v550 >> 0u;
    uint32_t v572 = v571 & 127u;
    int v573 = (int) v572;
    const uint8_t v574 = weft_iq3xxs_ksigns[v573];
    int v575 = (int) v574;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v576 = __riscv_vmv_v_x_u8m1(v575, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v577 = __riscv_vand_vv_u8m1(v576, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v578 = __riscv_vmsne_vx_u8m1_b8(v577, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v579 = __riscv_vneg_v_i8m1(v570, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v580 = __riscv_vmerge_vvm_i8m1(v570, v579, v578, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v581 = __riscv_vsext_vf4_i32m4(v580, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v582 = __riscv_vfcvt_f_x_v_f32m4(v581, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v583 = __riscv_vfmul_vf_f32m4(v582, v556, 8);
    float* v584 = v13 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v584, v583, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v585[2];
    const uint8_t v586 = v557[2];
    int v587 = (int) v586;
    int v588 = v587 << 2;
    uint16_t v589 = (uint16_t) v588;
    v585[0] = v589;
    const uint8_t v590 = v557[3];
    int v591 = (int) v590;
    int v592 = v591 << 2;
    uint16_t v593 = (uint16_t) v592;
    v585[1] = v593;
    uint16_t* v594 = &v585[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v595 = __riscv_vle16_v_u16mf2(v594, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v596 = __riscv_vluxei16_v_i32m1(v7, v595, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v597 = __riscv_vreinterpret_v_i32m1_i8m1(v596);
    uint32_t v598 = v550 >> 7u;
    uint32_t v599 = v598 & 127u;
    int v600 = (int) v599;
    const uint8_t v601 = weft_iq3xxs_ksigns[v600];
    int v602 = (int) v601;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v603 = __riscv_vmv_v_x_u8m1(v602, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v604 = __riscv_vand_vv_u8m1(v603, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v605 = __riscv_vmsne_vx_u8m1_b8(v604, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v606 = __riscv_vneg_v_i8m1(v597, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v607 = __riscv_vmerge_vvm_i8m1(v597, v606, v605, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v608 = __riscv_vsext_vf4_i32m4(v607, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v609 = __riscv_vfcvt_f_x_v_f32m4(v608, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v610 = __riscv_vfmul_vf_f32m4(v609, v556, 8);
    float* v611 = v13 + 136;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v611, v610, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v612[2];
    const uint8_t v613 = v557[4];
    int v614 = (int) v613;
    int v615 = v614 << 2;
    uint16_t v616 = (uint16_t) v615;
    v612[0] = v616;
    const uint8_t v617 = v557[5];
    int v618 = (int) v617;
    int v619 = v618 << 2;
    uint16_t v620 = (uint16_t) v619;
    v612[1] = v620;
    uint16_t* v621 = &v612[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v622 = __riscv_vle16_v_u16mf2(v621, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v623 = __riscv_vluxei16_v_i32m1(v7, v622, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v624 = __riscv_vreinterpret_v_i32m1_i8m1(v623);
    uint32_t v625 = v550 >> 14u;
    uint32_t v626 = v625 & 127u;
    int v627 = (int) v626;
    const uint8_t v628 = weft_iq3xxs_ksigns[v627];
    int v629 = (int) v628;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v630 = __riscv_vmv_v_x_u8m1(v629, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v631 = __riscv_vand_vv_u8m1(v630, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v632 = __riscv_vmsne_vx_u8m1_b8(v631, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v633 = __riscv_vneg_v_i8m1(v624, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v634 = __riscv_vmerge_vvm_i8m1(v624, v633, v632, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v635 = __riscv_vsext_vf4_i32m4(v634, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v636 = __riscv_vfcvt_f_x_v_f32m4(v635, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v637 = __riscv_vfmul_vf_f32m4(v636, v556, 8);
    float* v638 = v13 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v638, v637, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v639[2];
    const uint8_t v640 = v557[6];
    int v641 = (int) v640;
    int v642 = v641 << 2;
    uint16_t v643 = (uint16_t) v642;
    v639[0] = v643;
    const uint8_t v644 = v557[7];
    int v645 = (int) v644;
    int v646 = v645 << 2;
    uint16_t v647 = (uint16_t) v646;
    v639[1] = v647;
    uint16_t* v648 = &v639[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v649 = __riscv_vle16_v_u16mf2(v648, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v650 = __riscv_vluxei16_v_i32m1(v7, v649, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v651 = __riscv_vreinterpret_v_i32m1_i8m1(v650);
    uint32_t v652 = v550 >> 21u;
    uint32_t v653 = v652 & 127u;
    int v654 = (int) v653;
    const uint8_t v655 = weft_iq3xxs_ksigns[v654];
    int v656 = (int) v655;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v657 = __riscv_vmv_v_x_u8m1(v656, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v658 = __riscv_vand_vv_u8m1(v657, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v659 = __riscv_vmsne_vx_u8m1_b8(v658, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v660 = __riscv_vneg_v_i8m1(v651, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v661 = __riscv_vmerge_vvm_i8m1(v651, v660, v659, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v662 = __riscv_vsext_vf4_i32m4(v661, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v663 = __riscv_vfcvt_f_x_v_f32m4(v662, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v664 = __riscv_vfmul_vf_f32m4(v663, v556, 8);
    float* v665 = v13 + 152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v665, v664, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v666 = v18 + 20;
    const uint8_t v667 = v666[0];
    uint32_t v668 = (uint32_t) v667;
    const uint8_t v669 = v666[1];
    uint32_t v670 = (uint32_t) v669;
    uint32_t v671 = v670 << 8u;
    uint32_t v672 = v668 | v671;
    const uint8_t v673 = v666[2];
    uint32_t v674 = (uint32_t) v673;
    uint32_t v675 = v674 << 16u;
    uint32_t v676 = v672 | v675;
    const uint8_t v677 = v666[3];
    uint32_t v678 = (uint32_t) v677;
    uint32_t v679 = v678 << 24u;
    uint32_t v680 = v676 | v679;
    uint32_t v681 = v680 >> 28u;
    int v682 = (int) v681;
    float v683 = (float) v682;
    float v684 = 0.5f + v683;
    float v685 = v14 * v684;
    float v686 = v685 * 0.5f;
    const uint8_t* v687 = v16 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v688[2];
    const uint8_t v689 = v687[0];
    int v690 = (int) v689;
    int v691 = v690 << 2;
    uint16_t v692 = (uint16_t) v691;
    v688[0] = v692;
    const uint8_t v693 = v687[1];
    int v694 = (int) v693;
    int v695 = v694 << 2;
    uint16_t v696 = (uint16_t) v695;
    v688[1] = v696;
    uint16_t* v697 = &v688[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v698 = __riscv_vle16_v_u16mf2(v697, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v699 = __riscv_vluxei16_v_i32m1(v7, v698, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v700 = __riscv_vreinterpret_v_i32m1_i8m1(v699);
    uint32_t v701 = v680 >> 0u;
    uint32_t v702 = v701 & 127u;
    int v703 = (int) v702;
    const uint8_t v704 = weft_iq3xxs_ksigns[v703];
    int v705 = (int) v704;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v706 = __riscv_vmv_v_x_u8m1(v705, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v707 = __riscv_vand_vv_u8m1(v706, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v708 = __riscv_vmsne_vx_u8m1_b8(v707, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v709 = __riscv_vneg_v_i8m1(v700, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v710 = __riscv_vmerge_vvm_i8m1(v700, v709, v708, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v711 = __riscv_vsext_vf4_i32m4(v710, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v712 = __riscv_vfcvt_f_x_v_f32m4(v711, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v713 = __riscv_vfmul_vf_f32m4(v712, v686, 8);
    float* v714 = v13 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v714, v713, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v715[2];
    const uint8_t v716 = v687[2];
    int v717 = (int) v716;
    int v718 = v717 << 2;
    uint16_t v719 = (uint16_t) v718;
    v715[0] = v719;
    const uint8_t v720 = v687[3];
    int v721 = (int) v720;
    int v722 = v721 << 2;
    uint16_t v723 = (uint16_t) v722;
    v715[1] = v723;
    uint16_t* v724 = &v715[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v725 = __riscv_vle16_v_u16mf2(v724, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v726 = __riscv_vluxei16_v_i32m1(v7, v725, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v727 = __riscv_vreinterpret_v_i32m1_i8m1(v726);
    uint32_t v728 = v680 >> 7u;
    uint32_t v729 = v728 & 127u;
    int v730 = (int) v729;
    const uint8_t v731 = weft_iq3xxs_ksigns[v730];
    int v732 = (int) v731;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v733 = __riscv_vmv_v_x_u8m1(v732, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v734 = __riscv_vand_vv_u8m1(v733, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v735 = __riscv_vmsne_vx_u8m1_b8(v734, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v736 = __riscv_vneg_v_i8m1(v727, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v737 = __riscv_vmerge_vvm_i8m1(v727, v736, v735, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v738 = __riscv_vsext_vf4_i32m4(v737, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v739 = __riscv_vfcvt_f_x_v_f32m4(v738, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v740 = __riscv_vfmul_vf_f32m4(v739, v686, 8);
    float* v741 = v13 + 168;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v741, v740, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v742[2];
    const uint8_t v743 = v687[4];
    int v744 = (int) v743;
    int v745 = v744 << 2;
    uint16_t v746 = (uint16_t) v745;
    v742[0] = v746;
    const uint8_t v747 = v687[5];
    int v748 = (int) v747;
    int v749 = v748 << 2;
    uint16_t v750 = (uint16_t) v749;
    v742[1] = v750;
    uint16_t* v751 = &v742[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v752 = __riscv_vle16_v_u16mf2(v751, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v753 = __riscv_vluxei16_v_i32m1(v7, v752, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v754 = __riscv_vreinterpret_v_i32m1_i8m1(v753);
    uint32_t v755 = v680 >> 14u;
    uint32_t v756 = v755 & 127u;
    int v757 = (int) v756;
    const uint8_t v758 = weft_iq3xxs_ksigns[v757];
    int v759 = (int) v758;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v760 = __riscv_vmv_v_x_u8m1(v759, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v761 = __riscv_vand_vv_u8m1(v760, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v762 = __riscv_vmsne_vx_u8m1_b8(v761, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v763 = __riscv_vneg_v_i8m1(v754, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v764 = __riscv_vmerge_vvm_i8m1(v754, v763, v762, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v765 = __riscv_vsext_vf4_i32m4(v764, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v766 = __riscv_vfcvt_f_x_v_f32m4(v765, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v767 = __riscv_vfmul_vf_f32m4(v766, v686, 8);
    float* v768 = v13 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v768, v767, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v769[2];
    const uint8_t v770 = v687[6];
    int v771 = (int) v770;
    int v772 = v771 << 2;
    uint16_t v773 = (uint16_t) v772;
    v769[0] = v773;
    const uint8_t v774 = v687[7];
    int v775 = (int) v774;
    int v776 = v775 << 2;
    uint16_t v777 = (uint16_t) v776;
    v769[1] = v777;
    uint16_t* v778 = &v769[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v779 = __riscv_vle16_v_u16mf2(v778, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v780 = __riscv_vluxei16_v_i32m1(v7, v779, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v781 = __riscv_vreinterpret_v_i32m1_i8m1(v780);
    uint32_t v782 = v680 >> 21u;
    uint32_t v783 = v782 & 127u;
    int v784 = (int) v783;
    const uint8_t v785 = weft_iq3xxs_ksigns[v784];
    int v786 = (int) v785;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v787 = __riscv_vmv_v_x_u8m1(v786, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v788 = __riscv_vand_vv_u8m1(v787, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v789 = __riscv_vmsne_vx_u8m1_b8(v788, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v790 = __riscv_vneg_v_i8m1(v781, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v791 = __riscv_vmerge_vvm_i8m1(v781, v790, v789, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v792 = __riscv_vsext_vf4_i32m4(v791, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v793 = __riscv_vfcvt_f_x_v_f32m4(v792, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v794 = __riscv_vfmul_vf_f32m4(v793, v686, 8);
    float* v795 = v13 + 184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v795, v794, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v796 = v18 + 24;
    const uint8_t v797 = v796[0];
    uint32_t v798 = (uint32_t) v797;
    const uint8_t v799 = v796[1];
    uint32_t v800 = (uint32_t) v799;
    uint32_t v801 = v800 << 8u;
    uint32_t v802 = v798 | v801;
    const uint8_t v803 = v796[2];
    uint32_t v804 = (uint32_t) v803;
    uint32_t v805 = v804 << 16u;
    uint32_t v806 = v802 | v805;
    const uint8_t v807 = v796[3];
    uint32_t v808 = (uint32_t) v807;
    uint32_t v809 = v808 << 24u;
    uint32_t v810 = v806 | v809;
    uint32_t v811 = v810 >> 28u;
    int v812 = (int) v811;
    float v813 = (float) v812;
    float v814 = 0.5f + v813;
    float v815 = v14 * v814;
    float v816 = v815 * 0.5f;
    const uint8_t* v817 = v16 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v818[2];
    const uint8_t v819 = v817[0];
    int v820 = (int) v819;
    int v821 = v820 << 2;
    uint16_t v822 = (uint16_t) v821;
    v818[0] = v822;
    const uint8_t v823 = v817[1];
    int v824 = (int) v823;
    int v825 = v824 << 2;
    uint16_t v826 = (uint16_t) v825;
    v818[1] = v826;
    uint16_t* v827 = &v818[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v828 = __riscv_vle16_v_u16mf2(v827, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v829 = __riscv_vluxei16_v_i32m1(v7, v828, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v830 = __riscv_vreinterpret_v_i32m1_i8m1(v829);
    uint32_t v831 = v810 >> 0u;
    uint32_t v832 = v831 & 127u;
    int v833 = (int) v832;
    const uint8_t v834 = weft_iq3xxs_ksigns[v833];
    int v835 = (int) v834;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v836 = __riscv_vmv_v_x_u8m1(v835, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v837 = __riscv_vand_vv_u8m1(v836, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v838 = __riscv_vmsne_vx_u8m1_b8(v837, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v839 = __riscv_vneg_v_i8m1(v830, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v840 = __riscv_vmerge_vvm_i8m1(v830, v839, v838, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v841 = __riscv_vsext_vf4_i32m4(v840, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v842 = __riscv_vfcvt_f_x_v_f32m4(v841, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v843 = __riscv_vfmul_vf_f32m4(v842, v816, 8);
    float* v844 = v13 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v844, v843, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v845[2];
    const uint8_t v846 = v817[2];
    int v847 = (int) v846;
    int v848 = v847 << 2;
    uint16_t v849 = (uint16_t) v848;
    v845[0] = v849;
    const uint8_t v850 = v817[3];
    int v851 = (int) v850;
    int v852 = v851 << 2;
    uint16_t v853 = (uint16_t) v852;
    v845[1] = v853;
    uint16_t* v854 = &v845[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v855 = __riscv_vle16_v_u16mf2(v854, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v856 = __riscv_vluxei16_v_i32m1(v7, v855, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v857 = __riscv_vreinterpret_v_i32m1_i8m1(v856);
    uint32_t v858 = v810 >> 7u;
    uint32_t v859 = v858 & 127u;
    int v860 = (int) v859;
    const uint8_t v861 = weft_iq3xxs_ksigns[v860];
    int v862 = (int) v861;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v863 = __riscv_vmv_v_x_u8m1(v862, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v864 = __riscv_vand_vv_u8m1(v863, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v865 = __riscv_vmsne_vx_u8m1_b8(v864, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v866 = __riscv_vneg_v_i8m1(v857, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v867 = __riscv_vmerge_vvm_i8m1(v857, v866, v865, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v868 = __riscv_vsext_vf4_i32m4(v867, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v869 = __riscv_vfcvt_f_x_v_f32m4(v868, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v870 = __riscv_vfmul_vf_f32m4(v869, v816, 8);
    float* v871 = v13 + 200;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v871, v870, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v872[2];
    const uint8_t v873 = v817[4];
    int v874 = (int) v873;
    int v875 = v874 << 2;
    uint16_t v876 = (uint16_t) v875;
    v872[0] = v876;
    const uint8_t v877 = v817[5];
    int v878 = (int) v877;
    int v879 = v878 << 2;
    uint16_t v880 = (uint16_t) v879;
    v872[1] = v880;
    uint16_t* v881 = &v872[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v882 = __riscv_vle16_v_u16mf2(v881, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v883 = __riscv_vluxei16_v_i32m1(v7, v882, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v884 = __riscv_vreinterpret_v_i32m1_i8m1(v883);
    uint32_t v885 = v810 >> 14u;
    uint32_t v886 = v885 & 127u;
    int v887 = (int) v886;
    const uint8_t v888 = weft_iq3xxs_ksigns[v887];
    int v889 = (int) v888;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v890 = __riscv_vmv_v_x_u8m1(v889, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v891 = __riscv_vand_vv_u8m1(v890, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v892 = __riscv_vmsne_vx_u8m1_b8(v891, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v893 = __riscv_vneg_v_i8m1(v884, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v894 = __riscv_vmerge_vvm_i8m1(v884, v893, v892, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v895 = __riscv_vsext_vf4_i32m4(v894, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v896 = __riscv_vfcvt_f_x_v_f32m4(v895, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v897 = __riscv_vfmul_vf_f32m4(v896, v816, 8);
    float* v898 = v13 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v898, v897, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v899[2];
    const uint8_t v900 = v817[6];
    int v901 = (int) v900;
    int v902 = v901 << 2;
    uint16_t v903 = (uint16_t) v902;
    v899[0] = v903;
    const uint8_t v904 = v817[7];
    int v905 = (int) v904;
    int v906 = v905 << 2;
    uint16_t v907 = (uint16_t) v906;
    v899[1] = v907;
    uint16_t* v908 = &v899[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v909 = __riscv_vle16_v_u16mf2(v908, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v910 = __riscv_vluxei16_v_i32m1(v7, v909, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v911 = __riscv_vreinterpret_v_i32m1_i8m1(v910);
    uint32_t v912 = v810 >> 21u;
    uint32_t v913 = v912 & 127u;
    int v914 = (int) v913;
    const uint8_t v915 = weft_iq3xxs_ksigns[v914];
    int v916 = (int) v915;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v917 = __riscv_vmv_v_x_u8m1(v916, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v918 = __riscv_vand_vv_u8m1(v917, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v919 = __riscv_vmsne_vx_u8m1_b8(v918, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v920 = __riscv_vneg_v_i8m1(v911, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v921 = __riscv_vmerge_vvm_i8m1(v911, v920, v919, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v922 = __riscv_vsext_vf4_i32m4(v921, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v923 = __riscv_vfcvt_f_x_v_f32m4(v922, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v924 = __riscv_vfmul_vf_f32m4(v923, v816, 8);
    float* v925 = v13 + 216;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v925, v924, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v926 = v18 + 28;
    const uint8_t v927 = v926[0];
    uint32_t v928 = (uint32_t) v927;
    const uint8_t v929 = v926[1];
    uint32_t v930 = (uint32_t) v929;
    uint32_t v931 = v930 << 8u;
    uint32_t v932 = v928 | v931;
    const uint8_t v933 = v926[2];
    uint32_t v934 = (uint32_t) v933;
    uint32_t v935 = v934 << 16u;
    uint32_t v936 = v932 | v935;
    const uint8_t v937 = v926[3];
    uint32_t v938 = (uint32_t) v937;
    uint32_t v939 = v938 << 24u;
    uint32_t v940 = v936 | v939;
    uint32_t v941 = v940 >> 28u;
    int v942 = (int) v941;
    float v943 = (float) v942;
    float v944 = 0.5f + v943;
    float v945 = v14 * v944;
    float v946 = v945 * 0.5f;
    const uint8_t* v947 = v16 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v948[2];
    const uint8_t v949 = v947[0];
    int v950 = (int) v949;
    int v951 = v950 << 2;
    uint16_t v952 = (uint16_t) v951;
    v948[0] = v952;
    const uint8_t v953 = v947[1];
    int v954 = (int) v953;
    int v955 = v954 << 2;
    uint16_t v956 = (uint16_t) v955;
    v948[1] = v956;
    uint16_t* v957 = &v948[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v958 = __riscv_vle16_v_u16mf2(v957, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v959 = __riscv_vluxei16_v_i32m1(v7, v958, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v960 = __riscv_vreinterpret_v_i32m1_i8m1(v959);
    uint32_t v961 = v940 >> 0u;
    uint32_t v962 = v961 & 127u;
    int v963 = (int) v962;
    const uint8_t v964 = weft_iq3xxs_ksigns[v963];
    int v965 = (int) v964;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v966 = __riscv_vmv_v_x_u8m1(v965, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v967 = __riscv_vand_vv_u8m1(v966, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v968 = __riscv_vmsne_vx_u8m1_b8(v967, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v969 = __riscv_vneg_v_i8m1(v960, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v970 = __riscv_vmerge_vvm_i8m1(v960, v969, v968, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v971 = __riscv_vsext_vf4_i32m4(v970, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v972 = __riscv_vfcvt_f_x_v_f32m4(v971, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v973 = __riscv_vfmul_vf_f32m4(v972, v946, 8);
    float* v974 = v13 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v974, v973, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v975[2];
    const uint8_t v976 = v947[2];
    int v977 = (int) v976;
    int v978 = v977 << 2;
    uint16_t v979 = (uint16_t) v978;
    v975[0] = v979;
    const uint8_t v980 = v947[3];
    int v981 = (int) v980;
    int v982 = v981 << 2;
    uint16_t v983 = (uint16_t) v982;
    v975[1] = v983;
    uint16_t* v984 = &v975[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v985 = __riscv_vle16_v_u16mf2(v984, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v986 = __riscv_vluxei16_v_i32m1(v7, v985, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v987 = __riscv_vreinterpret_v_i32m1_i8m1(v986);
    uint32_t v988 = v940 >> 7u;
    uint32_t v989 = v988 & 127u;
    int v990 = (int) v989;
    const uint8_t v991 = weft_iq3xxs_ksigns[v990];
    int v992 = (int) v991;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v993 = __riscv_vmv_v_x_u8m1(v992, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v994 = __riscv_vand_vv_u8m1(v993, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v995 = __riscv_vmsne_vx_u8m1_b8(v994, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v996 = __riscv_vneg_v_i8m1(v987, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v997 = __riscv_vmerge_vvm_i8m1(v987, v996, v995, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v998 = __riscv_vsext_vf4_i32m4(v997, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v999 = __riscv_vfcvt_f_x_v_f32m4(v998, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v1000 = __riscv_vfmul_vf_f32m4(v999, v946, 8);
    float* v1001 = v13 + 232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v1001, v1000, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v1002[2];
    const uint8_t v1003 = v947[4];
    int v1004 = (int) v1003;
    int v1005 = v1004 << 2;
    uint16_t v1006 = (uint16_t) v1005;
    v1002[0] = v1006;
    const uint8_t v1007 = v947[5];
    int v1008 = (int) v1007;
    int v1009 = v1008 << 2;
    uint16_t v1010 = (uint16_t) v1009;
    v1002[1] = v1010;
    uint16_t* v1011 = &v1002[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1012 = __riscv_vle16_v_u16mf2(v1011, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1013 = __riscv_vluxei16_v_i32m1(v7, v1012, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1014 = __riscv_vreinterpret_v_i32m1_i8m1(v1013);
    uint32_t v1015 = v940 >> 14u;
    uint32_t v1016 = v1015 & 127u;
    int v1017 = (int) v1016;
    const uint8_t v1018 = weft_iq3xxs_ksigns[v1017];
    int v1019 = (int) v1018;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1020 = __riscv_vmv_v_x_u8m1(v1019, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1021 = __riscv_vand_vv_u8m1(v1020, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1022 = __riscv_vmsne_vx_u8m1_b8(v1021, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1023 = __riscv_vneg_v_i8m1(v1014, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1024 = __riscv_vmerge_vvm_i8m1(v1014, v1023, v1022, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v1025 = __riscv_vsext_vf4_i32m4(v1024, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v1026 = __riscv_vfcvt_f_x_v_f32m4(v1025, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v1027 = __riscv_vfmul_vf_f32m4(v1026, v946, 8);
    float* v1028 = v13 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v1028, v1027, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_group
    uint16_t v1029[2];
    const uint8_t v1030 = v947[6];
    int v1031 = (int) v1030;
    int v1032 = v1031 << 2;
    uint16_t v1033 = (uint16_t) v1032;
    v1029[0] = v1033;
    const uint8_t v1034 = v947[7];
    int v1035 = (int) v1034;
    int v1036 = v1035 << 2;
    uint16_t v1037 = (uint16_t) v1036;
    v1029[1] = v1037;
    uint16_t* v1038 = &v1029[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1039 = __riscv_vle16_v_u16mf2(v1038, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1040 = __riscv_vluxei16_v_i32m1(v7, v1039, 2);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1041 = __riscv_vreinterpret_v_i32m1_i8m1(v1040);
    uint32_t v1042 = v940 >> 21u;
    uint32_t v1043 = v1042 & 127u;
    int v1044 = (int) v1043;
    const uint8_t v1045 = weft_iq3xxs_ksigns[v1044];
    int v1046 = (int) v1045;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1047 = __riscv_vmv_v_x_u8m1(v1046, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1048 = __riscv_vand_vv_u8m1(v1047, v6, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1049 = __riscv_vmsne_vx_u8m1_b8(v1048, 0, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1050 = __riscv_vneg_v_i8m1(v1041, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1051 = __riscv_vmerge_vvm_i8m1(v1041, v1050, v1049, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v1052 = __riscv_vsext_vf4_i32m4(v1051, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v1053 = __riscv_vfcvt_f_x_v_f32m4(v1052, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v1054 = __riscv_vfmul_vf_f32m4(v1053, v946, 8);
    float* v1055 = v13 + 248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v1055, v1054, 8);
  }
  return;
}


