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
  static const uint8_t weft_iq3xxs_kmask_lo[4] = {1, 2, 4, 8};
  static const uint8_t weft_iq3xxs_kmask_hi[4] = {16, 32, 64, 128};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_lo_load
  vuint8mf4_t v6 = __riscv_vle8_v_u8mf4(weft_iq3xxs_kmask_lo, 4);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_hi_load
  vuint8mf4_t v7 = __riscv_vle8_v_u8mf4(weft_iq3xxs_kmask_hi, 4);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_u8_view
  const uint8_t* v8 = (const uint8_t*) weft_iq3xxs_grid;
  for (size_t v9 = 0; v9 < v5; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v10 = v9 * 98;
    const uint8_t* v11 = v2 + v10;
    size_t v12 = v9 * 256;
    float* v13 = v3 + v12;
    float* v14 = (float*) v13;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v11);
    const uint8_t* v16 = v11 + 2;
    const uint8_t* v17 = (const uint8_t*) v16;
    const uint8_t* v18 = v11 + 66;
    const uint8_t* v19 = (const uint8_t*) v18;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v20 = v19[0];
    uint32_t v21 = (uint32_t) v20;
    const uint8_t v22 = v19[1];
    uint32_t v23 = (uint32_t) v22;
    uint32_t v24 = v23 << 8u;
    uint32_t v25 = v21 | v24;
    const uint8_t v26 = v19[2];
    uint32_t v27 = (uint32_t) v26;
    uint32_t v28 = v27 << 16u;
    uint32_t v29 = v25 | v28;
    const uint8_t v30 = v19[3];
    uint32_t v31 = (uint32_t) v30;
    uint32_t v32 = v31 << 24u;
    uint32_t v33 = v29 | v32;
    uint32_t v34 = v33 >> 28u;
    int v35 = (int) v34;
    float v36 = (float) v35;
    float v37 = 0.5f + v36;
    float v38 = v15 * v37;
    float v39 = v38 * 0.5f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint32_t v40 = v33 >> 0u;
    uint32_t v41 = v40 & 127u;
    int v42 = (int) v41;
    const uint8_t v43 = weft_iq3xxs_ksigns[v42];
    uint8_t v44 = (uint8_t) v43;
    const uint8_t v45 = v17[0];
    int v46 = (int) v45;
    size_t v47 = (size_t) v46;
    size_t v48 = v47 * 4;
    const uint8_t* v49 = v8 + v48;
    const int8_t* v50 = (const int8_t*) v49;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v51 = __riscv_vle8_v_i8mf4(v50, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v52 = __riscv_vand_vx_u8mf4(v6, v44, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v53 = __riscv_vmsne_vx_u8mf4_b32(v52, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v54 = __riscv_vneg_v_i8mf4(v51, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v55 = __riscv_vmerge_vvm_i8mf4(v51, v54, v53, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v56 = __riscv_vsext_vf4_i32m1(v55, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v57 = __riscv_vfcvt_f_x_v_f32m1(v56, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v58 = __riscv_vfmul_vf_f32m1(v57, v39, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v14, v58, 4);
    const uint8_t v59 = v17[1];
    int v60 = (int) v59;
    size_t v61 = (size_t) v60;
    size_t v62 = v61 * 4;
    const uint8_t* v63 = v8 + v62;
    const int8_t* v64 = (const int8_t*) v63;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v65 = __riscv_vle8_v_i8mf4(v64, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v66 = __riscv_vand_vx_u8mf4(v7, v44, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v67 = __riscv_vmsne_vx_u8mf4_b32(v66, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v68 = __riscv_vneg_v_i8mf4(v65, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v69 = __riscv_vmerge_vvm_i8mf4(v65, v68, v67, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v70 = __riscv_vsext_vf4_i32m1(v69, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v71 = __riscv_vfcvt_f_x_v_f32m1(v70, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v72 = __riscv_vfmul_vf_f32m1(v71, v39, 4);
    float* v73 = v14 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v73, v72, 4);
    uint32_t v74 = v33 >> 7u;
    uint32_t v75 = v74 & 127u;
    int v76 = (int) v75;
    const uint8_t v77 = weft_iq3xxs_ksigns[v76];
    uint8_t v78 = (uint8_t) v77;
    const uint8_t v79 = v17[2];
    int v80 = (int) v79;
    size_t v81 = (size_t) v80;
    size_t v82 = v81 * 4;
    const uint8_t* v83 = v8 + v82;
    const int8_t* v84 = (const int8_t*) v83;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v85 = __riscv_vle8_v_i8mf4(v84, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v86 = __riscv_vand_vx_u8mf4(v6, v78, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v87 = __riscv_vmsne_vx_u8mf4_b32(v86, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v88 = __riscv_vneg_v_i8mf4(v85, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v89 = __riscv_vmerge_vvm_i8mf4(v85, v88, v87, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v90 = __riscv_vsext_vf4_i32m1(v89, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v91 = __riscv_vfcvt_f_x_v_f32m1(v90, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v92 = __riscv_vfmul_vf_f32m1(v91, v39, 4);
    float* v93 = v14 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v93, v92, 4);
    const uint8_t v94 = v17[3];
    int v95 = (int) v94;
    size_t v96 = (size_t) v95;
    size_t v97 = v96 * 4;
    const uint8_t* v98 = v8 + v97;
    const int8_t* v99 = (const int8_t*) v98;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v100 = __riscv_vle8_v_i8mf4(v99, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v101 = __riscv_vand_vx_u8mf4(v7, v78, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v102 = __riscv_vmsne_vx_u8mf4_b32(v101, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v103 = __riscv_vneg_v_i8mf4(v100, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v104 = __riscv_vmerge_vvm_i8mf4(v100, v103, v102, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v105 = __riscv_vsext_vf4_i32m1(v104, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v106 = __riscv_vfcvt_f_x_v_f32m1(v105, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v107 = __riscv_vfmul_vf_f32m1(v106, v39, 4);
    float* v108 = v14 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v108, v107, 4);
    uint32_t v109 = v33 >> 14u;
    uint32_t v110 = v109 & 127u;
    int v111 = (int) v110;
    const uint8_t v112 = weft_iq3xxs_ksigns[v111];
    uint8_t v113 = (uint8_t) v112;
    const uint8_t v114 = v17[4];
    int v115 = (int) v114;
    size_t v116 = (size_t) v115;
    size_t v117 = v116 * 4;
    const uint8_t* v118 = v8 + v117;
    const int8_t* v119 = (const int8_t*) v118;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v120 = __riscv_vle8_v_i8mf4(v119, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v121 = __riscv_vand_vx_u8mf4(v6, v113, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v122 = __riscv_vmsne_vx_u8mf4_b32(v121, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v123 = __riscv_vneg_v_i8mf4(v120, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v124 = __riscv_vmerge_vvm_i8mf4(v120, v123, v122, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v125 = __riscv_vsext_vf4_i32m1(v124, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v126 = __riscv_vfcvt_f_x_v_f32m1(v125, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v127 = __riscv_vfmul_vf_f32m1(v126, v39, 4);
    float* v128 = v14 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v128, v127, 4);
    const uint8_t v129 = v17[5];
    int v130 = (int) v129;
    size_t v131 = (size_t) v130;
    size_t v132 = v131 * 4;
    const uint8_t* v133 = v8 + v132;
    const int8_t* v134 = (const int8_t*) v133;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v135 = __riscv_vle8_v_i8mf4(v134, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v136 = __riscv_vand_vx_u8mf4(v7, v113, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v137 = __riscv_vmsne_vx_u8mf4_b32(v136, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v138 = __riscv_vneg_v_i8mf4(v135, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v139 = __riscv_vmerge_vvm_i8mf4(v135, v138, v137, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v140 = __riscv_vsext_vf4_i32m1(v139, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v141 = __riscv_vfcvt_f_x_v_f32m1(v140, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v142 = __riscv_vfmul_vf_f32m1(v141, v39, 4);
    float* v143 = v14 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v143, v142, 4);
    uint32_t v144 = v33 >> 21u;
    uint32_t v145 = v144 & 127u;
    int v146 = (int) v145;
    const uint8_t v147 = weft_iq3xxs_ksigns[v146];
    uint8_t v148 = (uint8_t) v147;
    const uint8_t v149 = v17[6];
    int v150 = (int) v149;
    size_t v151 = (size_t) v150;
    size_t v152 = v151 * 4;
    const uint8_t* v153 = v8 + v152;
    const int8_t* v154 = (const int8_t*) v153;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v155 = __riscv_vle8_v_i8mf4(v154, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v156 = __riscv_vand_vx_u8mf4(v6, v148, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v157 = __riscv_vmsne_vx_u8mf4_b32(v156, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v158 = __riscv_vneg_v_i8mf4(v155, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v159 = __riscv_vmerge_vvm_i8mf4(v155, v158, v157, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v160 = __riscv_vsext_vf4_i32m1(v159, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v161 = __riscv_vfcvt_f_x_v_f32m1(v160, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v162 = __riscv_vfmul_vf_f32m1(v161, v39, 4);
    float* v163 = v14 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v163, v162, 4);
    const uint8_t v164 = v17[7];
    int v165 = (int) v164;
    size_t v166 = (size_t) v165;
    size_t v167 = v166 * 4;
    const uint8_t* v168 = v8 + v167;
    const int8_t* v169 = (const int8_t*) v168;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v170 = __riscv_vle8_v_i8mf4(v169, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v171 = __riscv_vand_vx_u8mf4(v7, v148, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v172 = __riscv_vmsne_vx_u8mf4_b32(v171, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v173 = __riscv_vneg_v_i8mf4(v170, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v174 = __riscv_vmerge_vvm_i8mf4(v170, v173, v172, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v175 = __riscv_vsext_vf4_i32m1(v174, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v176 = __riscv_vfcvt_f_x_v_f32m1(v175, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v177 = __riscv_vfmul_vf_f32m1(v176, v39, 4);
    float* v178 = v14 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v178, v177, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v179 = v19 + 4;
    const uint8_t v180 = v179[0];
    uint32_t v181 = (uint32_t) v180;
    const uint8_t v182 = v179[1];
    uint32_t v183 = (uint32_t) v182;
    uint32_t v184 = v183 << 8u;
    uint32_t v185 = v181 | v184;
    const uint8_t v186 = v179[2];
    uint32_t v187 = (uint32_t) v186;
    uint32_t v188 = v187 << 16u;
    uint32_t v189 = v185 | v188;
    const uint8_t v190 = v179[3];
    uint32_t v191 = (uint32_t) v190;
    uint32_t v192 = v191 << 24u;
    uint32_t v193 = v189 | v192;
    uint32_t v194 = v193 >> 28u;
    int v195 = (int) v194;
    float v196 = (float) v195;
    float v197 = 0.5f + v196;
    float v198 = v15 * v197;
    float v199 = v198 * 0.5f;
    const uint8_t* v200 = v17 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v201 = v14 + 32;
    uint32_t v202 = v193 >> 0u;
    uint32_t v203 = v202 & 127u;
    int v204 = (int) v203;
    const uint8_t v205 = weft_iq3xxs_ksigns[v204];
    uint8_t v206 = (uint8_t) v205;
    const uint8_t v207 = v200[0];
    int v208 = (int) v207;
    size_t v209 = (size_t) v208;
    size_t v210 = v209 * 4;
    const uint8_t* v211 = v8 + v210;
    const int8_t* v212 = (const int8_t*) v211;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v213 = __riscv_vle8_v_i8mf4(v212, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v214 = __riscv_vand_vx_u8mf4(v6, v206, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v215 = __riscv_vmsne_vx_u8mf4_b32(v214, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v216 = __riscv_vneg_v_i8mf4(v213, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v217 = __riscv_vmerge_vvm_i8mf4(v213, v216, v215, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v218 = __riscv_vsext_vf4_i32m1(v217, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v219 = __riscv_vfcvt_f_x_v_f32m1(v218, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v220 = __riscv_vfmul_vf_f32m1(v219, v199, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v201, v220, 4);
    const uint8_t v221 = v200[1];
    int v222 = (int) v221;
    size_t v223 = (size_t) v222;
    size_t v224 = v223 * 4;
    const uint8_t* v225 = v8 + v224;
    const int8_t* v226 = (const int8_t*) v225;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v227 = __riscv_vle8_v_i8mf4(v226, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v228 = __riscv_vand_vx_u8mf4(v7, v206, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v229 = __riscv_vmsne_vx_u8mf4_b32(v228, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v230 = __riscv_vneg_v_i8mf4(v227, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v231 = __riscv_vmerge_vvm_i8mf4(v227, v230, v229, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v232 = __riscv_vsext_vf4_i32m1(v231, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v233 = __riscv_vfcvt_f_x_v_f32m1(v232, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v234 = __riscv_vfmul_vf_f32m1(v233, v199, 4);
    float* v235 = v201 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v235, v234, 4);
    uint32_t v236 = v193 >> 7u;
    uint32_t v237 = v236 & 127u;
    int v238 = (int) v237;
    const uint8_t v239 = weft_iq3xxs_ksigns[v238];
    uint8_t v240 = (uint8_t) v239;
    const uint8_t v241 = v200[2];
    int v242 = (int) v241;
    size_t v243 = (size_t) v242;
    size_t v244 = v243 * 4;
    const uint8_t* v245 = v8 + v244;
    const int8_t* v246 = (const int8_t*) v245;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v247 = __riscv_vle8_v_i8mf4(v246, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v248 = __riscv_vand_vx_u8mf4(v6, v240, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v249 = __riscv_vmsne_vx_u8mf4_b32(v248, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v250 = __riscv_vneg_v_i8mf4(v247, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v251 = __riscv_vmerge_vvm_i8mf4(v247, v250, v249, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v252 = __riscv_vsext_vf4_i32m1(v251, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v253 = __riscv_vfcvt_f_x_v_f32m1(v252, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v254 = __riscv_vfmul_vf_f32m1(v253, v199, 4);
    float* v255 = v201 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v255, v254, 4);
    const uint8_t v256 = v200[3];
    int v257 = (int) v256;
    size_t v258 = (size_t) v257;
    size_t v259 = v258 * 4;
    const uint8_t* v260 = v8 + v259;
    const int8_t* v261 = (const int8_t*) v260;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v262 = __riscv_vle8_v_i8mf4(v261, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v263 = __riscv_vand_vx_u8mf4(v7, v240, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v264 = __riscv_vmsne_vx_u8mf4_b32(v263, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v265 = __riscv_vneg_v_i8mf4(v262, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v266 = __riscv_vmerge_vvm_i8mf4(v262, v265, v264, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v267 = __riscv_vsext_vf4_i32m1(v266, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v268 = __riscv_vfcvt_f_x_v_f32m1(v267, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v269 = __riscv_vfmul_vf_f32m1(v268, v199, 4);
    float* v270 = v201 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v270, v269, 4);
    uint32_t v271 = v193 >> 14u;
    uint32_t v272 = v271 & 127u;
    int v273 = (int) v272;
    const uint8_t v274 = weft_iq3xxs_ksigns[v273];
    uint8_t v275 = (uint8_t) v274;
    const uint8_t v276 = v200[4];
    int v277 = (int) v276;
    size_t v278 = (size_t) v277;
    size_t v279 = v278 * 4;
    const uint8_t* v280 = v8 + v279;
    const int8_t* v281 = (const int8_t*) v280;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v282 = __riscv_vle8_v_i8mf4(v281, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v283 = __riscv_vand_vx_u8mf4(v6, v275, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v284 = __riscv_vmsne_vx_u8mf4_b32(v283, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v285 = __riscv_vneg_v_i8mf4(v282, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v286 = __riscv_vmerge_vvm_i8mf4(v282, v285, v284, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v287 = __riscv_vsext_vf4_i32m1(v286, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v288 = __riscv_vfcvt_f_x_v_f32m1(v287, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v289 = __riscv_vfmul_vf_f32m1(v288, v199, 4);
    float* v290 = v201 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v290, v289, 4);
    const uint8_t v291 = v200[5];
    int v292 = (int) v291;
    size_t v293 = (size_t) v292;
    size_t v294 = v293 * 4;
    const uint8_t* v295 = v8 + v294;
    const int8_t* v296 = (const int8_t*) v295;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v297 = __riscv_vle8_v_i8mf4(v296, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v298 = __riscv_vand_vx_u8mf4(v7, v275, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v299 = __riscv_vmsne_vx_u8mf4_b32(v298, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v300 = __riscv_vneg_v_i8mf4(v297, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v301 = __riscv_vmerge_vvm_i8mf4(v297, v300, v299, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v302 = __riscv_vsext_vf4_i32m1(v301, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v303 = __riscv_vfcvt_f_x_v_f32m1(v302, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v304 = __riscv_vfmul_vf_f32m1(v303, v199, 4);
    float* v305 = v201 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v305, v304, 4);
    uint32_t v306 = v193 >> 21u;
    uint32_t v307 = v306 & 127u;
    int v308 = (int) v307;
    const uint8_t v309 = weft_iq3xxs_ksigns[v308];
    uint8_t v310 = (uint8_t) v309;
    const uint8_t v311 = v200[6];
    int v312 = (int) v311;
    size_t v313 = (size_t) v312;
    size_t v314 = v313 * 4;
    const uint8_t* v315 = v8 + v314;
    const int8_t* v316 = (const int8_t*) v315;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v317 = __riscv_vle8_v_i8mf4(v316, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v318 = __riscv_vand_vx_u8mf4(v6, v310, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v319 = __riscv_vmsne_vx_u8mf4_b32(v318, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v320 = __riscv_vneg_v_i8mf4(v317, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v321 = __riscv_vmerge_vvm_i8mf4(v317, v320, v319, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v322 = __riscv_vsext_vf4_i32m1(v321, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v323 = __riscv_vfcvt_f_x_v_f32m1(v322, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v324 = __riscv_vfmul_vf_f32m1(v323, v199, 4);
    float* v325 = v201 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v325, v324, 4);
    const uint8_t v326 = v200[7];
    int v327 = (int) v326;
    size_t v328 = (size_t) v327;
    size_t v329 = v328 * 4;
    const uint8_t* v330 = v8 + v329;
    const int8_t* v331 = (const int8_t*) v330;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v332 = __riscv_vle8_v_i8mf4(v331, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v333 = __riscv_vand_vx_u8mf4(v7, v310, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v334 = __riscv_vmsne_vx_u8mf4_b32(v333, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v335 = __riscv_vneg_v_i8mf4(v332, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v336 = __riscv_vmerge_vvm_i8mf4(v332, v335, v334, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v337 = __riscv_vsext_vf4_i32m1(v336, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v338 = __riscv_vfcvt_f_x_v_f32m1(v337, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v339 = __riscv_vfmul_vf_f32m1(v338, v199, 4);
    float* v340 = v201 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v340, v339, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v341 = v19 + 8;
    const uint8_t v342 = v341[0];
    uint32_t v343 = (uint32_t) v342;
    const uint8_t v344 = v341[1];
    uint32_t v345 = (uint32_t) v344;
    uint32_t v346 = v345 << 8u;
    uint32_t v347 = v343 | v346;
    const uint8_t v348 = v341[2];
    uint32_t v349 = (uint32_t) v348;
    uint32_t v350 = v349 << 16u;
    uint32_t v351 = v347 | v350;
    const uint8_t v352 = v341[3];
    uint32_t v353 = (uint32_t) v352;
    uint32_t v354 = v353 << 24u;
    uint32_t v355 = v351 | v354;
    uint32_t v356 = v355 >> 28u;
    int v357 = (int) v356;
    float v358 = (float) v357;
    float v359 = 0.5f + v358;
    float v360 = v15 * v359;
    float v361 = v360 * 0.5f;
    const uint8_t* v362 = v17 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v363 = v14 + 64;
    uint32_t v364 = v355 >> 0u;
    uint32_t v365 = v364 & 127u;
    int v366 = (int) v365;
    const uint8_t v367 = weft_iq3xxs_ksigns[v366];
    uint8_t v368 = (uint8_t) v367;
    const uint8_t v369 = v362[0];
    int v370 = (int) v369;
    size_t v371 = (size_t) v370;
    size_t v372 = v371 * 4;
    const uint8_t* v373 = v8 + v372;
    const int8_t* v374 = (const int8_t*) v373;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v375 = __riscv_vle8_v_i8mf4(v374, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v376 = __riscv_vand_vx_u8mf4(v6, v368, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v377 = __riscv_vmsne_vx_u8mf4_b32(v376, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v378 = __riscv_vneg_v_i8mf4(v375, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v379 = __riscv_vmerge_vvm_i8mf4(v375, v378, v377, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v380 = __riscv_vsext_vf4_i32m1(v379, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v381 = __riscv_vfcvt_f_x_v_f32m1(v380, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v382 = __riscv_vfmul_vf_f32m1(v381, v361, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v363, v382, 4);
    const uint8_t v383 = v362[1];
    int v384 = (int) v383;
    size_t v385 = (size_t) v384;
    size_t v386 = v385 * 4;
    const uint8_t* v387 = v8 + v386;
    const int8_t* v388 = (const int8_t*) v387;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v389 = __riscv_vle8_v_i8mf4(v388, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v390 = __riscv_vand_vx_u8mf4(v7, v368, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v391 = __riscv_vmsne_vx_u8mf4_b32(v390, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v392 = __riscv_vneg_v_i8mf4(v389, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v393 = __riscv_vmerge_vvm_i8mf4(v389, v392, v391, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v394 = __riscv_vsext_vf4_i32m1(v393, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v395 = __riscv_vfcvt_f_x_v_f32m1(v394, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v396 = __riscv_vfmul_vf_f32m1(v395, v361, 4);
    float* v397 = v363 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v397, v396, 4);
    uint32_t v398 = v355 >> 7u;
    uint32_t v399 = v398 & 127u;
    int v400 = (int) v399;
    const uint8_t v401 = weft_iq3xxs_ksigns[v400];
    uint8_t v402 = (uint8_t) v401;
    const uint8_t v403 = v362[2];
    int v404 = (int) v403;
    size_t v405 = (size_t) v404;
    size_t v406 = v405 * 4;
    const uint8_t* v407 = v8 + v406;
    const int8_t* v408 = (const int8_t*) v407;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v409 = __riscv_vle8_v_i8mf4(v408, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v410 = __riscv_vand_vx_u8mf4(v6, v402, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v411 = __riscv_vmsne_vx_u8mf4_b32(v410, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v412 = __riscv_vneg_v_i8mf4(v409, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v413 = __riscv_vmerge_vvm_i8mf4(v409, v412, v411, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v414 = __riscv_vsext_vf4_i32m1(v413, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v415 = __riscv_vfcvt_f_x_v_f32m1(v414, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v416 = __riscv_vfmul_vf_f32m1(v415, v361, 4);
    float* v417 = v363 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v417, v416, 4);
    const uint8_t v418 = v362[3];
    int v419 = (int) v418;
    size_t v420 = (size_t) v419;
    size_t v421 = v420 * 4;
    const uint8_t* v422 = v8 + v421;
    const int8_t* v423 = (const int8_t*) v422;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v424 = __riscv_vle8_v_i8mf4(v423, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v425 = __riscv_vand_vx_u8mf4(v7, v402, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v426 = __riscv_vmsne_vx_u8mf4_b32(v425, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v427 = __riscv_vneg_v_i8mf4(v424, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v428 = __riscv_vmerge_vvm_i8mf4(v424, v427, v426, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v429 = __riscv_vsext_vf4_i32m1(v428, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v430 = __riscv_vfcvt_f_x_v_f32m1(v429, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v431 = __riscv_vfmul_vf_f32m1(v430, v361, 4);
    float* v432 = v363 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v432, v431, 4);
    uint32_t v433 = v355 >> 14u;
    uint32_t v434 = v433 & 127u;
    int v435 = (int) v434;
    const uint8_t v436 = weft_iq3xxs_ksigns[v435];
    uint8_t v437 = (uint8_t) v436;
    const uint8_t v438 = v362[4];
    int v439 = (int) v438;
    size_t v440 = (size_t) v439;
    size_t v441 = v440 * 4;
    const uint8_t* v442 = v8 + v441;
    const int8_t* v443 = (const int8_t*) v442;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v444 = __riscv_vle8_v_i8mf4(v443, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v445 = __riscv_vand_vx_u8mf4(v6, v437, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v446 = __riscv_vmsne_vx_u8mf4_b32(v445, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v447 = __riscv_vneg_v_i8mf4(v444, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v448 = __riscv_vmerge_vvm_i8mf4(v444, v447, v446, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v449 = __riscv_vsext_vf4_i32m1(v448, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v450 = __riscv_vfcvt_f_x_v_f32m1(v449, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v451 = __riscv_vfmul_vf_f32m1(v450, v361, 4);
    float* v452 = v363 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v452, v451, 4);
    const uint8_t v453 = v362[5];
    int v454 = (int) v453;
    size_t v455 = (size_t) v454;
    size_t v456 = v455 * 4;
    const uint8_t* v457 = v8 + v456;
    const int8_t* v458 = (const int8_t*) v457;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v459 = __riscv_vle8_v_i8mf4(v458, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v460 = __riscv_vand_vx_u8mf4(v7, v437, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v461 = __riscv_vmsne_vx_u8mf4_b32(v460, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v462 = __riscv_vneg_v_i8mf4(v459, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v463 = __riscv_vmerge_vvm_i8mf4(v459, v462, v461, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v464 = __riscv_vsext_vf4_i32m1(v463, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v465 = __riscv_vfcvt_f_x_v_f32m1(v464, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v466 = __riscv_vfmul_vf_f32m1(v465, v361, 4);
    float* v467 = v363 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v467, v466, 4);
    uint32_t v468 = v355 >> 21u;
    uint32_t v469 = v468 & 127u;
    int v470 = (int) v469;
    const uint8_t v471 = weft_iq3xxs_ksigns[v470];
    uint8_t v472 = (uint8_t) v471;
    const uint8_t v473 = v362[6];
    int v474 = (int) v473;
    size_t v475 = (size_t) v474;
    size_t v476 = v475 * 4;
    const uint8_t* v477 = v8 + v476;
    const int8_t* v478 = (const int8_t*) v477;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v479 = __riscv_vle8_v_i8mf4(v478, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v480 = __riscv_vand_vx_u8mf4(v6, v472, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v481 = __riscv_vmsne_vx_u8mf4_b32(v480, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v482 = __riscv_vneg_v_i8mf4(v479, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v483 = __riscv_vmerge_vvm_i8mf4(v479, v482, v481, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v484 = __riscv_vsext_vf4_i32m1(v483, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v485 = __riscv_vfcvt_f_x_v_f32m1(v484, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v486 = __riscv_vfmul_vf_f32m1(v485, v361, 4);
    float* v487 = v363 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v487, v486, 4);
    const uint8_t v488 = v362[7];
    int v489 = (int) v488;
    size_t v490 = (size_t) v489;
    size_t v491 = v490 * 4;
    const uint8_t* v492 = v8 + v491;
    const int8_t* v493 = (const int8_t*) v492;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v494 = __riscv_vle8_v_i8mf4(v493, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v495 = __riscv_vand_vx_u8mf4(v7, v472, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v496 = __riscv_vmsne_vx_u8mf4_b32(v495, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v497 = __riscv_vneg_v_i8mf4(v494, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v498 = __riscv_vmerge_vvm_i8mf4(v494, v497, v496, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v499 = __riscv_vsext_vf4_i32m1(v498, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v500 = __riscv_vfcvt_f_x_v_f32m1(v499, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v501 = __riscv_vfmul_vf_f32m1(v500, v361, 4);
    float* v502 = v363 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v502, v501, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v503 = v19 + 12;
    const uint8_t v504 = v503[0];
    uint32_t v505 = (uint32_t) v504;
    const uint8_t v506 = v503[1];
    uint32_t v507 = (uint32_t) v506;
    uint32_t v508 = v507 << 8u;
    uint32_t v509 = v505 | v508;
    const uint8_t v510 = v503[2];
    uint32_t v511 = (uint32_t) v510;
    uint32_t v512 = v511 << 16u;
    uint32_t v513 = v509 | v512;
    const uint8_t v514 = v503[3];
    uint32_t v515 = (uint32_t) v514;
    uint32_t v516 = v515 << 24u;
    uint32_t v517 = v513 | v516;
    uint32_t v518 = v517 >> 28u;
    int v519 = (int) v518;
    float v520 = (float) v519;
    float v521 = 0.5f + v520;
    float v522 = v15 * v521;
    float v523 = v522 * 0.5f;
    const uint8_t* v524 = v17 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v525 = v14 + 96;
    uint32_t v526 = v517 >> 0u;
    uint32_t v527 = v526 & 127u;
    int v528 = (int) v527;
    const uint8_t v529 = weft_iq3xxs_ksigns[v528];
    uint8_t v530 = (uint8_t) v529;
    const uint8_t v531 = v524[0];
    int v532 = (int) v531;
    size_t v533 = (size_t) v532;
    size_t v534 = v533 * 4;
    const uint8_t* v535 = v8 + v534;
    const int8_t* v536 = (const int8_t*) v535;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v537 = __riscv_vle8_v_i8mf4(v536, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v538 = __riscv_vand_vx_u8mf4(v6, v530, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v539 = __riscv_vmsne_vx_u8mf4_b32(v538, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v540 = __riscv_vneg_v_i8mf4(v537, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v541 = __riscv_vmerge_vvm_i8mf4(v537, v540, v539, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v542 = __riscv_vsext_vf4_i32m1(v541, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v543 = __riscv_vfcvt_f_x_v_f32m1(v542, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v544 = __riscv_vfmul_vf_f32m1(v543, v523, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v525, v544, 4);
    const uint8_t v545 = v524[1];
    int v546 = (int) v545;
    size_t v547 = (size_t) v546;
    size_t v548 = v547 * 4;
    const uint8_t* v549 = v8 + v548;
    const int8_t* v550 = (const int8_t*) v549;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v551 = __riscv_vle8_v_i8mf4(v550, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v552 = __riscv_vand_vx_u8mf4(v7, v530, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v553 = __riscv_vmsne_vx_u8mf4_b32(v552, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v554 = __riscv_vneg_v_i8mf4(v551, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v555 = __riscv_vmerge_vvm_i8mf4(v551, v554, v553, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v556 = __riscv_vsext_vf4_i32m1(v555, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v557 = __riscv_vfcvt_f_x_v_f32m1(v556, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v558 = __riscv_vfmul_vf_f32m1(v557, v523, 4);
    float* v559 = v525 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v559, v558, 4);
    uint32_t v560 = v517 >> 7u;
    uint32_t v561 = v560 & 127u;
    int v562 = (int) v561;
    const uint8_t v563 = weft_iq3xxs_ksigns[v562];
    uint8_t v564 = (uint8_t) v563;
    const uint8_t v565 = v524[2];
    int v566 = (int) v565;
    size_t v567 = (size_t) v566;
    size_t v568 = v567 * 4;
    const uint8_t* v569 = v8 + v568;
    const int8_t* v570 = (const int8_t*) v569;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v571 = __riscv_vle8_v_i8mf4(v570, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v572 = __riscv_vand_vx_u8mf4(v6, v564, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v573 = __riscv_vmsne_vx_u8mf4_b32(v572, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v574 = __riscv_vneg_v_i8mf4(v571, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v575 = __riscv_vmerge_vvm_i8mf4(v571, v574, v573, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v576 = __riscv_vsext_vf4_i32m1(v575, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v577 = __riscv_vfcvt_f_x_v_f32m1(v576, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v578 = __riscv_vfmul_vf_f32m1(v577, v523, 4);
    float* v579 = v525 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v579, v578, 4);
    const uint8_t v580 = v524[3];
    int v581 = (int) v580;
    size_t v582 = (size_t) v581;
    size_t v583 = v582 * 4;
    const uint8_t* v584 = v8 + v583;
    const int8_t* v585 = (const int8_t*) v584;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v586 = __riscv_vle8_v_i8mf4(v585, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v587 = __riscv_vand_vx_u8mf4(v7, v564, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v588 = __riscv_vmsne_vx_u8mf4_b32(v587, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v589 = __riscv_vneg_v_i8mf4(v586, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v590 = __riscv_vmerge_vvm_i8mf4(v586, v589, v588, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v591 = __riscv_vsext_vf4_i32m1(v590, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v592 = __riscv_vfcvt_f_x_v_f32m1(v591, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v593 = __riscv_vfmul_vf_f32m1(v592, v523, 4);
    float* v594 = v525 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v594, v593, 4);
    uint32_t v595 = v517 >> 14u;
    uint32_t v596 = v595 & 127u;
    int v597 = (int) v596;
    const uint8_t v598 = weft_iq3xxs_ksigns[v597];
    uint8_t v599 = (uint8_t) v598;
    const uint8_t v600 = v524[4];
    int v601 = (int) v600;
    size_t v602 = (size_t) v601;
    size_t v603 = v602 * 4;
    const uint8_t* v604 = v8 + v603;
    const int8_t* v605 = (const int8_t*) v604;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v606 = __riscv_vle8_v_i8mf4(v605, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v607 = __riscv_vand_vx_u8mf4(v6, v599, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v608 = __riscv_vmsne_vx_u8mf4_b32(v607, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v609 = __riscv_vneg_v_i8mf4(v606, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v610 = __riscv_vmerge_vvm_i8mf4(v606, v609, v608, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v611 = __riscv_vsext_vf4_i32m1(v610, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v612 = __riscv_vfcvt_f_x_v_f32m1(v611, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v613 = __riscv_vfmul_vf_f32m1(v612, v523, 4);
    float* v614 = v525 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v614, v613, 4);
    const uint8_t v615 = v524[5];
    int v616 = (int) v615;
    size_t v617 = (size_t) v616;
    size_t v618 = v617 * 4;
    const uint8_t* v619 = v8 + v618;
    const int8_t* v620 = (const int8_t*) v619;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v621 = __riscv_vle8_v_i8mf4(v620, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v622 = __riscv_vand_vx_u8mf4(v7, v599, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v623 = __riscv_vmsne_vx_u8mf4_b32(v622, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v624 = __riscv_vneg_v_i8mf4(v621, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v625 = __riscv_vmerge_vvm_i8mf4(v621, v624, v623, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v626 = __riscv_vsext_vf4_i32m1(v625, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v627 = __riscv_vfcvt_f_x_v_f32m1(v626, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v628 = __riscv_vfmul_vf_f32m1(v627, v523, 4);
    float* v629 = v525 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v629, v628, 4);
    uint32_t v630 = v517 >> 21u;
    uint32_t v631 = v630 & 127u;
    int v632 = (int) v631;
    const uint8_t v633 = weft_iq3xxs_ksigns[v632];
    uint8_t v634 = (uint8_t) v633;
    const uint8_t v635 = v524[6];
    int v636 = (int) v635;
    size_t v637 = (size_t) v636;
    size_t v638 = v637 * 4;
    const uint8_t* v639 = v8 + v638;
    const int8_t* v640 = (const int8_t*) v639;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v641 = __riscv_vle8_v_i8mf4(v640, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v642 = __riscv_vand_vx_u8mf4(v6, v634, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v643 = __riscv_vmsne_vx_u8mf4_b32(v642, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v644 = __riscv_vneg_v_i8mf4(v641, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v645 = __riscv_vmerge_vvm_i8mf4(v641, v644, v643, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v646 = __riscv_vsext_vf4_i32m1(v645, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v647 = __riscv_vfcvt_f_x_v_f32m1(v646, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v648 = __riscv_vfmul_vf_f32m1(v647, v523, 4);
    float* v649 = v525 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v649, v648, 4);
    const uint8_t v650 = v524[7];
    int v651 = (int) v650;
    size_t v652 = (size_t) v651;
    size_t v653 = v652 * 4;
    const uint8_t* v654 = v8 + v653;
    const int8_t* v655 = (const int8_t*) v654;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v656 = __riscv_vle8_v_i8mf4(v655, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v657 = __riscv_vand_vx_u8mf4(v7, v634, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v658 = __riscv_vmsne_vx_u8mf4_b32(v657, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v659 = __riscv_vneg_v_i8mf4(v656, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v660 = __riscv_vmerge_vvm_i8mf4(v656, v659, v658, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v661 = __riscv_vsext_vf4_i32m1(v660, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v662 = __riscv_vfcvt_f_x_v_f32m1(v661, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v663 = __riscv_vfmul_vf_f32m1(v662, v523, 4);
    float* v664 = v525 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v664, v663, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v665 = v19 + 16;
    const uint8_t v666 = v665[0];
    uint32_t v667 = (uint32_t) v666;
    const uint8_t v668 = v665[1];
    uint32_t v669 = (uint32_t) v668;
    uint32_t v670 = v669 << 8u;
    uint32_t v671 = v667 | v670;
    const uint8_t v672 = v665[2];
    uint32_t v673 = (uint32_t) v672;
    uint32_t v674 = v673 << 16u;
    uint32_t v675 = v671 | v674;
    const uint8_t v676 = v665[3];
    uint32_t v677 = (uint32_t) v676;
    uint32_t v678 = v677 << 24u;
    uint32_t v679 = v675 | v678;
    uint32_t v680 = v679 >> 28u;
    int v681 = (int) v680;
    float v682 = (float) v681;
    float v683 = 0.5f + v682;
    float v684 = v15 * v683;
    float v685 = v684 * 0.5f;
    const uint8_t* v686 = v17 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v687 = v14 + 128;
    uint32_t v688 = v679 >> 0u;
    uint32_t v689 = v688 & 127u;
    int v690 = (int) v689;
    const uint8_t v691 = weft_iq3xxs_ksigns[v690];
    uint8_t v692 = (uint8_t) v691;
    const uint8_t v693 = v686[0];
    int v694 = (int) v693;
    size_t v695 = (size_t) v694;
    size_t v696 = v695 * 4;
    const uint8_t* v697 = v8 + v696;
    const int8_t* v698 = (const int8_t*) v697;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v699 = __riscv_vle8_v_i8mf4(v698, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v700 = __riscv_vand_vx_u8mf4(v6, v692, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v701 = __riscv_vmsne_vx_u8mf4_b32(v700, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v702 = __riscv_vneg_v_i8mf4(v699, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v703 = __riscv_vmerge_vvm_i8mf4(v699, v702, v701, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v704 = __riscv_vsext_vf4_i32m1(v703, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v705 = __riscv_vfcvt_f_x_v_f32m1(v704, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v706 = __riscv_vfmul_vf_f32m1(v705, v685, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v687, v706, 4);
    const uint8_t v707 = v686[1];
    int v708 = (int) v707;
    size_t v709 = (size_t) v708;
    size_t v710 = v709 * 4;
    const uint8_t* v711 = v8 + v710;
    const int8_t* v712 = (const int8_t*) v711;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v713 = __riscv_vle8_v_i8mf4(v712, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v714 = __riscv_vand_vx_u8mf4(v7, v692, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v715 = __riscv_vmsne_vx_u8mf4_b32(v714, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v716 = __riscv_vneg_v_i8mf4(v713, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v717 = __riscv_vmerge_vvm_i8mf4(v713, v716, v715, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v718 = __riscv_vsext_vf4_i32m1(v717, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v719 = __riscv_vfcvt_f_x_v_f32m1(v718, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v720 = __riscv_vfmul_vf_f32m1(v719, v685, 4);
    float* v721 = v687 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v721, v720, 4);
    uint32_t v722 = v679 >> 7u;
    uint32_t v723 = v722 & 127u;
    int v724 = (int) v723;
    const uint8_t v725 = weft_iq3xxs_ksigns[v724];
    uint8_t v726 = (uint8_t) v725;
    const uint8_t v727 = v686[2];
    int v728 = (int) v727;
    size_t v729 = (size_t) v728;
    size_t v730 = v729 * 4;
    const uint8_t* v731 = v8 + v730;
    const int8_t* v732 = (const int8_t*) v731;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v733 = __riscv_vle8_v_i8mf4(v732, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v734 = __riscv_vand_vx_u8mf4(v6, v726, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v735 = __riscv_vmsne_vx_u8mf4_b32(v734, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v736 = __riscv_vneg_v_i8mf4(v733, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v737 = __riscv_vmerge_vvm_i8mf4(v733, v736, v735, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v738 = __riscv_vsext_vf4_i32m1(v737, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v739 = __riscv_vfcvt_f_x_v_f32m1(v738, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v740 = __riscv_vfmul_vf_f32m1(v739, v685, 4);
    float* v741 = v687 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v741, v740, 4);
    const uint8_t v742 = v686[3];
    int v743 = (int) v742;
    size_t v744 = (size_t) v743;
    size_t v745 = v744 * 4;
    const uint8_t* v746 = v8 + v745;
    const int8_t* v747 = (const int8_t*) v746;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v748 = __riscv_vle8_v_i8mf4(v747, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v749 = __riscv_vand_vx_u8mf4(v7, v726, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v750 = __riscv_vmsne_vx_u8mf4_b32(v749, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v751 = __riscv_vneg_v_i8mf4(v748, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v752 = __riscv_vmerge_vvm_i8mf4(v748, v751, v750, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v753 = __riscv_vsext_vf4_i32m1(v752, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v754 = __riscv_vfcvt_f_x_v_f32m1(v753, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v755 = __riscv_vfmul_vf_f32m1(v754, v685, 4);
    float* v756 = v687 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v756, v755, 4);
    uint32_t v757 = v679 >> 14u;
    uint32_t v758 = v757 & 127u;
    int v759 = (int) v758;
    const uint8_t v760 = weft_iq3xxs_ksigns[v759];
    uint8_t v761 = (uint8_t) v760;
    const uint8_t v762 = v686[4];
    int v763 = (int) v762;
    size_t v764 = (size_t) v763;
    size_t v765 = v764 * 4;
    const uint8_t* v766 = v8 + v765;
    const int8_t* v767 = (const int8_t*) v766;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v768 = __riscv_vle8_v_i8mf4(v767, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v769 = __riscv_vand_vx_u8mf4(v6, v761, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v770 = __riscv_vmsne_vx_u8mf4_b32(v769, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v771 = __riscv_vneg_v_i8mf4(v768, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v772 = __riscv_vmerge_vvm_i8mf4(v768, v771, v770, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v773 = __riscv_vsext_vf4_i32m1(v772, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v774 = __riscv_vfcvt_f_x_v_f32m1(v773, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v775 = __riscv_vfmul_vf_f32m1(v774, v685, 4);
    float* v776 = v687 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v776, v775, 4);
    const uint8_t v777 = v686[5];
    int v778 = (int) v777;
    size_t v779 = (size_t) v778;
    size_t v780 = v779 * 4;
    const uint8_t* v781 = v8 + v780;
    const int8_t* v782 = (const int8_t*) v781;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v783 = __riscv_vle8_v_i8mf4(v782, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v784 = __riscv_vand_vx_u8mf4(v7, v761, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v785 = __riscv_vmsne_vx_u8mf4_b32(v784, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v786 = __riscv_vneg_v_i8mf4(v783, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v787 = __riscv_vmerge_vvm_i8mf4(v783, v786, v785, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v788 = __riscv_vsext_vf4_i32m1(v787, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v789 = __riscv_vfcvt_f_x_v_f32m1(v788, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v790 = __riscv_vfmul_vf_f32m1(v789, v685, 4);
    float* v791 = v687 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v791, v790, 4);
    uint32_t v792 = v679 >> 21u;
    uint32_t v793 = v792 & 127u;
    int v794 = (int) v793;
    const uint8_t v795 = weft_iq3xxs_ksigns[v794];
    uint8_t v796 = (uint8_t) v795;
    const uint8_t v797 = v686[6];
    int v798 = (int) v797;
    size_t v799 = (size_t) v798;
    size_t v800 = v799 * 4;
    const uint8_t* v801 = v8 + v800;
    const int8_t* v802 = (const int8_t*) v801;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v803 = __riscv_vle8_v_i8mf4(v802, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v804 = __riscv_vand_vx_u8mf4(v6, v796, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v805 = __riscv_vmsne_vx_u8mf4_b32(v804, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v806 = __riscv_vneg_v_i8mf4(v803, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v807 = __riscv_vmerge_vvm_i8mf4(v803, v806, v805, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v808 = __riscv_vsext_vf4_i32m1(v807, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v809 = __riscv_vfcvt_f_x_v_f32m1(v808, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v810 = __riscv_vfmul_vf_f32m1(v809, v685, 4);
    float* v811 = v687 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v811, v810, 4);
    const uint8_t v812 = v686[7];
    int v813 = (int) v812;
    size_t v814 = (size_t) v813;
    size_t v815 = v814 * 4;
    const uint8_t* v816 = v8 + v815;
    const int8_t* v817 = (const int8_t*) v816;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v818 = __riscv_vle8_v_i8mf4(v817, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v819 = __riscv_vand_vx_u8mf4(v7, v796, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v820 = __riscv_vmsne_vx_u8mf4_b32(v819, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v821 = __riscv_vneg_v_i8mf4(v818, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v822 = __riscv_vmerge_vvm_i8mf4(v818, v821, v820, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v823 = __riscv_vsext_vf4_i32m1(v822, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v824 = __riscv_vfcvt_f_x_v_f32m1(v823, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v825 = __riscv_vfmul_vf_f32m1(v824, v685, 4);
    float* v826 = v687 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v826, v825, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v827 = v19 + 20;
    const uint8_t v828 = v827[0];
    uint32_t v829 = (uint32_t) v828;
    const uint8_t v830 = v827[1];
    uint32_t v831 = (uint32_t) v830;
    uint32_t v832 = v831 << 8u;
    uint32_t v833 = v829 | v832;
    const uint8_t v834 = v827[2];
    uint32_t v835 = (uint32_t) v834;
    uint32_t v836 = v835 << 16u;
    uint32_t v837 = v833 | v836;
    const uint8_t v838 = v827[3];
    uint32_t v839 = (uint32_t) v838;
    uint32_t v840 = v839 << 24u;
    uint32_t v841 = v837 | v840;
    uint32_t v842 = v841 >> 28u;
    int v843 = (int) v842;
    float v844 = (float) v843;
    float v845 = 0.5f + v844;
    float v846 = v15 * v845;
    float v847 = v846 * 0.5f;
    const uint8_t* v848 = v17 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v849 = v14 + 160;
    uint32_t v850 = v841 >> 0u;
    uint32_t v851 = v850 & 127u;
    int v852 = (int) v851;
    const uint8_t v853 = weft_iq3xxs_ksigns[v852];
    uint8_t v854 = (uint8_t) v853;
    const uint8_t v855 = v848[0];
    int v856 = (int) v855;
    size_t v857 = (size_t) v856;
    size_t v858 = v857 * 4;
    const uint8_t* v859 = v8 + v858;
    const int8_t* v860 = (const int8_t*) v859;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v861 = __riscv_vle8_v_i8mf4(v860, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v862 = __riscv_vand_vx_u8mf4(v6, v854, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v863 = __riscv_vmsne_vx_u8mf4_b32(v862, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v864 = __riscv_vneg_v_i8mf4(v861, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v865 = __riscv_vmerge_vvm_i8mf4(v861, v864, v863, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v866 = __riscv_vsext_vf4_i32m1(v865, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v867 = __riscv_vfcvt_f_x_v_f32m1(v866, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v868 = __riscv_vfmul_vf_f32m1(v867, v847, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v849, v868, 4);
    const uint8_t v869 = v848[1];
    int v870 = (int) v869;
    size_t v871 = (size_t) v870;
    size_t v872 = v871 * 4;
    const uint8_t* v873 = v8 + v872;
    const int8_t* v874 = (const int8_t*) v873;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v875 = __riscv_vle8_v_i8mf4(v874, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v876 = __riscv_vand_vx_u8mf4(v7, v854, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v877 = __riscv_vmsne_vx_u8mf4_b32(v876, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v878 = __riscv_vneg_v_i8mf4(v875, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v879 = __riscv_vmerge_vvm_i8mf4(v875, v878, v877, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v880 = __riscv_vsext_vf4_i32m1(v879, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v881 = __riscv_vfcvt_f_x_v_f32m1(v880, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v882 = __riscv_vfmul_vf_f32m1(v881, v847, 4);
    float* v883 = v849 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v883, v882, 4);
    uint32_t v884 = v841 >> 7u;
    uint32_t v885 = v884 & 127u;
    int v886 = (int) v885;
    const uint8_t v887 = weft_iq3xxs_ksigns[v886];
    uint8_t v888 = (uint8_t) v887;
    const uint8_t v889 = v848[2];
    int v890 = (int) v889;
    size_t v891 = (size_t) v890;
    size_t v892 = v891 * 4;
    const uint8_t* v893 = v8 + v892;
    const int8_t* v894 = (const int8_t*) v893;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v895 = __riscv_vle8_v_i8mf4(v894, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v896 = __riscv_vand_vx_u8mf4(v6, v888, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v897 = __riscv_vmsne_vx_u8mf4_b32(v896, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v898 = __riscv_vneg_v_i8mf4(v895, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v899 = __riscv_vmerge_vvm_i8mf4(v895, v898, v897, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v900 = __riscv_vsext_vf4_i32m1(v899, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v901 = __riscv_vfcvt_f_x_v_f32m1(v900, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v902 = __riscv_vfmul_vf_f32m1(v901, v847, 4);
    float* v903 = v849 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v903, v902, 4);
    const uint8_t v904 = v848[3];
    int v905 = (int) v904;
    size_t v906 = (size_t) v905;
    size_t v907 = v906 * 4;
    const uint8_t* v908 = v8 + v907;
    const int8_t* v909 = (const int8_t*) v908;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v910 = __riscv_vle8_v_i8mf4(v909, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v911 = __riscv_vand_vx_u8mf4(v7, v888, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v912 = __riscv_vmsne_vx_u8mf4_b32(v911, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v913 = __riscv_vneg_v_i8mf4(v910, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v914 = __riscv_vmerge_vvm_i8mf4(v910, v913, v912, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v915 = __riscv_vsext_vf4_i32m1(v914, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v916 = __riscv_vfcvt_f_x_v_f32m1(v915, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v917 = __riscv_vfmul_vf_f32m1(v916, v847, 4);
    float* v918 = v849 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v918, v917, 4);
    uint32_t v919 = v841 >> 14u;
    uint32_t v920 = v919 & 127u;
    int v921 = (int) v920;
    const uint8_t v922 = weft_iq3xxs_ksigns[v921];
    uint8_t v923 = (uint8_t) v922;
    const uint8_t v924 = v848[4];
    int v925 = (int) v924;
    size_t v926 = (size_t) v925;
    size_t v927 = v926 * 4;
    const uint8_t* v928 = v8 + v927;
    const int8_t* v929 = (const int8_t*) v928;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v930 = __riscv_vle8_v_i8mf4(v929, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v931 = __riscv_vand_vx_u8mf4(v6, v923, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v932 = __riscv_vmsne_vx_u8mf4_b32(v931, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v933 = __riscv_vneg_v_i8mf4(v930, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v934 = __riscv_vmerge_vvm_i8mf4(v930, v933, v932, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v935 = __riscv_vsext_vf4_i32m1(v934, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v936 = __riscv_vfcvt_f_x_v_f32m1(v935, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v937 = __riscv_vfmul_vf_f32m1(v936, v847, 4);
    float* v938 = v849 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v938, v937, 4);
    const uint8_t v939 = v848[5];
    int v940 = (int) v939;
    size_t v941 = (size_t) v940;
    size_t v942 = v941 * 4;
    const uint8_t* v943 = v8 + v942;
    const int8_t* v944 = (const int8_t*) v943;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v945 = __riscv_vle8_v_i8mf4(v944, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v946 = __riscv_vand_vx_u8mf4(v7, v923, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v947 = __riscv_vmsne_vx_u8mf4_b32(v946, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v948 = __riscv_vneg_v_i8mf4(v945, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v949 = __riscv_vmerge_vvm_i8mf4(v945, v948, v947, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v950 = __riscv_vsext_vf4_i32m1(v949, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v951 = __riscv_vfcvt_f_x_v_f32m1(v950, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v952 = __riscv_vfmul_vf_f32m1(v951, v847, 4);
    float* v953 = v849 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v953, v952, 4);
    uint32_t v954 = v841 >> 21u;
    uint32_t v955 = v954 & 127u;
    int v956 = (int) v955;
    const uint8_t v957 = weft_iq3xxs_ksigns[v956];
    uint8_t v958 = (uint8_t) v957;
    const uint8_t v959 = v848[6];
    int v960 = (int) v959;
    size_t v961 = (size_t) v960;
    size_t v962 = v961 * 4;
    const uint8_t* v963 = v8 + v962;
    const int8_t* v964 = (const int8_t*) v963;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v965 = __riscv_vle8_v_i8mf4(v964, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v966 = __riscv_vand_vx_u8mf4(v6, v958, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v967 = __riscv_vmsne_vx_u8mf4_b32(v966, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v968 = __riscv_vneg_v_i8mf4(v965, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v969 = __riscv_vmerge_vvm_i8mf4(v965, v968, v967, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v970 = __riscv_vsext_vf4_i32m1(v969, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v971 = __riscv_vfcvt_f_x_v_f32m1(v970, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v972 = __riscv_vfmul_vf_f32m1(v971, v847, 4);
    float* v973 = v849 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v973, v972, 4);
    const uint8_t v974 = v848[7];
    int v975 = (int) v974;
    size_t v976 = (size_t) v975;
    size_t v977 = v976 * 4;
    const uint8_t* v978 = v8 + v977;
    const int8_t* v979 = (const int8_t*) v978;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v980 = __riscv_vle8_v_i8mf4(v979, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v981 = __riscv_vand_vx_u8mf4(v7, v958, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v982 = __riscv_vmsne_vx_u8mf4_b32(v981, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v983 = __riscv_vneg_v_i8mf4(v980, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v984 = __riscv_vmerge_vvm_i8mf4(v980, v983, v982, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v985 = __riscv_vsext_vf4_i32m1(v984, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v986 = __riscv_vfcvt_f_x_v_f32m1(v985, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v987 = __riscv_vfmul_vf_f32m1(v986, v847, 4);
    float* v988 = v849 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v988, v987, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v989 = v19 + 24;
    const uint8_t v990 = v989[0];
    uint32_t v991 = (uint32_t) v990;
    const uint8_t v992 = v989[1];
    uint32_t v993 = (uint32_t) v992;
    uint32_t v994 = v993 << 8u;
    uint32_t v995 = v991 | v994;
    const uint8_t v996 = v989[2];
    uint32_t v997 = (uint32_t) v996;
    uint32_t v998 = v997 << 16u;
    uint32_t v999 = v995 | v998;
    const uint8_t v1000 = v989[3];
    uint32_t v1001 = (uint32_t) v1000;
    uint32_t v1002 = v1001 << 24u;
    uint32_t v1003 = v999 | v1002;
    uint32_t v1004 = v1003 >> 28u;
    int v1005 = (int) v1004;
    float v1006 = (float) v1005;
    float v1007 = 0.5f + v1006;
    float v1008 = v15 * v1007;
    float v1009 = v1008 * 0.5f;
    const uint8_t* v1010 = v17 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v1011 = v14 + 192;
    uint32_t v1012 = v1003 >> 0u;
    uint32_t v1013 = v1012 & 127u;
    int v1014 = (int) v1013;
    const uint8_t v1015 = weft_iq3xxs_ksigns[v1014];
    uint8_t v1016 = (uint8_t) v1015;
    const uint8_t v1017 = v1010[0];
    int v1018 = (int) v1017;
    size_t v1019 = (size_t) v1018;
    size_t v1020 = v1019 * 4;
    const uint8_t* v1021 = v8 + v1020;
    const int8_t* v1022 = (const int8_t*) v1021;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1023 = __riscv_vle8_v_i8mf4(v1022, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1024 = __riscv_vand_vx_u8mf4(v6, v1016, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1025 = __riscv_vmsne_vx_u8mf4_b32(v1024, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1026 = __riscv_vneg_v_i8mf4(v1023, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1027 = __riscv_vmerge_vvm_i8mf4(v1023, v1026, v1025, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1028 = __riscv_vsext_vf4_i32m1(v1027, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1029 = __riscv_vfcvt_f_x_v_f32m1(v1028, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1030 = __riscv_vfmul_vf_f32m1(v1029, v1009, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1011, v1030, 4);
    const uint8_t v1031 = v1010[1];
    int v1032 = (int) v1031;
    size_t v1033 = (size_t) v1032;
    size_t v1034 = v1033 * 4;
    const uint8_t* v1035 = v8 + v1034;
    const int8_t* v1036 = (const int8_t*) v1035;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1037 = __riscv_vle8_v_i8mf4(v1036, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1038 = __riscv_vand_vx_u8mf4(v7, v1016, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1039 = __riscv_vmsne_vx_u8mf4_b32(v1038, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1040 = __riscv_vneg_v_i8mf4(v1037, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1041 = __riscv_vmerge_vvm_i8mf4(v1037, v1040, v1039, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1042 = __riscv_vsext_vf4_i32m1(v1041, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1043 = __riscv_vfcvt_f_x_v_f32m1(v1042, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1044 = __riscv_vfmul_vf_f32m1(v1043, v1009, 4);
    float* v1045 = v1011 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1045, v1044, 4);
    uint32_t v1046 = v1003 >> 7u;
    uint32_t v1047 = v1046 & 127u;
    int v1048 = (int) v1047;
    const uint8_t v1049 = weft_iq3xxs_ksigns[v1048];
    uint8_t v1050 = (uint8_t) v1049;
    const uint8_t v1051 = v1010[2];
    int v1052 = (int) v1051;
    size_t v1053 = (size_t) v1052;
    size_t v1054 = v1053 * 4;
    const uint8_t* v1055 = v8 + v1054;
    const int8_t* v1056 = (const int8_t*) v1055;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1057 = __riscv_vle8_v_i8mf4(v1056, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1058 = __riscv_vand_vx_u8mf4(v6, v1050, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1059 = __riscv_vmsne_vx_u8mf4_b32(v1058, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1060 = __riscv_vneg_v_i8mf4(v1057, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1061 = __riscv_vmerge_vvm_i8mf4(v1057, v1060, v1059, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1062 = __riscv_vsext_vf4_i32m1(v1061, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1063 = __riscv_vfcvt_f_x_v_f32m1(v1062, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1064 = __riscv_vfmul_vf_f32m1(v1063, v1009, 4);
    float* v1065 = v1011 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1065, v1064, 4);
    const uint8_t v1066 = v1010[3];
    int v1067 = (int) v1066;
    size_t v1068 = (size_t) v1067;
    size_t v1069 = v1068 * 4;
    const uint8_t* v1070 = v8 + v1069;
    const int8_t* v1071 = (const int8_t*) v1070;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1072 = __riscv_vle8_v_i8mf4(v1071, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1073 = __riscv_vand_vx_u8mf4(v7, v1050, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1074 = __riscv_vmsne_vx_u8mf4_b32(v1073, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1075 = __riscv_vneg_v_i8mf4(v1072, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1076 = __riscv_vmerge_vvm_i8mf4(v1072, v1075, v1074, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1077 = __riscv_vsext_vf4_i32m1(v1076, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1078 = __riscv_vfcvt_f_x_v_f32m1(v1077, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1079 = __riscv_vfmul_vf_f32m1(v1078, v1009, 4);
    float* v1080 = v1011 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1080, v1079, 4);
    uint32_t v1081 = v1003 >> 14u;
    uint32_t v1082 = v1081 & 127u;
    int v1083 = (int) v1082;
    const uint8_t v1084 = weft_iq3xxs_ksigns[v1083];
    uint8_t v1085 = (uint8_t) v1084;
    const uint8_t v1086 = v1010[4];
    int v1087 = (int) v1086;
    size_t v1088 = (size_t) v1087;
    size_t v1089 = v1088 * 4;
    const uint8_t* v1090 = v8 + v1089;
    const int8_t* v1091 = (const int8_t*) v1090;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1092 = __riscv_vle8_v_i8mf4(v1091, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1093 = __riscv_vand_vx_u8mf4(v6, v1085, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1094 = __riscv_vmsne_vx_u8mf4_b32(v1093, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1095 = __riscv_vneg_v_i8mf4(v1092, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1096 = __riscv_vmerge_vvm_i8mf4(v1092, v1095, v1094, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1097 = __riscv_vsext_vf4_i32m1(v1096, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1098 = __riscv_vfcvt_f_x_v_f32m1(v1097, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1099 = __riscv_vfmul_vf_f32m1(v1098, v1009, 4);
    float* v1100 = v1011 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1100, v1099, 4);
    const uint8_t v1101 = v1010[5];
    int v1102 = (int) v1101;
    size_t v1103 = (size_t) v1102;
    size_t v1104 = v1103 * 4;
    const uint8_t* v1105 = v8 + v1104;
    const int8_t* v1106 = (const int8_t*) v1105;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1107 = __riscv_vle8_v_i8mf4(v1106, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1108 = __riscv_vand_vx_u8mf4(v7, v1085, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1109 = __riscv_vmsne_vx_u8mf4_b32(v1108, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1110 = __riscv_vneg_v_i8mf4(v1107, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1111 = __riscv_vmerge_vvm_i8mf4(v1107, v1110, v1109, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1112 = __riscv_vsext_vf4_i32m1(v1111, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1113 = __riscv_vfcvt_f_x_v_f32m1(v1112, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1114 = __riscv_vfmul_vf_f32m1(v1113, v1009, 4);
    float* v1115 = v1011 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1115, v1114, 4);
    uint32_t v1116 = v1003 >> 21u;
    uint32_t v1117 = v1116 & 127u;
    int v1118 = (int) v1117;
    const uint8_t v1119 = weft_iq3xxs_ksigns[v1118];
    uint8_t v1120 = (uint8_t) v1119;
    const uint8_t v1121 = v1010[6];
    int v1122 = (int) v1121;
    size_t v1123 = (size_t) v1122;
    size_t v1124 = v1123 * 4;
    const uint8_t* v1125 = v8 + v1124;
    const int8_t* v1126 = (const int8_t*) v1125;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1127 = __riscv_vle8_v_i8mf4(v1126, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1128 = __riscv_vand_vx_u8mf4(v6, v1120, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1129 = __riscv_vmsne_vx_u8mf4_b32(v1128, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1130 = __riscv_vneg_v_i8mf4(v1127, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1131 = __riscv_vmerge_vvm_i8mf4(v1127, v1130, v1129, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1132 = __riscv_vsext_vf4_i32m1(v1131, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1133 = __riscv_vfcvt_f_x_v_f32m1(v1132, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1134 = __riscv_vfmul_vf_f32m1(v1133, v1009, 4);
    float* v1135 = v1011 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1135, v1134, 4);
    const uint8_t v1136 = v1010[7];
    int v1137 = (int) v1136;
    size_t v1138 = (size_t) v1137;
    size_t v1139 = v1138 * 4;
    const uint8_t* v1140 = v8 + v1139;
    const int8_t* v1141 = (const int8_t*) v1140;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1142 = __riscv_vle8_v_i8mf4(v1141, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1143 = __riscv_vand_vx_u8mf4(v7, v1120, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1144 = __riscv_vmsne_vx_u8mf4_b32(v1143, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1145 = __riscv_vneg_v_i8mf4(v1142, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1146 = __riscv_vmerge_vvm_i8mf4(v1142, v1145, v1144, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1147 = __riscv_vsext_vf4_i32m1(v1146, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1148 = __riscv_vfcvt_f_x_v_f32m1(v1147, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1149 = __riscv_vfmul_vf_f32m1(v1148, v1009, 4);
    float* v1150 = v1011 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1150, v1149, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v1151 = v19 + 28;
    const uint8_t v1152 = v1151[0];
    uint32_t v1153 = (uint32_t) v1152;
    const uint8_t v1154 = v1151[1];
    uint32_t v1155 = (uint32_t) v1154;
    uint32_t v1156 = v1155 << 8u;
    uint32_t v1157 = v1153 | v1156;
    const uint8_t v1158 = v1151[2];
    uint32_t v1159 = (uint32_t) v1158;
    uint32_t v1160 = v1159 << 16u;
    uint32_t v1161 = v1157 | v1160;
    const uint8_t v1162 = v1151[3];
    uint32_t v1163 = (uint32_t) v1162;
    uint32_t v1164 = v1163 << 24u;
    uint32_t v1165 = v1161 | v1164;
    uint32_t v1166 = v1165 >> 28u;
    int v1167 = (int) v1166;
    float v1168 = (float) v1167;
    float v1169 = 0.5f + v1168;
    float v1170 = v15 * v1169;
    float v1171 = v1170 * 0.5f;
    const uint8_t* v1172 = v17 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    float* v1173 = v14 + 224;
    uint32_t v1174 = v1165 >> 0u;
    uint32_t v1175 = v1174 & 127u;
    int v1176 = (int) v1175;
    const uint8_t v1177 = weft_iq3xxs_ksigns[v1176];
    uint8_t v1178 = (uint8_t) v1177;
    const uint8_t v1179 = v1172[0];
    int v1180 = (int) v1179;
    size_t v1181 = (size_t) v1180;
    size_t v1182 = v1181 * 4;
    const uint8_t* v1183 = v8 + v1182;
    const int8_t* v1184 = (const int8_t*) v1183;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1185 = __riscv_vle8_v_i8mf4(v1184, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1186 = __riscv_vand_vx_u8mf4(v6, v1178, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1187 = __riscv_vmsne_vx_u8mf4_b32(v1186, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1188 = __riscv_vneg_v_i8mf4(v1185, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1189 = __riscv_vmerge_vvm_i8mf4(v1185, v1188, v1187, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1190 = __riscv_vsext_vf4_i32m1(v1189, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1191 = __riscv_vfcvt_f_x_v_f32m1(v1190, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1192 = __riscv_vfmul_vf_f32m1(v1191, v1171, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1173, v1192, 4);
    const uint8_t v1193 = v1172[1];
    int v1194 = (int) v1193;
    size_t v1195 = (size_t) v1194;
    size_t v1196 = v1195 * 4;
    const uint8_t* v1197 = v8 + v1196;
    const int8_t* v1198 = (const int8_t*) v1197;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1199 = __riscv_vle8_v_i8mf4(v1198, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1200 = __riscv_vand_vx_u8mf4(v7, v1178, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1201 = __riscv_vmsne_vx_u8mf4_b32(v1200, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1202 = __riscv_vneg_v_i8mf4(v1199, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1203 = __riscv_vmerge_vvm_i8mf4(v1199, v1202, v1201, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1204 = __riscv_vsext_vf4_i32m1(v1203, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1205 = __riscv_vfcvt_f_x_v_f32m1(v1204, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1206 = __riscv_vfmul_vf_f32m1(v1205, v1171, 4);
    float* v1207 = v1173 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1207, v1206, 4);
    uint32_t v1208 = v1165 >> 7u;
    uint32_t v1209 = v1208 & 127u;
    int v1210 = (int) v1209;
    const uint8_t v1211 = weft_iq3xxs_ksigns[v1210];
    uint8_t v1212 = (uint8_t) v1211;
    const uint8_t v1213 = v1172[2];
    int v1214 = (int) v1213;
    size_t v1215 = (size_t) v1214;
    size_t v1216 = v1215 * 4;
    const uint8_t* v1217 = v8 + v1216;
    const int8_t* v1218 = (const int8_t*) v1217;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1219 = __riscv_vle8_v_i8mf4(v1218, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1220 = __riscv_vand_vx_u8mf4(v6, v1212, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1221 = __riscv_vmsne_vx_u8mf4_b32(v1220, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1222 = __riscv_vneg_v_i8mf4(v1219, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1223 = __riscv_vmerge_vvm_i8mf4(v1219, v1222, v1221, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1224 = __riscv_vsext_vf4_i32m1(v1223, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1225 = __riscv_vfcvt_f_x_v_f32m1(v1224, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1226 = __riscv_vfmul_vf_f32m1(v1225, v1171, 4);
    float* v1227 = v1173 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1227, v1226, 4);
    const uint8_t v1228 = v1172[3];
    int v1229 = (int) v1228;
    size_t v1230 = (size_t) v1229;
    size_t v1231 = v1230 * 4;
    const uint8_t* v1232 = v8 + v1231;
    const int8_t* v1233 = (const int8_t*) v1232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1234 = __riscv_vle8_v_i8mf4(v1233, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1235 = __riscv_vand_vx_u8mf4(v7, v1212, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1236 = __riscv_vmsne_vx_u8mf4_b32(v1235, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1237 = __riscv_vneg_v_i8mf4(v1234, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1238 = __riscv_vmerge_vvm_i8mf4(v1234, v1237, v1236, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1239 = __riscv_vsext_vf4_i32m1(v1238, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1240 = __riscv_vfcvt_f_x_v_f32m1(v1239, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1241 = __riscv_vfmul_vf_f32m1(v1240, v1171, 4);
    float* v1242 = v1173 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1242, v1241, 4);
    uint32_t v1243 = v1165 >> 14u;
    uint32_t v1244 = v1243 & 127u;
    int v1245 = (int) v1244;
    const uint8_t v1246 = weft_iq3xxs_ksigns[v1245];
    uint8_t v1247 = (uint8_t) v1246;
    const uint8_t v1248 = v1172[4];
    int v1249 = (int) v1248;
    size_t v1250 = (size_t) v1249;
    size_t v1251 = v1250 * 4;
    const uint8_t* v1252 = v8 + v1251;
    const int8_t* v1253 = (const int8_t*) v1252;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1254 = __riscv_vle8_v_i8mf4(v1253, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1255 = __riscv_vand_vx_u8mf4(v6, v1247, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1256 = __riscv_vmsne_vx_u8mf4_b32(v1255, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1257 = __riscv_vneg_v_i8mf4(v1254, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1258 = __riscv_vmerge_vvm_i8mf4(v1254, v1257, v1256, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1259 = __riscv_vsext_vf4_i32m1(v1258, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1260 = __riscv_vfcvt_f_x_v_f32m1(v1259, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1261 = __riscv_vfmul_vf_f32m1(v1260, v1171, 4);
    float* v1262 = v1173 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1262, v1261, 4);
    const uint8_t v1263 = v1172[5];
    int v1264 = (int) v1263;
    size_t v1265 = (size_t) v1264;
    size_t v1266 = v1265 * 4;
    const uint8_t* v1267 = v8 + v1266;
    const int8_t* v1268 = (const int8_t*) v1267;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1269 = __riscv_vle8_v_i8mf4(v1268, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1270 = __riscv_vand_vx_u8mf4(v7, v1247, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1271 = __riscv_vmsne_vx_u8mf4_b32(v1270, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1272 = __riscv_vneg_v_i8mf4(v1269, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1273 = __riscv_vmerge_vvm_i8mf4(v1269, v1272, v1271, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1274 = __riscv_vsext_vf4_i32m1(v1273, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1275 = __riscv_vfcvt_f_x_v_f32m1(v1274, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1276 = __riscv_vfmul_vf_f32m1(v1275, v1171, 4);
    float* v1277 = v1173 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1277, v1276, 4);
    uint32_t v1278 = v1165 >> 21u;
    uint32_t v1279 = v1278 & 127u;
    int v1280 = (int) v1279;
    const uint8_t v1281 = weft_iq3xxs_ksigns[v1280];
    uint8_t v1282 = (uint8_t) v1281;
    const uint8_t v1283 = v1172[6];
    int v1284 = (int) v1283;
    size_t v1285 = (size_t) v1284;
    size_t v1286 = v1285 * 4;
    const uint8_t* v1287 = v8 + v1286;
    const int8_t* v1288 = (const int8_t*) v1287;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1289 = __riscv_vle8_v_i8mf4(v1288, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1290 = __riscv_vand_vx_u8mf4(v6, v1282, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1291 = __riscv_vmsne_vx_u8mf4_b32(v1290, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1292 = __riscv_vneg_v_i8mf4(v1289, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1293 = __riscv_vmerge_vvm_i8mf4(v1289, v1292, v1291, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1294 = __riscv_vsext_vf4_i32m1(v1293, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1295 = __riscv_vfcvt_f_x_v_f32m1(v1294, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1296 = __riscv_vfmul_vf_f32m1(v1295, v1171, 4);
    float* v1297 = v1173 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1297, v1296, 4);
    const uint8_t v1298 = v1172[7];
    int v1299 = (int) v1298;
    size_t v1300 = (size_t) v1299;
    size_t v1301 = v1300 * 4;
    const uint8_t* v1302 = v8 + v1301;
    const int8_t* v1303 = (const int8_t*) v1302;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1304 = __riscv_vle8_v_i8mf4(v1303, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1305 = __riscv_vand_vx_u8mf4(v7, v1282, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1306 = __riscv_vmsne_vx_u8mf4_b32(v1305, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1307 = __riscv_vneg_v_i8mf4(v1304, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1308 = __riscv_vmerge_vvm_i8mf4(v1304, v1307, v1306, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1309 = __riscv_vsext_vf4_i32m1(v1308, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1310 = __riscv_vfcvt_f_x_v_f32m1(v1309, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1311 = __riscv_vfmul_vf_f32m1(v1310, v1171, 4);
    float* v1312 = v1173 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1312, v1311, 4);
  }
  return;
}


