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
  static const uint8_t weft_iq3xxs_kmask32[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m2_t v6 = __riscv_vle8_v_u8m2(weft_iq3xxs_kmask32, 32);
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
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v39[8];
    const uint8_t v40 = v16[0];
    int v41 = (int) v40;
    const int32_t v42 = v7[v41];
    int32_t v43 = (int32_t) v42;
    v39[0] = v43;
    const uint8_t v44 = v16[1];
    int v45 = (int) v44;
    const int32_t v46 = v7[v45];
    int32_t v47 = (int32_t) v46;
    v39[1] = v47;
    const uint8_t v48 = v16[2];
    int v49 = (int) v48;
    const int32_t v50 = v7[v49];
    int32_t v51 = (int32_t) v50;
    v39[2] = v51;
    const uint8_t v52 = v16[3];
    int v53 = (int) v52;
    const int32_t v54 = v7[v53];
    int32_t v55 = (int32_t) v54;
    v39[3] = v55;
    const uint8_t v56 = v16[4];
    int v57 = (int) v56;
    const int32_t v58 = v7[v57];
    int32_t v59 = (int32_t) v58;
    v39[4] = v59;
    const uint8_t v60 = v16[5];
    int v61 = (int) v60;
    const int32_t v62 = v7[v61];
    int32_t v63 = (int32_t) v62;
    v39[5] = v63;
    const uint8_t v64 = v16[6];
    int v65 = (int) v64;
    const int32_t v66 = v7[v65];
    int32_t v67 = (int32_t) v66;
    v39[6] = v67;
    const uint8_t v68 = v16[7];
    int v69 = (int) v68;
    const int32_t v70 = v7[v69];
    int32_t v71 = (int32_t) v70;
    v39[7] = v71;
    int32_t* v72 = &v39[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v73 = __riscv_vle32_v_i32m2(v72, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v74 = __riscv_vreinterpret_v_i32m2_i8m2(v73);
    uint8_t v75[32];
    uint32_t v76 = v32 >> 0u;
    uint32_t v77 = v76 & 127u;
    int v78 = (int) v77;
    const uint8_t v79 = weft_iq3xxs_ksigns[v78];
    uint8_t v80 = (uint8_t) v79;
    v75[0] = v80;
    v75[1] = v80;
    v75[2] = v80;
    v75[3] = v80;
    v75[4] = v80;
    v75[5] = v80;
    v75[6] = v80;
    v75[7] = v80;
    uint32_t v81 = v32 >> 7u;
    uint32_t v82 = v81 & 127u;
    int v83 = (int) v82;
    const uint8_t v84 = weft_iq3xxs_ksigns[v83];
    uint8_t v85 = (uint8_t) v84;
    v75[8] = v85;
    v75[9] = v85;
    v75[10] = v85;
    v75[11] = v85;
    v75[12] = v85;
    v75[13] = v85;
    v75[14] = v85;
    v75[15] = v85;
    uint32_t v86 = v32 >> 14u;
    uint32_t v87 = v86 & 127u;
    int v88 = (int) v87;
    const uint8_t v89 = weft_iq3xxs_ksigns[v88];
    uint8_t v90 = (uint8_t) v89;
    v75[16] = v90;
    v75[17] = v90;
    v75[18] = v90;
    v75[19] = v90;
    v75[20] = v90;
    v75[21] = v90;
    v75[22] = v90;
    v75[23] = v90;
    uint32_t v91 = v32 >> 21u;
    uint32_t v92 = v91 & 127u;
    int v93 = (int) v92;
    const uint8_t v94 = weft_iq3xxs_ksigns[v93];
    uint8_t v95 = (uint8_t) v94;
    v75[24] = v95;
    v75[25] = v95;
    v75[26] = v95;
    v75[27] = v95;
    v75[28] = v95;
    v75[29] = v95;
    v75[30] = v95;
    v75[31] = v95;
    uint8_t* v96 = &v75[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v97 = __riscv_vle8_v_u8m2(v96, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v98 = __riscv_vand_vv_u8m2(v97, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v99 = __riscv_vmsne_vx_u8m2_b4(v98, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v100 = __riscv_vneg_v_i8m2(v74, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v101 = __riscv_vmerge_vvm_i8m2(v74, v100, v99, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v102 = __riscv_vsext_vf4_i32m8(v101, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v103 = __riscv_vfcvt_f_x_v_f32m8(v102, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v104 = __riscv_vfmul_vf_f32m8(v103, v38, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v13, v104, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v105 = v18 + 4;
    const uint8_t v106 = v105[0];
    uint32_t v107 = (uint32_t) v106;
    const uint8_t v108 = v105[1];
    uint32_t v109 = (uint32_t) v108;
    uint32_t v110 = v109 << 8u;
    uint32_t v111 = v107 | v110;
    const uint8_t v112 = v105[2];
    uint32_t v113 = (uint32_t) v112;
    uint32_t v114 = v113 << 16u;
    uint32_t v115 = v111 | v114;
    const uint8_t v116 = v105[3];
    uint32_t v117 = (uint32_t) v116;
    uint32_t v118 = v117 << 24u;
    uint32_t v119 = v115 | v118;
    uint32_t v120 = v119 >> 28u;
    int v121 = (int) v120;
    float v122 = (float) v121;
    float v123 = 0.5f + v122;
    float v124 = v14 * v123;
    float v125 = v124 * 0.5f;
    const uint8_t* v126 = v16 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v127[8];
    const uint8_t v128 = v126[0];
    int v129 = (int) v128;
    const int32_t v130 = v7[v129];
    int32_t v131 = (int32_t) v130;
    v127[0] = v131;
    const uint8_t v132 = v126[1];
    int v133 = (int) v132;
    const int32_t v134 = v7[v133];
    int32_t v135 = (int32_t) v134;
    v127[1] = v135;
    const uint8_t v136 = v126[2];
    int v137 = (int) v136;
    const int32_t v138 = v7[v137];
    int32_t v139 = (int32_t) v138;
    v127[2] = v139;
    const uint8_t v140 = v126[3];
    int v141 = (int) v140;
    const int32_t v142 = v7[v141];
    int32_t v143 = (int32_t) v142;
    v127[3] = v143;
    const uint8_t v144 = v126[4];
    int v145 = (int) v144;
    const int32_t v146 = v7[v145];
    int32_t v147 = (int32_t) v146;
    v127[4] = v147;
    const uint8_t v148 = v126[5];
    int v149 = (int) v148;
    const int32_t v150 = v7[v149];
    int32_t v151 = (int32_t) v150;
    v127[5] = v151;
    const uint8_t v152 = v126[6];
    int v153 = (int) v152;
    const int32_t v154 = v7[v153];
    int32_t v155 = (int32_t) v154;
    v127[6] = v155;
    const uint8_t v156 = v126[7];
    int v157 = (int) v156;
    const int32_t v158 = v7[v157];
    int32_t v159 = (int32_t) v158;
    v127[7] = v159;
    int32_t* v160 = &v127[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v161 = __riscv_vle32_v_i32m2(v160, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v162 = __riscv_vreinterpret_v_i32m2_i8m2(v161);
    uint8_t v163[32];
    uint32_t v164 = v119 >> 0u;
    uint32_t v165 = v164 & 127u;
    int v166 = (int) v165;
    const uint8_t v167 = weft_iq3xxs_ksigns[v166];
    uint8_t v168 = (uint8_t) v167;
    v163[0] = v168;
    v163[1] = v168;
    v163[2] = v168;
    v163[3] = v168;
    v163[4] = v168;
    v163[5] = v168;
    v163[6] = v168;
    v163[7] = v168;
    uint32_t v169 = v119 >> 7u;
    uint32_t v170 = v169 & 127u;
    int v171 = (int) v170;
    const uint8_t v172 = weft_iq3xxs_ksigns[v171];
    uint8_t v173 = (uint8_t) v172;
    v163[8] = v173;
    v163[9] = v173;
    v163[10] = v173;
    v163[11] = v173;
    v163[12] = v173;
    v163[13] = v173;
    v163[14] = v173;
    v163[15] = v173;
    uint32_t v174 = v119 >> 14u;
    uint32_t v175 = v174 & 127u;
    int v176 = (int) v175;
    const uint8_t v177 = weft_iq3xxs_ksigns[v176];
    uint8_t v178 = (uint8_t) v177;
    v163[16] = v178;
    v163[17] = v178;
    v163[18] = v178;
    v163[19] = v178;
    v163[20] = v178;
    v163[21] = v178;
    v163[22] = v178;
    v163[23] = v178;
    uint32_t v179 = v119 >> 21u;
    uint32_t v180 = v179 & 127u;
    int v181 = (int) v180;
    const uint8_t v182 = weft_iq3xxs_ksigns[v181];
    uint8_t v183 = (uint8_t) v182;
    v163[24] = v183;
    v163[25] = v183;
    v163[26] = v183;
    v163[27] = v183;
    v163[28] = v183;
    v163[29] = v183;
    v163[30] = v183;
    v163[31] = v183;
    uint8_t* v184 = &v163[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v185 = __riscv_vle8_v_u8m2(v184, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v186 = __riscv_vand_vv_u8m2(v185, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v187 = __riscv_vmsne_vx_u8m2_b4(v186, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v188 = __riscv_vneg_v_i8m2(v162, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v189 = __riscv_vmerge_vvm_i8m2(v162, v188, v187, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v190 = __riscv_vsext_vf4_i32m8(v189, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v191 = __riscv_vfcvt_f_x_v_f32m8(v190, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v192 = __riscv_vfmul_vf_f32m8(v191, v125, 32);
    float* v193 = v13 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v193, v192, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v194 = v18 + 8;
    const uint8_t v195 = v194[0];
    uint32_t v196 = (uint32_t) v195;
    const uint8_t v197 = v194[1];
    uint32_t v198 = (uint32_t) v197;
    uint32_t v199 = v198 << 8u;
    uint32_t v200 = v196 | v199;
    const uint8_t v201 = v194[2];
    uint32_t v202 = (uint32_t) v201;
    uint32_t v203 = v202 << 16u;
    uint32_t v204 = v200 | v203;
    const uint8_t v205 = v194[3];
    uint32_t v206 = (uint32_t) v205;
    uint32_t v207 = v206 << 24u;
    uint32_t v208 = v204 | v207;
    uint32_t v209 = v208 >> 28u;
    int v210 = (int) v209;
    float v211 = (float) v210;
    float v212 = 0.5f + v211;
    float v213 = v14 * v212;
    float v214 = v213 * 0.5f;
    const uint8_t* v215 = v16 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v216[8];
    const uint8_t v217 = v215[0];
    int v218 = (int) v217;
    const int32_t v219 = v7[v218];
    int32_t v220 = (int32_t) v219;
    v216[0] = v220;
    const uint8_t v221 = v215[1];
    int v222 = (int) v221;
    const int32_t v223 = v7[v222];
    int32_t v224 = (int32_t) v223;
    v216[1] = v224;
    const uint8_t v225 = v215[2];
    int v226 = (int) v225;
    const int32_t v227 = v7[v226];
    int32_t v228 = (int32_t) v227;
    v216[2] = v228;
    const uint8_t v229 = v215[3];
    int v230 = (int) v229;
    const int32_t v231 = v7[v230];
    int32_t v232 = (int32_t) v231;
    v216[3] = v232;
    const uint8_t v233 = v215[4];
    int v234 = (int) v233;
    const int32_t v235 = v7[v234];
    int32_t v236 = (int32_t) v235;
    v216[4] = v236;
    const uint8_t v237 = v215[5];
    int v238 = (int) v237;
    const int32_t v239 = v7[v238];
    int32_t v240 = (int32_t) v239;
    v216[5] = v240;
    const uint8_t v241 = v215[6];
    int v242 = (int) v241;
    const int32_t v243 = v7[v242];
    int32_t v244 = (int32_t) v243;
    v216[6] = v244;
    const uint8_t v245 = v215[7];
    int v246 = (int) v245;
    const int32_t v247 = v7[v246];
    int32_t v248 = (int32_t) v247;
    v216[7] = v248;
    int32_t* v249 = &v216[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v250 = __riscv_vle32_v_i32m2(v249, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v251 = __riscv_vreinterpret_v_i32m2_i8m2(v250);
    uint8_t v252[32];
    uint32_t v253 = v208 >> 0u;
    uint32_t v254 = v253 & 127u;
    int v255 = (int) v254;
    const uint8_t v256 = weft_iq3xxs_ksigns[v255];
    uint8_t v257 = (uint8_t) v256;
    v252[0] = v257;
    v252[1] = v257;
    v252[2] = v257;
    v252[3] = v257;
    v252[4] = v257;
    v252[5] = v257;
    v252[6] = v257;
    v252[7] = v257;
    uint32_t v258 = v208 >> 7u;
    uint32_t v259 = v258 & 127u;
    int v260 = (int) v259;
    const uint8_t v261 = weft_iq3xxs_ksigns[v260];
    uint8_t v262 = (uint8_t) v261;
    v252[8] = v262;
    v252[9] = v262;
    v252[10] = v262;
    v252[11] = v262;
    v252[12] = v262;
    v252[13] = v262;
    v252[14] = v262;
    v252[15] = v262;
    uint32_t v263 = v208 >> 14u;
    uint32_t v264 = v263 & 127u;
    int v265 = (int) v264;
    const uint8_t v266 = weft_iq3xxs_ksigns[v265];
    uint8_t v267 = (uint8_t) v266;
    v252[16] = v267;
    v252[17] = v267;
    v252[18] = v267;
    v252[19] = v267;
    v252[20] = v267;
    v252[21] = v267;
    v252[22] = v267;
    v252[23] = v267;
    uint32_t v268 = v208 >> 21u;
    uint32_t v269 = v268 & 127u;
    int v270 = (int) v269;
    const uint8_t v271 = weft_iq3xxs_ksigns[v270];
    uint8_t v272 = (uint8_t) v271;
    v252[24] = v272;
    v252[25] = v272;
    v252[26] = v272;
    v252[27] = v272;
    v252[28] = v272;
    v252[29] = v272;
    v252[30] = v272;
    v252[31] = v272;
    uint8_t* v273 = &v252[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v274 = __riscv_vle8_v_u8m2(v273, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v275 = __riscv_vand_vv_u8m2(v274, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v276 = __riscv_vmsne_vx_u8m2_b4(v275, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v277 = __riscv_vneg_v_i8m2(v251, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v278 = __riscv_vmerge_vvm_i8m2(v251, v277, v276, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v279 = __riscv_vsext_vf4_i32m8(v278, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v280 = __riscv_vfcvt_f_x_v_f32m8(v279, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v281 = __riscv_vfmul_vf_f32m8(v280, v214, 32);
    float* v282 = v13 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v282, v281, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v283 = v18 + 12;
    const uint8_t v284 = v283[0];
    uint32_t v285 = (uint32_t) v284;
    const uint8_t v286 = v283[1];
    uint32_t v287 = (uint32_t) v286;
    uint32_t v288 = v287 << 8u;
    uint32_t v289 = v285 | v288;
    const uint8_t v290 = v283[2];
    uint32_t v291 = (uint32_t) v290;
    uint32_t v292 = v291 << 16u;
    uint32_t v293 = v289 | v292;
    const uint8_t v294 = v283[3];
    uint32_t v295 = (uint32_t) v294;
    uint32_t v296 = v295 << 24u;
    uint32_t v297 = v293 | v296;
    uint32_t v298 = v297 >> 28u;
    int v299 = (int) v298;
    float v300 = (float) v299;
    float v301 = 0.5f + v300;
    float v302 = v14 * v301;
    float v303 = v302 * 0.5f;
    const uint8_t* v304 = v16 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v305[8];
    const uint8_t v306 = v304[0];
    int v307 = (int) v306;
    const int32_t v308 = v7[v307];
    int32_t v309 = (int32_t) v308;
    v305[0] = v309;
    const uint8_t v310 = v304[1];
    int v311 = (int) v310;
    const int32_t v312 = v7[v311];
    int32_t v313 = (int32_t) v312;
    v305[1] = v313;
    const uint8_t v314 = v304[2];
    int v315 = (int) v314;
    const int32_t v316 = v7[v315];
    int32_t v317 = (int32_t) v316;
    v305[2] = v317;
    const uint8_t v318 = v304[3];
    int v319 = (int) v318;
    const int32_t v320 = v7[v319];
    int32_t v321 = (int32_t) v320;
    v305[3] = v321;
    const uint8_t v322 = v304[4];
    int v323 = (int) v322;
    const int32_t v324 = v7[v323];
    int32_t v325 = (int32_t) v324;
    v305[4] = v325;
    const uint8_t v326 = v304[5];
    int v327 = (int) v326;
    const int32_t v328 = v7[v327];
    int32_t v329 = (int32_t) v328;
    v305[5] = v329;
    const uint8_t v330 = v304[6];
    int v331 = (int) v330;
    const int32_t v332 = v7[v331];
    int32_t v333 = (int32_t) v332;
    v305[6] = v333;
    const uint8_t v334 = v304[7];
    int v335 = (int) v334;
    const int32_t v336 = v7[v335];
    int32_t v337 = (int32_t) v336;
    v305[7] = v337;
    int32_t* v338 = &v305[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v339 = __riscv_vle32_v_i32m2(v338, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v340 = __riscv_vreinterpret_v_i32m2_i8m2(v339);
    uint8_t v341[32];
    uint32_t v342 = v297 >> 0u;
    uint32_t v343 = v342 & 127u;
    int v344 = (int) v343;
    const uint8_t v345 = weft_iq3xxs_ksigns[v344];
    uint8_t v346 = (uint8_t) v345;
    v341[0] = v346;
    v341[1] = v346;
    v341[2] = v346;
    v341[3] = v346;
    v341[4] = v346;
    v341[5] = v346;
    v341[6] = v346;
    v341[7] = v346;
    uint32_t v347 = v297 >> 7u;
    uint32_t v348 = v347 & 127u;
    int v349 = (int) v348;
    const uint8_t v350 = weft_iq3xxs_ksigns[v349];
    uint8_t v351 = (uint8_t) v350;
    v341[8] = v351;
    v341[9] = v351;
    v341[10] = v351;
    v341[11] = v351;
    v341[12] = v351;
    v341[13] = v351;
    v341[14] = v351;
    v341[15] = v351;
    uint32_t v352 = v297 >> 14u;
    uint32_t v353 = v352 & 127u;
    int v354 = (int) v353;
    const uint8_t v355 = weft_iq3xxs_ksigns[v354];
    uint8_t v356 = (uint8_t) v355;
    v341[16] = v356;
    v341[17] = v356;
    v341[18] = v356;
    v341[19] = v356;
    v341[20] = v356;
    v341[21] = v356;
    v341[22] = v356;
    v341[23] = v356;
    uint32_t v357 = v297 >> 21u;
    uint32_t v358 = v357 & 127u;
    int v359 = (int) v358;
    const uint8_t v360 = weft_iq3xxs_ksigns[v359];
    uint8_t v361 = (uint8_t) v360;
    v341[24] = v361;
    v341[25] = v361;
    v341[26] = v361;
    v341[27] = v361;
    v341[28] = v361;
    v341[29] = v361;
    v341[30] = v361;
    v341[31] = v361;
    uint8_t* v362 = &v341[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v363 = __riscv_vle8_v_u8m2(v362, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v364 = __riscv_vand_vv_u8m2(v363, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v365 = __riscv_vmsne_vx_u8m2_b4(v364, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v366 = __riscv_vneg_v_i8m2(v340, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v367 = __riscv_vmerge_vvm_i8m2(v340, v366, v365, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v368 = __riscv_vsext_vf4_i32m8(v367, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v369 = __riscv_vfcvt_f_x_v_f32m8(v368, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v370 = __riscv_vfmul_vf_f32m8(v369, v303, 32);
    float* v371 = v13 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v371, v370, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v372 = v18 + 16;
    const uint8_t v373 = v372[0];
    uint32_t v374 = (uint32_t) v373;
    const uint8_t v375 = v372[1];
    uint32_t v376 = (uint32_t) v375;
    uint32_t v377 = v376 << 8u;
    uint32_t v378 = v374 | v377;
    const uint8_t v379 = v372[2];
    uint32_t v380 = (uint32_t) v379;
    uint32_t v381 = v380 << 16u;
    uint32_t v382 = v378 | v381;
    const uint8_t v383 = v372[3];
    uint32_t v384 = (uint32_t) v383;
    uint32_t v385 = v384 << 24u;
    uint32_t v386 = v382 | v385;
    uint32_t v387 = v386 >> 28u;
    int v388 = (int) v387;
    float v389 = (float) v388;
    float v390 = 0.5f + v389;
    float v391 = v14 * v390;
    float v392 = v391 * 0.5f;
    const uint8_t* v393 = v16 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v394[8];
    const uint8_t v395 = v393[0];
    int v396 = (int) v395;
    const int32_t v397 = v7[v396];
    int32_t v398 = (int32_t) v397;
    v394[0] = v398;
    const uint8_t v399 = v393[1];
    int v400 = (int) v399;
    const int32_t v401 = v7[v400];
    int32_t v402 = (int32_t) v401;
    v394[1] = v402;
    const uint8_t v403 = v393[2];
    int v404 = (int) v403;
    const int32_t v405 = v7[v404];
    int32_t v406 = (int32_t) v405;
    v394[2] = v406;
    const uint8_t v407 = v393[3];
    int v408 = (int) v407;
    const int32_t v409 = v7[v408];
    int32_t v410 = (int32_t) v409;
    v394[3] = v410;
    const uint8_t v411 = v393[4];
    int v412 = (int) v411;
    const int32_t v413 = v7[v412];
    int32_t v414 = (int32_t) v413;
    v394[4] = v414;
    const uint8_t v415 = v393[5];
    int v416 = (int) v415;
    const int32_t v417 = v7[v416];
    int32_t v418 = (int32_t) v417;
    v394[5] = v418;
    const uint8_t v419 = v393[6];
    int v420 = (int) v419;
    const int32_t v421 = v7[v420];
    int32_t v422 = (int32_t) v421;
    v394[6] = v422;
    const uint8_t v423 = v393[7];
    int v424 = (int) v423;
    const int32_t v425 = v7[v424];
    int32_t v426 = (int32_t) v425;
    v394[7] = v426;
    int32_t* v427 = &v394[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v428 = __riscv_vle32_v_i32m2(v427, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v429 = __riscv_vreinterpret_v_i32m2_i8m2(v428);
    uint8_t v430[32];
    uint32_t v431 = v386 >> 0u;
    uint32_t v432 = v431 & 127u;
    int v433 = (int) v432;
    const uint8_t v434 = weft_iq3xxs_ksigns[v433];
    uint8_t v435 = (uint8_t) v434;
    v430[0] = v435;
    v430[1] = v435;
    v430[2] = v435;
    v430[3] = v435;
    v430[4] = v435;
    v430[5] = v435;
    v430[6] = v435;
    v430[7] = v435;
    uint32_t v436 = v386 >> 7u;
    uint32_t v437 = v436 & 127u;
    int v438 = (int) v437;
    const uint8_t v439 = weft_iq3xxs_ksigns[v438];
    uint8_t v440 = (uint8_t) v439;
    v430[8] = v440;
    v430[9] = v440;
    v430[10] = v440;
    v430[11] = v440;
    v430[12] = v440;
    v430[13] = v440;
    v430[14] = v440;
    v430[15] = v440;
    uint32_t v441 = v386 >> 14u;
    uint32_t v442 = v441 & 127u;
    int v443 = (int) v442;
    const uint8_t v444 = weft_iq3xxs_ksigns[v443];
    uint8_t v445 = (uint8_t) v444;
    v430[16] = v445;
    v430[17] = v445;
    v430[18] = v445;
    v430[19] = v445;
    v430[20] = v445;
    v430[21] = v445;
    v430[22] = v445;
    v430[23] = v445;
    uint32_t v446 = v386 >> 21u;
    uint32_t v447 = v446 & 127u;
    int v448 = (int) v447;
    const uint8_t v449 = weft_iq3xxs_ksigns[v448];
    uint8_t v450 = (uint8_t) v449;
    v430[24] = v450;
    v430[25] = v450;
    v430[26] = v450;
    v430[27] = v450;
    v430[28] = v450;
    v430[29] = v450;
    v430[30] = v450;
    v430[31] = v450;
    uint8_t* v451 = &v430[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v452 = __riscv_vle8_v_u8m2(v451, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v453 = __riscv_vand_vv_u8m2(v452, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v454 = __riscv_vmsne_vx_u8m2_b4(v453, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v455 = __riscv_vneg_v_i8m2(v429, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v456 = __riscv_vmerge_vvm_i8m2(v429, v455, v454, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v457 = __riscv_vsext_vf4_i32m8(v456, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v458 = __riscv_vfcvt_f_x_v_f32m8(v457, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v459 = __riscv_vfmul_vf_f32m8(v458, v392, 32);
    float* v460 = v13 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v460, v459, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v461 = v18 + 20;
    const uint8_t v462 = v461[0];
    uint32_t v463 = (uint32_t) v462;
    const uint8_t v464 = v461[1];
    uint32_t v465 = (uint32_t) v464;
    uint32_t v466 = v465 << 8u;
    uint32_t v467 = v463 | v466;
    const uint8_t v468 = v461[2];
    uint32_t v469 = (uint32_t) v468;
    uint32_t v470 = v469 << 16u;
    uint32_t v471 = v467 | v470;
    const uint8_t v472 = v461[3];
    uint32_t v473 = (uint32_t) v472;
    uint32_t v474 = v473 << 24u;
    uint32_t v475 = v471 | v474;
    uint32_t v476 = v475 >> 28u;
    int v477 = (int) v476;
    float v478 = (float) v477;
    float v479 = 0.5f + v478;
    float v480 = v14 * v479;
    float v481 = v480 * 0.5f;
    const uint8_t* v482 = v16 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v483[8];
    const uint8_t v484 = v482[0];
    int v485 = (int) v484;
    const int32_t v486 = v7[v485];
    int32_t v487 = (int32_t) v486;
    v483[0] = v487;
    const uint8_t v488 = v482[1];
    int v489 = (int) v488;
    const int32_t v490 = v7[v489];
    int32_t v491 = (int32_t) v490;
    v483[1] = v491;
    const uint8_t v492 = v482[2];
    int v493 = (int) v492;
    const int32_t v494 = v7[v493];
    int32_t v495 = (int32_t) v494;
    v483[2] = v495;
    const uint8_t v496 = v482[3];
    int v497 = (int) v496;
    const int32_t v498 = v7[v497];
    int32_t v499 = (int32_t) v498;
    v483[3] = v499;
    const uint8_t v500 = v482[4];
    int v501 = (int) v500;
    const int32_t v502 = v7[v501];
    int32_t v503 = (int32_t) v502;
    v483[4] = v503;
    const uint8_t v504 = v482[5];
    int v505 = (int) v504;
    const int32_t v506 = v7[v505];
    int32_t v507 = (int32_t) v506;
    v483[5] = v507;
    const uint8_t v508 = v482[6];
    int v509 = (int) v508;
    const int32_t v510 = v7[v509];
    int32_t v511 = (int32_t) v510;
    v483[6] = v511;
    const uint8_t v512 = v482[7];
    int v513 = (int) v512;
    const int32_t v514 = v7[v513];
    int32_t v515 = (int32_t) v514;
    v483[7] = v515;
    int32_t* v516 = &v483[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v517 = __riscv_vle32_v_i32m2(v516, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v518 = __riscv_vreinterpret_v_i32m2_i8m2(v517);
    uint8_t v519[32];
    uint32_t v520 = v475 >> 0u;
    uint32_t v521 = v520 & 127u;
    int v522 = (int) v521;
    const uint8_t v523 = weft_iq3xxs_ksigns[v522];
    uint8_t v524 = (uint8_t) v523;
    v519[0] = v524;
    v519[1] = v524;
    v519[2] = v524;
    v519[3] = v524;
    v519[4] = v524;
    v519[5] = v524;
    v519[6] = v524;
    v519[7] = v524;
    uint32_t v525 = v475 >> 7u;
    uint32_t v526 = v525 & 127u;
    int v527 = (int) v526;
    const uint8_t v528 = weft_iq3xxs_ksigns[v527];
    uint8_t v529 = (uint8_t) v528;
    v519[8] = v529;
    v519[9] = v529;
    v519[10] = v529;
    v519[11] = v529;
    v519[12] = v529;
    v519[13] = v529;
    v519[14] = v529;
    v519[15] = v529;
    uint32_t v530 = v475 >> 14u;
    uint32_t v531 = v530 & 127u;
    int v532 = (int) v531;
    const uint8_t v533 = weft_iq3xxs_ksigns[v532];
    uint8_t v534 = (uint8_t) v533;
    v519[16] = v534;
    v519[17] = v534;
    v519[18] = v534;
    v519[19] = v534;
    v519[20] = v534;
    v519[21] = v534;
    v519[22] = v534;
    v519[23] = v534;
    uint32_t v535 = v475 >> 21u;
    uint32_t v536 = v535 & 127u;
    int v537 = (int) v536;
    const uint8_t v538 = weft_iq3xxs_ksigns[v537];
    uint8_t v539 = (uint8_t) v538;
    v519[24] = v539;
    v519[25] = v539;
    v519[26] = v539;
    v519[27] = v539;
    v519[28] = v539;
    v519[29] = v539;
    v519[30] = v539;
    v519[31] = v539;
    uint8_t* v540 = &v519[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v541 = __riscv_vle8_v_u8m2(v540, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v542 = __riscv_vand_vv_u8m2(v541, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v543 = __riscv_vmsne_vx_u8m2_b4(v542, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v544 = __riscv_vneg_v_i8m2(v518, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v545 = __riscv_vmerge_vvm_i8m2(v518, v544, v543, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v546 = __riscv_vsext_vf4_i32m8(v545, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v547 = __riscv_vfcvt_f_x_v_f32m8(v546, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v548 = __riscv_vfmul_vf_f32m8(v547, v481, 32);
    float* v549 = v13 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v549, v548, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v550 = v18 + 24;
    const uint8_t v551 = v550[0];
    uint32_t v552 = (uint32_t) v551;
    const uint8_t v553 = v550[1];
    uint32_t v554 = (uint32_t) v553;
    uint32_t v555 = v554 << 8u;
    uint32_t v556 = v552 | v555;
    const uint8_t v557 = v550[2];
    uint32_t v558 = (uint32_t) v557;
    uint32_t v559 = v558 << 16u;
    uint32_t v560 = v556 | v559;
    const uint8_t v561 = v550[3];
    uint32_t v562 = (uint32_t) v561;
    uint32_t v563 = v562 << 24u;
    uint32_t v564 = v560 | v563;
    uint32_t v565 = v564 >> 28u;
    int v566 = (int) v565;
    float v567 = (float) v566;
    float v568 = 0.5f + v567;
    float v569 = v14 * v568;
    float v570 = v569 * 0.5f;
    const uint8_t* v571 = v16 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v572[8];
    const uint8_t v573 = v571[0];
    int v574 = (int) v573;
    const int32_t v575 = v7[v574];
    int32_t v576 = (int32_t) v575;
    v572[0] = v576;
    const uint8_t v577 = v571[1];
    int v578 = (int) v577;
    const int32_t v579 = v7[v578];
    int32_t v580 = (int32_t) v579;
    v572[1] = v580;
    const uint8_t v581 = v571[2];
    int v582 = (int) v581;
    const int32_t v583 = v7[v582];
    int32_t v584 = (int32_t) v583;
    v572[2] = v584;
    const uint8_t v585 = v571[3];
    int v586 = (int) v585;
    const int32_t v587 = v7[v586];
    int32_t v588 = (int32_t) v587;
    v572[3] = v588;
    const uint8_t v589 = v571[4];
    int v590 = (int) v589;
    const int32_t v591 = v7[v590];
    int32_t v592 = (int32_t) v591;
    v572[4] = v592;
    const uint8_t v593 = v571[5];
    int v594 = (int) v593;
    const int32_t v595 = v7[v594];
    int32_t v596 = (int32_t) v595;
    v572[5] = v596;
    const uint8_t v597 = v571[6];
    int v598 = (int) v597;
    const int32_t v599 = v7[v598];
    int32_t v600 = (int32_t) v599;
    v572[6] = v600;
    const uint8_t v601 = v571[7];
    int v602 = (int) v601;
    const int32_t v603 = v7[v602];
    int32_t v604 = (int32_t) v603;
    v572[7] = v604;
    int32_t* v605 = &v572[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v606 = __riscv_vle32_v_i32m2(v605, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v607 = __riscv_vreinterpret_v_i32m2_i8m2(v606);
    uint8_t v608[32];
    uint32_t v609 = v564 >> 0u;
    uint32_t v610 = v609 & 127u;
    int v611 = (int) v610;
    const uint8_t v612 = weft_iq3xxs_ksigns[v611];
    uint8_t v613 = (uint8_t) v612;
    v608[0] = v613;
    v608[1] = v613;
    v608[2] = v613;
    v608[3] = v613;
    v608[4] = v613;
    v608[5] = v613;
    v608[6] = v613;
    v608[7] = v613;
    uint32_t v614 = v564 >> 7u;
    uint32_t v615 = v614 & 127u;
    int v616 = (int) v615;
    const uint8_t v617 = weft_iq3xxs_ksigns[v616];
    uint8_t v618 = (uint8_t) v617;
    v608[8] = v618;
    v608[9] = v618;
    v608[10] = v618;
    v608[11] = v618;
    v608[12] = v618;
    v608[13] = v618;
    v608[14] = v618;
    v608[15] = v618;
    uint32_t v619 = v564 >> 14u;
    uint32_t v620 = v619 & 127u;
    int v621 = (int) v620;
    const uint8_t v622 = weft_iq3xxs_ksigns[v621];
    uint8_t v623 = (uint8_t) v622;
    v608[16] = v623;
    v608[17] = v623;
    v608[18] = v623;
    v608[19] = v623;
    v608[20] = v623;
    v608[21] = v623;
    v608[22] = v623;
    v608[23] = v623;
    uint32_t v624 = v564 >> 21u;
    uint32_t v625 = v624 & 127u;
    int v626 = (int) v625;
    const uint8_t v627 = weft_iq3xxs_ksigns[v626];
    uint8_t v628 = (uint8_t) v627;
    v608[24] = v628;
    v608[25] = v628;
    v608[26] = v628;
    v608[27] = v628;
    v608[28] = v628;
    v608[29] = v628;
    v608[30] = v628;
    v608[31] = v628;
    uint8_t* v629 = &v608[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v630 = __riscv_vle8_v_u8m2(v629, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v631 = __riscv_vand_vv_u8m2(v630, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v632 = __riscv_vmsne_vx_u8m2_b4(v631, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v633 = __riscv_vneg_v_i8m2(v607, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v634 = __riscv_vmerge_vvm_i8m2(v607, v633, v632, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v635 = __riscv_vsext_vf4_i32m8(v634, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v636 = __riscv_vfcvt_f_x_v_f32m8(v635, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v637 = __riscv_vfmul_vf_f32m8(v636, v570, 32);
    float* v638 = v13 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v638, v637, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v639 = v18 + 28;
    const uint8_t v640 = v639[0];
    uint32_t v641 = (uint32_t) v640;
    const uint8_t v642 = v639[1];
    uint32_t v643 = (uint32_t) v642;
    uint32_t v644 = v643 << 8u;
    uint32_t v645 = v641 | v644;
    const uint8_t v646 = v639[2];
    uint32_t v647 = (uint32_t) v646;
    uint32_t v648 = v647 << 16u;
    uint32_t v649 = v645 | v648;
    const uint8_t v650 = v639[3];
    uint32_t v651 = (uint32_t) v650;
    uint32_t v652 = v651 << 24u;
    uint32_t v653 = v649 | v652;
    uint32_t v654 = v653 >> 28u;
    int v655 = (int) v654;
    float v656 = (float) v655;
    float v657 = 0.5f + v656;
    float v658 = v14 * v657;
    float v659 = v658 * 0.5f;
    const uint8_t* v660 = v16 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    int32_t v661[8];
    const uint8_t v662 = v660[0];
    int v663 = (int) v662;
    const int32_t v664 = v7[v663];
    int32_t v665 = (int32_t) v664;
    v661[0] = v665;
    const uint8_t v666 = v660[1];
    int v667 = (int) v666;
    const int32_t v668 = v7[v667];
    int32_t v669 = (int32_t) v668;
    v661[1] = v669;
    const uint8_t v670 = v660[2];
    int v671 = (int) v670;
    const int32_t v672 = v7[v671];
    int32_t v673 = (int32_t) v672;
    v661[2] = v673;
    const uint8_t v674 = v660[3];
    int v675 = (int) v674;
    const int32_t v676 = v7[v675];
    int32_t v677 = (int32_t) v676;
    v661[3] = v677;
    const uint8_t v678 = v660[4];
    int v679 = (int) v678;
    const int32_t v680 = v7[v679];
    int32_t v681 = (int32_t) v680;
    v661[4] = v681;
    const uint8_t v682 = v660[5];
    int v683 = (int) v682;
    const int32_t v684 = v7[v683];
    int32_t v685 = (int32_t) v684;
    v661[5] = v685;
    const uint8_t v686 = v660[6];
    int v687 = (int) v686;
    const int32_t v688 = v7[v687];
    int32_t v689 = (int32_t) v688;
    v661[6] = v689;
    const uint8_t v690 = v660[7];
    int v691 = (int) v690;
    const int32_t v692 = v7[v691];
    int32_t v693 = (int32_t) v692;
    v661[7] = v693;
    int32_t* v694 = &v661[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m2
    vint32m2_t v695 = __riscv_vle32_v_i32m2(v694, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v696 = __riscv_vreinterpret_v_i32m2_i8m2(v695);
    uint8_t v697[32];
    uint32_t v698 = v653 >> 0u;
    uint32_t v699 = v698 & 127u;
    int v700 = (int) v699;
    const uint8_t v701 = weft_iq3xxs_ksigns[v700];
    uint8_t v702 = (uint8_t) v701;
    v697[0] = v702;
    v697[1] = v702;
    v697[2] = v702;
    v697[3] = v702;
    v697[4] = v702;
    v697[5] = v702;
    v697[6] = v702;
    v697[7] = v702;
    uint32_t v703 = v653 >> 7u;
    uint32_t v704 = v703 & 127u;
    int v705 = (int) v704;
    const uint8_t v706 = weft_iq3xxs_ksigns[v705];
    uint8_t v707 = (uint8_t) v706;
    v697[8] = v707;
    v697[9] = v707;
    v697[10] = v707;
    v697[11] = v707;
    v697[12] = v707;
    v697[13] = v707;
    v697[14] = v707;
    v697[15] = v707;
    uint32_t v708 = v653 >> 14u;
    uint32_t v709 = v708 & 127u;
    int v710 = (int) v709;
    const uint8_t v711 = weft_iq3xxs_ksigns[v710];
    uint8_t v712 = (uint8_t) v711;
    v697[16] = v712;
    v697[17] = v712;
    v697[18] = v712;
    v697[19] = v712;
    v697[20] = v712;
    v697[21] = v712;
    v697[22] = v712;
    v697[23] = v712;
    uint32_t v713 = v653 >> 21u;
    uint32_t v714 = v713 & 127u;
    int v715 = (int) v714;
    const uint8_t v716 = weft_iq3xxs_ksigns[v715];
    uint8_t v717 = (uint8_t) v716;
    v697[24] = v717;
    v697[25] = v717;
    v697[26] = v717;
    v697[27] = v717;
    v697[28] = v717;
    v697[29] = v717;
    v697[30] = v717;
    v697[31] = v717;
    uint8_t* v718 = &v697[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v719 = __riscv_vle8_v_u8m2(v718, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v720 = __riscv_vand_vv_u8m2(v719, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v721 = __riscv_vmsne_vx_u8m2_b4(v720, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v722 = __riscv_vneg_v_i8m2(v696, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v723 = __riscv_vmerge_vvm_i8m2(v696, v722, v721, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v724 = __riscv_vsext_vf4_i32m8(v723, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v725 = __riscv_vfcvt_f_x_v_f32m8(v724, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v726 = __riscv_vfmul_vf_f32m8(v725, v659, 32);
    float* v727 = v13 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v727, v726, 32);
  }
  return;
}


