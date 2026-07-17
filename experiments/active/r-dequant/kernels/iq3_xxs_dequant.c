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
  static const uint8_t weft_iq3xxs_sigspread[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m2_t v6 = __riscv_vle8_v_u8m2(weft_iq3xxs_kmask32, 32);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sigspread_table_load
  vuint8m2_t v7 = __riscv_vle8_v_u8m2(weft_iq3xxs_sigspread, 32);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_i32_view
  const int32_t* v8 = (const int32_t*) weft_iq3xxs_grid;
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
    uint16_t v40[8];
    const uint8_t v41 = v17[0];
    int v42 = (int) v41;
    int v43 = v42 << 2;
    uint16_t v44 = (uint16_t) v43;
    v40[0] = v44;
    const uint8_t v45 = v17[1];
    int v46 = (int) v45;
    int v47 = v46 << 2;
    uint16_t v48 = (uint16_t) v47;
    v40[1] = v48;
    const uint8_t v49 = v17[2];
    int v50 = (int) v49;
    int v51 = v50 << 2;
    uint16_t v52 = (uint16_t) v51;
    v40[2] = v52;
    const uint8_t v53 = v17[3];
    int v54 = (int) v53;
    int v55 = v54 << 2;
    uint16_t v56 = (uint16_t) v55;
    v40[3] = v56;
    const uint8_t v57 = v17[4];
    int v58 = (int) v57;
    int v59 = v58 << 2;
    uint16_t v60 = (uint16_t) v59;
    v40[4] = v60;
    const uint8_t v61 = v17[5];
    int v62 = (int) v61;
    int v63 = v62 << 2;
    uint16_t v64 = (uint16_t) v63;
    v40[5] = v64;
    const uint8_t v65 = v17[6];
    int v66 = (int) v65;
    int v67 = v66 << 2;
    uint16_t v68 = (uint16_t) v67;
    v40[6] = v68;
    const uint8_t v69 = v17[7];
    int v70 = (int) v69;
    int v71 = v70 << 2;
    uint16_t v72 = (uint16_t) v71;
    v40[7] = v72;
    uint16_t* v73 = &v40[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v74 = __riscv_vle16_v_u16m1(v73, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v75 = __riscv_vluxei16_v_i32m2(v8, v74, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v76 = __riscv_vreinterpret_v_i32m2_i8m2(v75);
    uint8_t v77[4];
    uint32_t v78 = v33 >> 0u;
    uint32_t v79 = v78 & 127u;
    int v80 = (int) v79;
    const uint8_t v81 = weft_iq3xxs_ksigns[v80];
    uint8_t v82 = (uint8_t) v81;
    v77[0] = v82;
    uint32_t v83 = v33 >> 7u;
    uint32_t v84 = v83 & 127u;
    int v85 = (int) v84;
    const uint8_t v86 = weft_iq3xxs_ksigns[v85];
    uint8_t v87 = (uint8_t) v86;
    v77[1] = v87;
    uint32_t v88 = v33 >> 14u;
    uint32_t v89 = v88 & 127u;
    int v90 = (int) v89;
    const uint8_t v91 = weft_iq3xxs_ksigns[v90];
    uint8_t v92 = (uint8_t) v91;
    v77[2] = v92;
    uint32_t v93 = v33 >> 21u;
    uint32_t v94 = v93 & 127u;
    int v95 = (int) v94;
    const uint8_t v96 = weft_iq3xxs_ksigns[v95];
    uint8_t v97 = (uint8_t) v96;
    v77[3] = v97;
    uint8_t* v98 = &v77[0];
    const uint8_t* v99 = (const uint8_t*) v98;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v100 = __riscv_vluxei8_v_u8m2(v99, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v101 = __riscv_vand_vv_u8m2(v100, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v102 = __riscv_vmsne_vx_u8m2_b4(v101, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v103 = __riscv_vneg_v_i8m2(v76, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v104 = __riscv_vmerge_vvm_i8m2(v76, v103, v102, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v105 = __riscv_vsext_vf4_i32m8(v104, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v106 = __riscv_vfcvt_f_x_v_f32m8(v105, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v107 = __riscv_vfmul_vf_f32m8(v106, v39, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v14, v107, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v108 = v19 + 4;
    const uint8_t v109 = v108[0];
    uint32_t v110 = (uint32_t) v109;
    const uint8_t v111 = v108[1];
    uint32_t v112 = (uint32_t) v111;
    uint32_t v113 = v112 << 8u;
    uint32_t v114 = v110 | v113;
    const uint8_t v115 = v108[2];
    uint32_t v116 = (uint32_t) v115;
    uint32_t v117 = v116 << 16u;
    uint32_t v118 = v114 | v117;
    const uint8_t v119 = v108[3];
    uint32_t v120 = (uint32_t) v119;
    uint32_t v121 = v120 << 24u;
    uint32_t v122 = v118 | v121;
    uint32_t v123 = v122 >> 28u;
    int v124 = (int) v123;
    float v125 = (float) v124;
    float v126 = 0.5f + v125;
    float v127 = v15 * v126;
    float v128 = v127 * 0.5f;
    const uint8_t* v129 = v17 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v130[8];
    const uint8_t v131 = v129[0];
    int v132 = (int) v131;
    int v133 = v132 << 2;
    uint16_t v134 = (uint16_t) v133;
    v130[0] = v134;
    const uint8_t v135 = v129[1];
    int v136 = (int) v135;
    int v137 = v136 << 2;
    uint16_t v138 = (uint16_t) v137;
    v130[1] = v138;
    const uint8_t v139 = v129[2];
    int v140 = (int) v139;
    int v141 = v140 << 2;
    uint16_t v142 = (uint16_t) v141;
    v130[2] = v142;
    const uint8_t v143 = v129[3];
    int v144 = (int) v143;
    int v145 = v144 << 2;
    uint16_t v146 = (uint16_t) v145;
    v130[3] = v146;
    const uint8_t v147 = v129[4];
    int v148 = (int) v147;
    int v149 = v148 << 2;
    uint16_t v150 = (uint16_t) v149;
    v130[4] = v150;
    const uint8_t v151 = v129[5];
    int v152 = (int) v151;
    int v153 = v152 << 2;
    uint16_t v154 = (uint16_t) v153;
    v130[5] = v154;
    const uint8_t v155 = v129[6];
    int v156 = (int) v155;
    int v157 = v156 << 2;
    uint16_t v158 = (uint16_t) v157;
    v130[6] = v158;
    const uint8_t v159 = v129[7];
    int v160 = (int) v159;
    int v161 = v160 << 2;
    uint16_t v162 = (uint16_t) v161;
    v130[7] = v162;
    uint16_t* v163 = &v130[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v164 = __riscv_vle16_v_u16m1(v163, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v165 = __riscv_vluxei16_v_i32m2(v8, v164, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v166 = __riscv_vreinterpret_v_i32m2_i8m2(v165);
    uint8_t v167[4];
    uint32_t v168 = v122 >> 0u;
    uint32_t v169 = v168 & 127u;
    int v170 = (int) v169;
    const uint8_t v171 = weft_iq3xxs_ksigns[v170];
    uint8_t v172 = (uint8_t) v171;
    v167[0] = v172;
    uint32_t v173 = v122 >> 7u;
    uint32_t v174 = v173 & 127u;
    int v175 = (int) v174;
    const uint8_t v176 = weft_iq3xxs_ksigns[v175];
    uint8_t v177 = (uint8_t) v176;
    v167[1] = v177;
    uint32_t v178 = v122 >> 14u;
    uint32_t v179 = v178 & 127u;
    int v180 = (int) v179;
    const uint8_t v181 = weft_iq3xxs_ksigns[v180];
    uint8_t v182 = (uint8_t) v181;
    v167[2] = v182;
    uint32_t v183 = v122 >> 21u;
    uint32_t v184 = v183 & 127u;
    int v185 = (int) v184;
    const uint8_t v186 = weft_iq3xxs_ksigns[v185];
    uint8_t v187 = (uint8_t) v186;
    v167[3] = v187;
    uint8_t* v188 = &v167[0];
    const uint8_t* v189 = (const uint8_t*) v188;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v190 = __riscv_vluxei8_v_u8m2(v189, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v191 = __riscv_vand_vv_u8m2(v190, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v192 = __riscv_vmsne_vx_u8m2_b4(v191, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v193 = __riscv_vneg_v_i8m2(v166, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v194 = __riscv_vmerge_vvm_i8m2(v166, v193, v192, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v195 = __riscv_vsext_vf4_i32m8(v194, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v196 = __riscv_vfcvt_f_x_v_f32m8(v195, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v197 = __riscv_vfmul_vf_f32m8(v196, v128, 32);
    float* v198 = v14 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v198, v197, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v199 = v19 + 8;
    const uint8_t v200 = v199[0];
    uint32_t v201 = (uint32_t) v200;
    const uint8_t v202 = v199[1];
    uint32_t v203 = (uint32_t) v202;
    uint32_t v204 = v203 << 8u;
    uint32_t v205 = v201 | v204;
    const uint8_t v206 = v199[2];
    uint32_t v207 = (uint32_t) v206;
    uint32_t v208 = v207 << 16u;
    uint32_t v209 = v205 | v208;
    const uint8_t v210 = v199[3];
    uint32_t v211 = (uint32_t) v210;
    uint32_t v212 = v211 << 24u;
    uint32_t v213 = v209 | v212;
    uint32_t v214 = v213 >> 28u;
    int v215 = (int) v214;
    float v216 = (float) v215;
    float v217 = 0.5f + v216;
    float v218 = v15 * v217;
    float v219 = v218 * 0.5f;
    const uint8_t* v220 = v17 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v221[8];
    const uint8_t v222 = v220[0];
    int v223 = (int) v222;
    int v224 = v223 << 2;
    uint16_t v225 = (uint16_t) v224;
    v221[0] = v225;
    const uint8_t v226 = v220[1];
    int v227 = (int) v226;
    int v228 = v227 << 2;
    uint16_t v229 = (uint16_t) v228;
    v221[1] = v229;
    const uint8_t v230 = v220[2];
    int v231 = (int) v230;
    int v232 = v231 << 2;
    uint16_t v233 = (uint16_t) v232;
    v221[2] = v233;
    const uint8_t v234 = v220[3];
    int v235 = (int) v234;
    int v236 = v235 << 2;
    uint16_t v237 = (uint16_t) v236;
    v221[3] = v237;
    const uint8_t v238 = v220[4];
    int v239 = (int) v238;
    int v240 = v239 << 2;
    uint16_t v241 = (uint16_t) v240;
    v221[4] = v241;
    const uint8_t v242 = v220[5];
    int v243 = (int) v242;
    int v244 = v243 << 2;
    uint16_t v245 = (uint16_t) v244;
    v221[5] = v245;
    const uint8_t v246 = v220[6];
    int v247 = (int) v246;
    int v248 = v247 << 2;
    uint16_t v249 = (uint16_t) v248;
    v221[6] = v249;
    const uint8_t v250 = v220[7];
    int v251 = (int) v250;
    int v252 = v251 << 2;
    uint16_t v253 = (uint16_t) v252;
    v221[7] = v253;
    uint16_t* v254 = &v221[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v255 = __riscv_vle16_v_u16m1(v254, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v256 = __riscv_vluxei16_v_i32m2(v8, v255, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v257 = __riscv_vreinterpret_v_i32m2_i8m2(v256);
    uint8_t v258[4];
    uint32_t v259 = v213 >> 0u;
    uint32_t v260 = v259 & 127u;
    int v261 = (int) v260;
    const uint8_t v262 = weft_iq3xxs_ksigns[v261];
    uint8_t v263 = (uint8_t) v262;
    v258[0] = v263;
    uint32_t v264 = v213 >> 7u;
    uint32_t v265 = v264 & 127u;
    int v266 = (int) v265;
    const uint8_t v267 = weft_iq3xxs_ksigns[v266];
    uint8_t v268 = (uint8_t) v267;
    v258[1] = v268;
    uint32_t v269 = v213 >> 14u;
    uint32_t v270 = v269 & 127u;
    int v271 = (int) v270;
    const uint8_t v272 = weft_iq3xxs_ksigns[v271];
    uint8_t v273 = (uint8_t) v272;
    v258[2] = v273;
    uint32_t v274 = v213 >> 21u;
    uint32_t v275 = v274 & 127u;
    int v276 = (int) v275;
    const uint8_t v277 = weft_iq3xxs_ksigns[v276];
    uint8_t v278 = (uint8_t) v277;
    v258[3] = v278;
    uint8_t* v279 = &v258[0];
    const uint8_t* v280 = (const uint8_t*) v279;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v281 = __riscv_vluxei8_v_u8m2(v280, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v282 = __riscv_vand_vv_u8m2(v281, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v283 = __riscv_vmsne_vx_u8m2_b4(v282, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v284 = __riscv_vneg_v_i8m2(v257, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v285 = __riscv_vmerge_vvm_i8m2(v257, v284, v283, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v286 = __riscv_vsext_vf4_i32m8(v285, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v287 = __riscv_vfcvt_f_x_v_f32m8(v286, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v288 = __riscv_vfmul_vf_f32m8(v287, v219, 32);
    float* v289 = v14 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v289, v288, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v290 = v19 + 12;
    const uint8_t v291 = v290[0];
    uint32_t v292 = (uint32_t) v291;
    const uint8_t v293 = v290[1];
    uint32_t v294 = (uint32_t) v293;
    uint32_t v295 = v294 << 8u;
    uint32_t v296 = v292 | v295;
    const uint8_t v297 = v290[2];
    uint32_t v298 = (uint32_t) v297;
    uint32_t v299 = v298 << 16u;
    uint32_t v300 = v296 | v299;
    const uint8_t v301 = v290[3];
    uint32_t v302 = (uint32_t) v301;
    uint32_t v303 = v302 << 24u;
    uint32_t v304 = v300 | v303;
    uint32_t v305 = v304 >> 28u;
    int v306 = (int) v305;
    float v307 = (float) v306;
    float v308 = 0.5f + v307;
    float v309 = v15 * v308;
    float v310 = v309 * 0.5f;
    const uint8_t* v311 = v17 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v312[8];
    const uint8_t v313 = v311[0];
    int v314 = (int) v313;
    int v315 = v314 << 2;
    uint16_t v316 = (uint16_t) v315;
    v312[0] = v316;
    const uint8_t v317 = v311[1];
    int v318 = (int) v317;
    int v319 = v318 << 2;
    uint16_t v320 = (uint16_t) v319;
    v312[1] = v320;
    const uint8_t v321 = v311[2];
    int v322 = (int) v321;
    int v323 = v322 << 2;
    uint16_t v324 = (uint16_t) v323;
    v312[2] = v324;
    const uint8_t v325 = v311[3];
    int v326 = (int) v325;
    int v327 = v326 << 2;
    uint16_t v328 = (uint16_t) v327;
    v312[3] = v328;
    const uint8_t v329 = v311[4];
    int v330 = (int) v329;
    int v331 = v330 << 2;
    uint16_t v332 = (uint16_t) v331;
    v312[4] = v332;
    const uint8_t v333 = v311[5];
    int v334 = (int) v333;
    int v335 = v334 << 2;
    uint16_t v336 = (uint16_t) v335;
    v312[5] = v336;
    const uint8_t v337 = v311[6];
    int v338 = (int) v337;
    int v339 = v338 << 2;
    uint16_t v340 = (uint16_t) v339;
    v312[6] = v340;
    const uint8_t v341 = v311[7];
    int v342 = (int) v341;
    int v343 = v342 << 2;
    uint16_t v344 = (uint16_t) v343;
    v312[7] = v344;
    uint16_t* v345 = &v312[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v346 = __riscv_vle16_v_u16m1(v345, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v347 = __riscv_vluxei16_v_i32m2(v8, v346, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v348 = __riscv_vreinterpret_v_i32m2_i8m2(v347);
    uint8_t v349[4];
    uint32_t v350 = v304 >> 0u;
    uint32_t v351 = v350 & 127u;
    int v352 = (int) v351;
    const uint8_t v353 = weft_iq3xxs_ksigns[v352];
    uint8_t v354 = (uint8_t) v353;
    v349[0] = v354;
    uint32_t v355 = v304 >> 7u;
    uint32_t v356 = v355 & 127u;
    int v357 = (int) v356;
    const uint8_t v358 = weft_iq3xxs_ksigns[v357];
    uint8_t v359 = (uint8_t) v358;
    v349[1] = v359;
    uint32_t v360 = v304 >> 14u;
    uint32_t v361 = v360 & 127u;
    int v362 = (int) v361;
    const uint8_t v363 = weft_iq3xxs_ksigns[v362];
    uint8_t v364 = (uint8_t) v363;
    v349[2] = v364;
    uint32_t v365 = v304 >> 21u;
    uint32_t v366 = v365 & 127u;
    int v367 = (int) v366;
    const uint8_t v368 = weft_iq3xxs_ksigns[v367];
    uint8_t v369 = (uint8_t) v368;
    v349[3] = v369;
    uint8_t* v370 = &v349[0];
    const uint8_t* v371 = (const uint8_t*) v370;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v372 = __riscv_vluxei8_v_u8m2(v371, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v373 = __riscv_vand_vv_u8m2(v372, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v374 = __riscv_vmsne_vx_u8m2_b4(v373, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v375 = __riscv_vneg_v_i8m2(v348, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v376 = __riscv_vmerge_vvm_i8m2(v348, v375, v374, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v377 = __riscv_vsext_vf4_i32m8(v376, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v378 = __riscv_vfcvt_f_x_v_f32m8(v377, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v379 = __riscv_vfmul_vf_f32m8(v378, v310, 32);
    float* v380 = v14 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v380, v379, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v381 = v19 + 16;
    const uint8_t v382 = v381[0];
    uint32_t v383 = (uint32_t) v382;
    const uint8_t v384 = v381[1];
    uint32_t v385 = (uint32_t) v384;
    uint32_t v386 = v385 << 8u;
    uint32_t v387 = v383 | v386;
    const uint8_t v388 = v381[2];
    uint32_t v389 = (uint32_t) v388;
    uint32_t v390 = v389 << 16u;
    uint32_t v391 = v387 | v390;
    const uint8_t v392 = v381[3];
    uint32_t v393 = (uint32_t) v392;
    uint32_t v394 = v393 << 24u;
    uint32_t v395 = v391 | v394;
    uint32_t v396 = v395 >> 28u;
    int v397 = (int) v396;
    float v398 = (float) v397;
    float v399 = 0.5f + v398;
    float v400 = v15 * v399;
    float v401 = v400 * 0.5f;
    const uint8_t* v402 = v17 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v403[8];
    const uint8_t v404 = v402[0];
    int v405 = (int) v404;
    int v406 = v405 << 2;
    uint16_t v407 = (uint16_t) v406;
    v403[0] = v407;
    const uint8_t v408 = v402[1];
    int v409 = (int) v408;
    int v410 = v409 << 2;
    uint16_t v411 = (uint16_t) v410;
    v403[1] = v411;
    const uint8_t v412 = v402[2];
    int v413 = (int) v412;
    int v414 = v413 << 2;
    uint16_t v415 = (uint16_t) v414;
    v403[2] = v415;
    const uint8_t v416 = v402[3];
    int v417 = (int) v416;
    int v418 = v417 << 2;
    uint16_t v419 = (uint16_t) v418;
    v403[3] = v419;
    const uint8_t v420 = v402[4];
    int v421 = (int) v420;
    int v422 = v421 << 2;
    uint16_t v423 = (uint16_t) v422;
    v403[4] = v423;
    const uint8_t v424 = v402[5];
    int v425 = (int) v424;
    int v426 = v425 << 2;
    uint16_t v427 = (uint16_t) v426;
    v403[5] = v427;
    const uint8_t v428 = v402[6];
    int v429 = (int) v428;
    int v430 = v429 << 2;
    uint16_t v431 = (uint16_t) v430;
    v403[6] = v431;
    const uint8_t v432 = v402[7];
    int v433 = (int) v432;
    int v434 = v433 << 2;
    uint16_t v435 = (uint16_t) v434;
    v403[7] = v435;
    uint16_t* v436 = &v403[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v437 = __riscv_vle16_v_u16m1(v436, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v438 = __riscv_vluxei16_v_i32m2(v8, v437, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v439 = __riscv_vreinterpret_v_i32m2_i8m2(v438);
    uint8_t v440[4];
    uint32_t v441 = v395 >> 0u;
    uint32_t v442 = v441 & 127u;
    int v443 = (int) v442;
    const uint8_t v444 = weft_iq3xxs_ksigns[v443];
    uint8_t v445 = (uint8_t) v444;
    v440[0] = v445;
    uint32_t v446 = v395 >> 7u;
    uint32_t v447 = v446 & 127u;
    int v448 = (int) v447;
    const uint8_t v449 = weft_iq3xxs_ksigns[v448];
    uint8_t v450 = (uint8_t) v449;
    v440[1] = v450;
    uint32_t v451 = v395 >> 14u;
    uint32_t v452 = v451 & 127u;
    int v453 = (int) v452;
    const uint8_t v454 = weft_iq3xxs_ksigns[v453];
    uint8_t v455 = (uint8_t) v454;
    v440[2] = v455;
    uint32_t v456 = v395 >> 21u;
    uint32_t v457 = v456 & 127u;
    int v458 = (int) v457;
    const uint8_t v459 = weft_iq3xxs_ksigns[v458];
    uint8_t v460 = (uint8_t) v459;
    v440[3] = v460;
    uint8_t* v461 = &v440[0];
    const uint8_t* v462 = (const uint8_t*) v461;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v463 = __riscv_vluxei8_v_u8m2(v462, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v464 = __riscv_vand_vv_u8m2(v463, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v465 = __riscv_vmsne_vx_u8m2_b4(v464, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v466 = __riscv_vneg_v_i8m2(v439, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v467 = __riscv_vmerge_vvm_i8m2(v439, v466, v465, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v468 = __riscv_vsext_vf4_i32m8(v467, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v469 = __riscv_vfcvt_f_x_v_f32m8(v468, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v470 = __riscv_vfmul_vf_f32m8(v469, v401, 32);
    float* v471 = v14 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v471, v470, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v472 = v19 + 20;
    const uint8_t v473 = v472[0];
    uint32_t v474 = (uint32_t) v473;
    const uint8_t v475 = v472[1];
    uint32_t v476 = (uint32_t) v475;
    uint32_t v477 = v476 << 8u;
    uint32_t v478 = v474 | v477;
    const uint8_t v479 = v472[2];
    uint32_t v480 = (uint32_t) v479;
    uint32_t v481 = v480 << 16u;
    uint32_t v482 = v478 | v481;
    const uint8_t v483 = v472[3];
    uint32_t v484 = (uint32_t) v483;
    uint32_t v485 = v484 << 24u;
    uint32_t v486 = v482 | v485;
    uint32_t v487 = v486 >> 28u;
    int v488 = (int) v487;
    float v489 = (float) v488;
    float v490 = 0.5f + v489;
    float v491 = v15 * v490;
    float v492 = v491 * 0.5f;
    const uint8_t* v493 = v17 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v494[8];
    const uint8_t v495 = v493[0];
    int v496 = (int) v495;
    int v497 = v496 << 2;
    uint16_t v498 = (uint16_t) v497;
    v494[0] = v498;
    const uint8_t v499 = v493[1];
    int v500 = (int) v499;
    int v501 = v500 << 2;
    uint16_t v502 = (uint16_t) v501;
    v494[1] = v502;
    const uint8_t v503 = v493[2];
    int v504 = (int) v503;
    int v505 = v504 << 2;
    uint16_t v506 = (uint16_t) v505;
    v494[2] = v506;
    const uint8_t v507 = v493[3];
    int v508 = (int) v507;
    int v509 = v508 << 2;
    uint16_t v510 = (uint16_t) v509;
    v494[3] = v510;
    const uint8_t v511 = v493[4];
    int v512 = (int) v511;
    int v513 = v512 << 2;
    uint16_t v514 = (uint16_t) v513;
    v494[4] = v514;
    const uint8_t v515 = v493[5];
    int v516 = (int) v515;
    int v517 = v516 << 2;
    uint16_t v518 = (uint16_t) v517;
    v494[5] = v518;
    const uint8_t v519 = v493[6];
    int v520 = (int) v519;
    int v521 = v520 << 2;
    uint16_t v522 = (uint16_t) v521;
    v494[6] = v522;
    const uint8_t v523 = v493[7];
    int v524 = (int) v523;
    int v525 = v524 << 2;
    uint16_t v526 = (uint16_t) v525;
    v494[7] = v526;
    uint16_t* v527 = &v494[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v528 = __riscv_vle16_v_u16m1(v527, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v529 = __riscv_vluxei16_v_i32m2(v8, v528, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v530 = __riscv_vreinterpret_v_i32m2_i8m2(v529);
    uint8_t v531[4];
    uint32_t v532 = v486 >> 0u;
    uint32_t v533 = v532 & 127u;
    int v534 = (int) v533;
    const uint8_t v535 = weft_iq3xxs_ksigns[v534];
    uint8_t v536 = (uint8_t) v535;
    v531[0] = v536;
    uint32_t v537 = v486 >> 7u;
    uint32_t v538 = v537 & 127u;
    int v539 = (int) v538;
    const uint8_t v540 = weft_iq3xxs_ksigns[v539];
    uint8_t v541 = (uint8_t) v540;
    v531[1] = v541;
    uint32_t v542 = v486 >> 14u;
    uint32_t v543 = v542 & 127u;
    int v544 = (int) v543;
    const uint8_t v545 = weft_iq3xxs_ksigns[v544];
    uint8_t v546 = (uint8_t) v545;
    v531[2] = v546;
    uint32_t v547 = v486 >> 21u;
    uint32_t v548 = v547 & 127u;
    int v549 = (int) v548;
    const uint8_t v550 = weft_iq3xxs_ksigns[v549];
    uint8_t v551 = (uint8_t) v550;
    v531[3] = v551;
    uint8_t* v552 = &v531[0];
    const uint8_t* v553 = (const uint8_t*) v552;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v554 = __riscv_vluxei8_v_u8m2(v553, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v555 = __riscv_vand_vv_u8m2(v554, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v556 = __riscv_vmsne_vx_u8m2_b4(v555, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v557 = __riscv_vneg_v_i8m2(v530, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v558 = __riscv_vmerge_vvm_i8m2(v530, v557, v556, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v559 = __riscv_vsext_vf4_i32m8(v558, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v560 = __riscv_vfcvt_f_x_v_f32m8(v559, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v561 = __riscv_vfmul_vf_f32m8(v560, v492, 32);
    float* v562 = v14 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v562, v561, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v563 = v19 + 24;
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
    float v580 = (float) v579;
    float v581 = 0.5f + v580;
    float v582 = v15 * v581;
    float v583 = v582 * 0.5f;
    const uint8_t* v584 = v17 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v585[8];
    const uint8_t v586 = v584[0];
    int v587 = (int) v586;
    int v588 = v587 << 2;
    uint16_t v589 = (uint16_t) v588;
    v585[0] = v589;
    const uint8_t v590 = v584[1];
    int v591 = (int) v590;
    int v592 = v591 << 2;
    uint16_t v593 = (uint16_t) v592;
    v585[1] = v593;
    const uint8_t v594 = v584[2];
    int v595 = (int) v594;
    int v596 = v595 << 2;
    uint16_t v597 = (uint16_t) v596;
    v585[2] = v597;
    const uint8_t v598 = v584[3];
    int v599 = (int) v598;
    int v600 = v599 << 2;
    uint16_t v601 = (uint16_t) v600;
    v585[3] = v601;
    const uint8_t v602 = v584[4];
    int v603 = (int) v602;
    int v604 = v603 << 2;
    uint16_t v605 = (uint16_t) v604;
    v585[4] = v605;
    const uint8_t v606 = v584[5];
    int v607 = (int) v606;
    int v608 = v607 << 2;
    uint16_t v609 = (uint16_t) v608;
    v585[5] = v609;
    const uint8_t v610 = v584[6];
    int v611 = (int) v610;
    int v612 = v611 << 2;
    uint16_t v613 = (uint16_t) v612;
    v585[6] = v613;
    const uint8_t v614 = v584[7];
    int v615 = (int) v614;
    int v616 = v615 << 2;
    uint16_t v617 = (uint16_t) v616;
    v585[7] = v617;
    uint16_t* v618 = &v585[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v619 = __riscv_vle16_v_u16m1(v618, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v620 = __riscv_vluxei16_v_i32m2(v8, v619, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v621 = __riscv_vreinterpret_v_i32m2_i8m2(v620);
    uint8_t v622[4];
    uint32_t v623 = v577 >> 0u;
    uint32_t v624 = v623 & 127u;
    int v625 = (int) v624;
    const uint8_t v626 = weft_iq3xxs_ksigns[v625];
    uint8_t v627 = (uint8_t) v626;
    v622[0] = v627;
    uint32_t v628 = v577 >> 7u;
    uint32_t v629 = v628 & 127u;
    int v630 = (int) v629;
    const uint8_t v631 = weft_iq3xxs_ksigns[v630];
    uint8_t v632 = (uint8_t) v631;
    v622[1] = v632;
    uint32_t v633 = v577 >> 14u;
    uint32_t v634 = v633 & 127u;
    int v635 = (int) v634;
    const uint8_t v636 = weft_iq3xxs_ksigns[v635];
    uint8_t v637 = (uint8_t) v636;
    v622[2] = v637;
    uint32_t v638 = v577 >> 21u;
    uint32_t v639 = v638 & 127u;
    int v640 = (int) v639;
    const uint8_t v641 = weft_iq3xxs_ksigns[v640];
    uint8_t v642 = (uint8_t) v641;
    v622[3] = v642;
    uint8_t* v643 = &v622[0];
    const uint8_t* v644 = (const uint8_t*) v643;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v645 = __riscv_vluxei8_v_u8m2(v644, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v646 = __riscv_vand_vv_u8m2(v645, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v647 = __riscv_vmsne_vx_u8m2_b4(v646, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v648 = __riscv_vneg_v_i8m2(v621, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v649 = __riscv_vmerge_vvm_i8m2(v621, v648, v647, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v650 = __riscv_vsext_vf4_i32m8(v649, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v651 = __riscv_vfcvt_f_x_v_f32m8(v650, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v652 = __riscv_vfmul_vf_f32m8(v651, v583, 32);
    float* v653 = v14 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v653, v652, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v654 = v19 + 28;
    const uint8_t v655 = v654[0];
    uint32_t v656 = (uint32_t) v655;
    const uint8_t v657 = v654[1];
    uint32_t v658 = (uint32_t) v657;
    uint32_t v659 = v658 << 8u;
    uint32_t v660 = v656 | v659;
    const uint8_t v661 = v654[2];
    uint32_t v662 = (uint32_t) v661;
    uint32_t v663 = v662 << 16u;
    uint32_t v664 = v660 | v663;
    const uint8_t v665 = v654[3];
    uint32_t v666 = (uint32_t) v665;
    uint32_t v667 = v666 << 24u;
    uint32_t v668 = v664 | v667;
    uint32_t v669 = v668 >> 28u;
    int v670 = (int) v669;
    float v671 = (float) v670;
    float v672 = 0.5f + v671;
    float v673 = v15 * v672;
    float v674 = v673 * 0.5f;
    const uint8_t* v675 = v17 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t v676[8];
    const uint8_t v677 = v675[0];
    int v678 = (int) v677;
    int v679 = v678 << 2;
    uint16_t v680 = (uint16_t) v679;
    v676[0] = v680;
    const uint8_t v681 = v675[1];
    int v682 = (int) v681;
    int v683 = v682 << 2;
    uint16_t v684 = (uint16_t) v683;
    v676[1] = v684;
    const uint8_t v685 = v675[2];
    int v686 = (int) v685;
    int v687 = v686 << 2;
    uint16_t v688 = (uint16_t) v687;
    v676[2] = v688;
    const uint8_t v689 = v675[3];
    int v690 = (int) v689;
    int v691 = v690 << 2;
    uint16_t v692 = (uint16_t) v691;
    v676[3] = v692;
    const uint8_t v693 = v675[4];
    int v694 = (int) v693;
    int v695 = v694 << 2;
    uint16_t v696 = (uint16_t) v695;
    v676[4] = v696;
    const uint8_t v697 = v675[5];
    int v698 = (int) v697;
    int v699 = v698 << 2;
    uint16_t v700 = (uint16_t) v699;
    v676[5] = v700;
    const uint8_t v701 = v675[6];
    int v702 = (int) v701;
    int v703 = v702 << 2;
    uint16_t v704 = (uint16_t) v703;
    v676[6] = v704;
    const uint8_t v705 = v675[7];
    int v706 = (int) v705;
    int v707 = v706 << 2;
    uint16_t v708 = (uint16_t) v707;
    v676[7] = v708;
    uint16_t* v709 = &v676[0];
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v710 = __riscv_vle16_v_u16m1(v709, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m2
    vint32m2_t v711 = __riscv_vluxei16_v_i32m2(v8, v710, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m2_i8m2
    vint8m2_t v712 = __riscv_vreinterpret_v_i32m2_i8m2(v711);
    uint8_t v713[4];
    uint32_t v714 = v668 >> 0u;
    uint32_t v715 = v714 & 127u;
    int v716 = (int) v715;
    const uint8_t v717 = weft_iq3xxs_ksigns[v716];
    uint8_t v718 = (uint8_t) v717;
    v713[0] = v718;
    uint32_t v719 = v668 >> 7u;
    uint32_t v720 = v719 & 127u;
    int v721 = (int) v720;
    const uint8_t v722 = weft_iq3xxs_ksigns[v721];
    uint8_t v723 = (uint8_t) v722;
    v713[1] = v723;
    uint32_t v724 = v668 >> 14u;
    uint32_t v725 = v724 & 127u;
    int v726 = (int) v725;
    const uint8_t v727 = weft_iq3xxs_ksigns[v726];
    uint8_t v728 = (uint8_t) v727;
    v713[2] = v728;
    uint32_t v729 = v668 >> 21u;
    uint32_t v730 = v729 & 127u;
    int v731 = (int) v730;
    const uint8_t v732 = weft_iq3xxs_ksigns[v731];
    uint8_t v733 = (uint8_t) v732;
    v713[3] = v733;
    uint8_t* v734 = &v713[0];
    const uint8_t* v735 = (const uint8_t*) v734;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei8_v_u8m2
    vuint8m2_t v736 = __riscv_vluxei8_v_u8m2(v735, v7, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m2
    vuint8m2_t v737 = __riscv_vand_vv_u8m2(v736, v6, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v738 = __riscv_vmsne_vx_u8m2_b4(v737, 0, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m2
    vint8m2_t v739 = __riscv_vneg_v_i8m2(v712, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m2
    vint8m2_t v740 = __riscv_vmerge_vvm_i8m2(v712, v739, v738, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m8
    vint32m8_t v741 = __riscv_vsext_vf4_i32m8(v740, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m8
    vfloat32m8_t v742 = __riscv_vfcvt_f_x_v_f32m8(v741, 32);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v743 = __riscv_vfmul_vf_f32m8(v742, v674, 32);
    float* v744 = v14 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m8
    __riscv_vse32_v_f32m8(v744, v743, 32);
  }
  return;
}


