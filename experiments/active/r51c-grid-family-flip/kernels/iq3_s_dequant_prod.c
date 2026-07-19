#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const uint32_t weft_iq3s_grid[512] = {0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU, 0x01010301U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU, 0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U, 0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U, 0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU, 0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U, 0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU, 0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU, 0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU, 0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU, 0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU, 0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U, 0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U, 0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U, 0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U, 0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U, 0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U, 0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U, 0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU, 0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U, 0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U, 0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U, 0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U, 0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U, 0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU, 0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U, 0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U, 0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U, 0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU, 0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U, 0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U, 0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU, 0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U, 0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U, 0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U, 0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U, 0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU, 0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U, 0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U, 0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU, 0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU, 0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U, 0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U, 0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U, 0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U, 0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U, 0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U, 0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU, 0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU, 0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U, 0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU, 0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U, 0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U, 0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U, 0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U, 0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U, 0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU, 0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU, 0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U, 0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U, 0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U, 0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U, 0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U, 0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U, 0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U, 0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U, 0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U, 0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U, 0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U, 0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U, 0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U, 0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U, 0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U, 0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U, 0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU, 0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U, 0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U, 0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U, 0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U, 0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U, 0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U, 0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U, 0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU, 0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U, 0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U, 0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U, 0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U, 0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU, 0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U, 0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U, 0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU, 0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU, 0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U, 0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U, 0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U, 0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU, 0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U, 0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U, 0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU, 0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U, 0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U, 0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U, 0x0f0d0703U, 0x0f0f0101U};
  static const uint8_t weft_iq3s_kmask_lo[4] = {1, 2, 4, 8};
  static const uint8_t weft_iq3s_kmask_hi[4] = {16, 32, 64, 128};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_lo_load
  vuint8mf4_t v6 = __riscv_vle8_v_u8mf4(weft_iq3s_kmask_lo, 4);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=kmask_hi_load
  vuint8mf4_t v7 = __riscv_vle8_v_u8mf4(weft_iq3s_kmask_hi, 4);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_u8_view
  const uint8_t* v8 = (const uint8_t*) weft_iq3s_grid;
  for (size_t v9 = 0; v9 < v5; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v10 = v9 * 110;
    const uint8_t* v11 = v2 + v10;
    const uint8_t* v12 = (const uint8_t*) v11;
    size_t v13 = v9 * 256;
    float* v14 = v3 + v13;
    float* v15 = (float*) v14;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=group_scale
    const uint8_t v17 = v12[106];
    int v18 = (int) v17;
    int v19 = v18 & 15;
    int v20 = 2 * v19;
    int v21 = 1 + v20;
    float v22 = (float) v21;
    float v23 = v16 * v22;
    int v24 = v18 >> 4;
    int v25 = 2 * v24;
    int v26 = 1 + v25;
    float v27 = (float) v26;
    float v28 = v16 * v27;
    const uint8_t v29 = v12[66];
    int v30 = (int) v29;
    const uint8_t v31 = v12[67];
    int v32 = (int) v31;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v33 = v12[2];
    int v34 = (int) v33;
    const uint8_t v35 = v12[3];
    int v36 = (int) v35;
    int v37 = v30 << 8;
    int v38 = v37 & 256;
    int v39 = v34 | v38;
    int v40 = v30 << 7;
    int v41 = v40 & 256;
    int v42 = v36 | v41;
    const uint8_t v43 = v12[74];
    int v44 = (int) v43;
    uint8_t v45 = (uint8_t) v44;
    size_t v46 = (size_t) v39;
    size_t v47 = v46 * 4;
    const uint8_t* v48 = v8 + v47;
    const int8_t* v49 = (const int8_t*) v48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v50 = __riscv_vle8_v_i8mf4(v49, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v51 = __riscv_vand_vx_u8mf4(v6, v45, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v52 = __riscv_vmsne_vx_u8mf4_b32(v51, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v53 = __riscv_vneg_v_i8mf4(v50, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v54 = __riscv_vmerge_vvm_i8mf4(v50, v53, v52, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v55 = __riscv_vsext_vf4_i32m1(v54, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v56 = __riscv_vfcvt_f_x_v_f32m1(v55, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v57 = __riscv_vfmul_vf_f32m1(v56, v23, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v15, v57, 4);
    size_t v58 = (size_t) v42;
    size_t v59 = v58 * 4;
    const uint8_t* v60 = v8 + v59;
    const int8_t* v61 = (const int8_t*) v60;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v62 = __riscv_vle8_v_i8mf4(v61, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v63 = __riscv_vand_vx_u8mf4(v7, v45, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v64 = __riscv_vmsne_vx_u8mf4_b32(v63, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v65 = __riscv_vneg_v_i8mf4(v62, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v66 = __riscv_vmerge_vvm_i8mf4(v62, v65, v64, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v67 = __riscv_vsext_vf4_i32m1(v66, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v68 = __riscv_vfcvt_f_x_v_f32m1(v67, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v69 = __riscv_vfmul_vf_f32m1(v68, v23, 4);
    float* v70 = v15 + 4;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v70, v69, 4);
    const uint8_t v71 = v12[4];
    int v72 = (int) v71;
    const uint8_t v73 = v12[5];
    int v74 = (int) v73;
    int v75 = v30 << 6;
    int v76 = v75 & 256;
    int v77 = v72 | v76;
    int v78 = v30 << 5;
    int v79 = v78 & 256;
    int v80 = v74 | v79;
    const uint8_t v81 = v12[75];
    int v82 = (int) v81;
    uint8_t v83 = (uint8_t) v82;
    size_t v84 = (size_t) v77;
    size_t v85 = v84 * 4;
    const uint8_t* v86 = v8 + v85;
    const int8_t* v87 = (const int8_t*) v86;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v88 = __riscv_vle8_v_i8mf4(v87, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v89 = __riscv_vand_vx_u8mf4(v6, v83, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v90 = __riscv_vmsne_vx_u8mf4_b32(v89, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v91 = __riscv_vneg_v_i8mf4(v88, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v92 = __riscv_vmerge_vvm_i8mf4(v88, v91, v90, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v93 = __riscv_vsext_vf4_i32m1(v92, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v94 = __riscv_vfcvt_f_x_v_f32m1(v93, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v95 = __riscv_vfmul_vf_f32m1(v94, v23, 4);
    float* v96 = v15 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v96, v95, 4);
    size_t v97 = (size_t) v80;
    size_t v98 = v97 * 4;
    const uint8_t* v99 = v8 + v98;
    const int8_t* v100 = (const int8_t*) v99;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v101 = __riscv_vle8_v_i8mf4(v100, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v102 = __riscv_vand_vx_u8mf4(v7, v83, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v103 = __riscv_vmsne_vx_u8mf4_b32(v102, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v104 = __riscv_vneg_v_i8mf4(v101, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v105 = __riscv_vmerge_vvm_i8mf4(v101, v104, v103, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v106 = __riscv_vsext_vf4_i32m1(v105, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v107 = __riscv_vfcvt_f_x_v_f32m1(v106, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v108 = __riscv_vfmul_vf_f32m1(v107, v23, 4);
    float* v109 = v15 + 12;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v109, v108, 4);
    const uint8_t v110 = v12[6];
    int v111 = (int) v110;
    const uint8_t v112 = v12[7];
    int v113 = (int) v112;
    int v114 = v30 << 4;
    int v115 = v114 & 256;
    int v116 = v111 | v115;
    int v117 = v30 << 3;
    int v118 = v117 & 256;
    int v119 = v113 | v118;
    const uint8_t v120 = v12[76];
    int v121 = (int) v120;
    uint8_t v122 = (uint8_t) v121;
    size_t v123 = (size_t) v116;
    size_t v124 = v123 * 4;
    const uint8_t* v125 = v8 + v124;
    const int8_t* v126 = (const int8_t*) v125;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v127 = __riscv_vle8_v_i8mf4(v126, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v128 = __riscv_vand_vx_u8mf4(v6, v122, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v129 = __riscv_vmsne_vx_u8mf4_b32(v128, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v130 = __riscv_vneg_v_i8mf4(v127, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v131 = __riscv_vmerge_vvm_i8mf4(v127, v130, v129, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v132 = __riscv_vsext_vf4_i32m1(v131, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v133 = __riscv_vfcvt_f_x_v_f32m1(v132, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v134 = __riscv_vfmul_vf_f32m1(v133, v23, 4);
    float* v135 = v15 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v135, v134, 4);
    size_t v136 = (size_t) v119;
    size_t v137 = v136 * 4;
    const uint8_t* v138 = v8 + v137;
    const int8_t* v139 = (const int8_t*) v138;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v140 = __riscv_vle8_v_i8mf4(v139, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v141 = __riscv_vand_vx_u8mf4(v7, v122, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v142 = __riscv_vmsne_vx_u8mf4_b32(v141, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v143 = __riscv_vneg_v_i8mf4(v140, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v144 = __riscv_vmerge_vvm_i8mf4(v140, v143, v142, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v145 = __riscv_vsext_vf4_i32m1(v144, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v146 = __riscv_vfcvt_f_x_v_f32m1(v145, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v147 = __riscv_vfmul_vf_f32m1(v146, v23, 4);
    float* v148 = v15 + 20;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v148, v147, 4);
    const uint8_t v149 = v12[8];
    int v150 = (int) v149;
    const uint8_t v151 = v12[9];
    int v152 = (int) v151;
    int v153 = v30 << 2;
    int v154 = v153 & 256;
    int v155 = v150 | v154;
    int v156 = v30 << 1;
    int v157 = v156 & 256;
    int v158 = v152 | v157;
    const uint8_t v159 = v12[77];
    int v160 = (int) v159;
    uint8_t v161 = (uint8_t) v160;
    size_t v162 = (size_t) v155;
    size_t v163 = v162 * 4;
    const uint8_t* v164 = v8 + v163;
    const int8_t* v165 = (const int8_t*) v164;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v166 = __riscv_vle8_v_i8mf4(v165, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v167 = __riscv_vand_vx_u8mf4(v6, v161, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v168 = __riscv_vmsne_vx_u8mf4_b32(v167, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v169 = __riscv_vneg_v_i8mf4(v166, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v170 = __riscv_vmerge_vvm_i8mf4(v166, v169, v168, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v171 = __riscv_vsext_vf4_i32m1(v170, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v172 = __riscv_vfcvt_f_x_v_f32m1(v171, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v173 = __riscv_vfmul_vf_f32m1(v172, v23, 4);
    float* v174 = v15 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v174, v173, 4);
    size_t v175 = (size_t) v158;
    size_t v176 = v175 * 4;
    const uint8_t* v177 = v8 + v176;
    const int8_t* v178 = (const int8_t*) v177;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v179 = __riscv_vle8_v_i8mf4(v178, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v180 = __riscv_vand_vx_u8mf4(v7, v161, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v181 = __riscv_vmsne_vx_u8mf4_b32(v180, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v182 = __riscv_vneg_v_i8mf4(v179, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v183 = __riscv_vmerge_vvm_i8mf4(v179, v182, v181, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v184 = __riscv_vsext_vf4_i32m1(v183, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v185 = __riscv_vfcvt_f_x_v_f32m1(v184, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v186 = __riscv_vfmul_vf_f32m1(v185, v23, 4);
    float* v187 = v15 + 28;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v187, v186, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v188 = v12[10];
    int v189 = (int) v188;
    const uint8_t v190 = v12[11];
    int v191 = (int) v190;
    int v192 = v32 << 8;
    int v193 = v192 & 256;
    int v194 = v189 | v193;
    int v195 = v32 << 7;
    int v196 = v195 & 256;
    int v197 = v191 | v196;
    const uint8_t v198 = v12[78];
    int v199 = (int) v198;
    uint8_t v200 = (uint8_t) v199;
    size_t v201 = (size_t) v194;
    size_t v202 = v201 * 4;
    const uint8_t* v203 = v8 + v202;
    const int8_t* v204 = (const int8_t*) v203;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v205 = __riscv_vle8_v_i8mf4(v204, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v206 = __riscv_vand_vx_u8mf4(v6, v200, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v207 = __riscv_vmsne_vx_u8mf4_b32(v206, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v208 = __riscv_vneg_v_i8mf4(v205, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v209 = __riscv_vmerge_vvm_i8mf4(v205, v208, v207, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v210 = __riscv_vsext_vf4_i32m1(v209, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v211 = __riscv_vfcvt_f_x_v_f32m1(v210, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v212 = __riscv_vfmul_vf_f32m1(v211, v28, 4);
    float* v213 = v15 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v213, v212, 4);
    size_t v214 = (size_t) v197;
    size_t v215 = v214 * 4;
    const uint8_t* v216 = v8 + v215;
    const int8_t* v217 = (const int8_t*) v216;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v218 = __riscv_vle8_v_i8mf4(v217, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v219 = __riscv_vand_vx_u8mf4(v7, v200, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v220 = __riscv_vmsne_vx_u8mf4_b32(v219, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v221 = __riscv_vneg_v_i8mf4(v218, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v222 = __riscv_vmerge_vvm_i8mf4(v218, v221, v220, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v223 = __riscv_vsext_vf4_i32m1(v222, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v224 = __riscv_vfcvt_f_x_v_f32m1(v223, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v225 = __riscv_vfmul_vf_f32m1(v224, v28, 4);
    float* v226 = v15 + 36;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v226, v225, 4);
    const uint8_t v227 = v12[12];
    int v228 = (int) v227;
    const uint8_t v229 = v12[13];
    int v230 = (int) v229;
    int v231 = v32 << 6;
    int v232 = v231 & 256;
    int v233 = v228 | v232;
    int v234 = v32 << 5;
    int v235 = v234 & 256;
    int v236 = v230 | v235;
    const uint8_t v237 = v12[79];
    int v238 = (int) v237;
    uint8_t v239 = (uint8_t) v238;
    size_t v240 = (size_t) v233;
    size_t v241 = v240 * 4;
    const uint8_t* v242 = v8 + v241;
    const int8_t* v243 = (const int8_t*) v242;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v244 = __riscv_vle8_v_i8mf4(v243, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v245 = __riscv_vand_vx_u8mf4(v6, v239, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v246 = __riscv_vmsne_vx_u8mf4_b32(v245, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v247 = __riscv_vneg_v_i8mf4(v244, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v248 = __riscv_vmerge_vvm_i8mf4(v244, v247, v246, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v249 = __riscv_vsext_vf4_i32m1(v248, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v250 = __riscv_vfcvt_f_x_v_f32m1(v249, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v251 = __riscv_vfmul_vf_f32m1(v250, v28, 4);
    float* v252 = v15 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v252, v251, 4);
    size_t v253 = (size_t) v236;
    size_t v254 = v253 * 4;
    const uint8_t* v255 = v8 + v254;
    const int8_t* v256 = (const int8_t*) v255;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v257 = __riscv_vle8_v_i8mf4(v256, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v258 = __riscv_vand_vx_u8mf4(v7, v239, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v259 = __riscv_vmsne_vx_u8mf4_b32(v258, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v260 = __riscv_vneg_v_i8mf4(v257, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v261 = __riscv_vmerge_vvm_i8mf4(v257, v260, v259, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v262 = __riscv_vsext_vf4_i32m1(v261, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v263 = __riscv_vfcvt_f_x_v_f32m1(v262, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v264 = __riscv_vfmul_vf_f32m1(v263, v28, 4);
    float* v265 = v15 + 44;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v265, v264, 4);
    const uint8_t v266 = v12[14];
    int v267 = (int) v266;
    const uint8_t v268 = v12[15];
    int v269 = (int) v268;
    int v270 = v32 << 4;
    int v271 = v270 & 256;
    int v272 = v267 | v271;
    int v273 = v32 << 3;
    int v274 = v273 & 256;
    int v275 = v269 | v274;
    const uint8_t v276 = v12[80];
    int v277 = (int) v276;
    uint8_t v278 = (uint8_t) v277;
    size_t v279 = (size_t) v272;
    size_t v280 = v279 * 4;
    const uint8_t* v281 = v8 + v280;
    const int8_t* v282 = (const int8_t*) v281;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v283 = __riscv_vle8_v_i8mf4(v282, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v284 = __riscv_vand_vx_u8mf4(v6, v278, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v285 = __riscv_vmsne_vx_u8mf4_b32(v284, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v286 = __riscv_vneg_v_i8mf4(v283, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v287 = __riscv_vmerge_vvm_i8mf4(v283, v286, v285, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v288 = __riscv_vsext_vf4_i32m1(v287, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v289 = __riscv_vfcvt_f_x_v_f32m1(v288, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v290 = __riscv_vfmul_vf_f32m1(v289, v28, 4);
    float* v291 = v15 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v291, v290, 4);
    size_t v292 = (size_t) v275;
    size_t v293 = v292 * 4;
    const uint8_t* v294 = v8 + v293;
    const int8_t* v295 = (const int8_t*) v294;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v296 = __riscv_vle8_v_i8mf4(v295, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v297 = __riscv_vand_vx_u8mf4(v7, v278, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v298 = __riscv_vmsne_vx_u8mf4_b32(v297, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v299 = __riscv_vneg_v_i8mf4(v296, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v300 = __riscv_vmerge_vvm_i8mf4(v296, v299, v298, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v301 = __riscv_vsext_vf4_i32m1(v300, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v302 = __riscv_vfcvt_f_x_v_f32m1(v301, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v303 = __riscv_vfmul_vf_f32m1(v302, v28, 4);
    float* v304 = v15 + 52;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v304, v303, 4);
    const uint8_t v305 = v12[16];
    int v306 = (int) v305;
    const uint8_t v307 = v12[17];
    int v308 = (int) v307;
    int v309 = v32 << 2;
    int v310 = v309 & 256;
    int v311 = v306 | v310;
    int v312 = v32 << 1;
    int v313 = v312 & 256;
    int v314 = v308 | v313;
    const uint8_t v315 = v12[81];
    int v316 = (int) v315;
    uint8_t v317 = (uint8_t) v316;
    size_t v318 = (size_t) v311;
    size_t v319 = v318 * 4;
    const uint8_t* v320 = v8 + v319;
    const int8_t* v321 = (const int8_t*) v320;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v322 = __riscv_vle8_v_i8mf4(v321, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v323 = __riscv_vand_vx_u8mf4(v6, v317, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v324 = __riscv_vmsne_vx_u8mf4_b32(v323, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v325 = __riscv_vneg_v_i8mf4(v322, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v326 = __riscv_vmerge_vvm_i8mf4(v322, v325, v324, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v327 = __riscv_vsext_vf4_i32m1(v326, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v328 = __riscv_vfcvt_f_x_v_f32m1(v327, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v329 = __riscv_vfmul_vf_f32m1(v328, v28, 4);
    float* v330 = v15 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v330, v329, 4);
    size_t v331 = (size_t) v314;
    size_t v332 = v331 * 4;
    const uint8_t* v333 = v8 + v332;
    const int8_t* v334 = (const int8_t*) v333;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v335 = __riscv_vle8_v_i8mf4(v334, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v336 = __riscv_vand_vx_u8mf4(v7, v317, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v337 = __riscv_vmsne_vx_u8mf4_b32(v336, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v338 = __riscv_vneg_v_i8mf4(v335, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v339 = __riscv_vmerge_vvm_i8mf4(v335, v338, v337, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v340 = __riscv_vsext_vf4_i32m1(v339, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v341 = __riscv_vfcvt_f_x_v_f32m1(v340, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v342 = __riscv_vfmul_vf_f32m1(v341, v28, 4);
    float* v343 = v15 + 60;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v343, v342, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=group_scale
    const uint8_t v344 = v12[107];
    int v345 = (int) v344;
    int v346 = v345 & 15;
    int v347 = 2 * v346;
    int v348 = 1 + v347;
    float v349 = (float) v348;
    float v350 = v16 * v349;
    int v351 = v345 >> 4;
    int v352 = 2 * v351;
    int v353 = 1 + v352;
    float v354 = (float) v353;
    float v355 = v16 * v354;
    const uint8_t v356 = v12[68];
    int v357 = (int) v356;
    const uint8_t v358 = v12[69];
    int v359 = (int) v358;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v360 = v12[18];
    int v361 = (int) v360;
    const uint8_t v362 = v12[19];
    int v363 = (int) v362;
    int v364 = v357 << 8;
    int v365 = v364 & 256;
    int v366 = v361 | v365;
    int v367 = v357 << 7;
    int v368 = v367 & 256;
    int v369 = v363 | v368;
    const uint8_t v370 = v12[82];
    int v371 = (int) v370;
    uint8_t v372 = (uint8_t) v371;
    size_t v373 = (size_t) v366;
    size_t v374 = v373 * 4;
    const uint8_t* v375 = v8 + v374;
    const int8_t* v376 = (const int8_t*) v375;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v377 = __riscv_vle8_v_i8mf4(v376, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v378 = __riscv_vand_vx_u8mf4(v6, v372, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v379 = __riscv_vmsne_vx_u8mf4_b32(v378, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v380 = __riscv_vneg_v_i8mf4(v377, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v381 = __riscv_vmerge_vvm_i8mf4(v377, v380, v379, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v382 = __riscv_vsext_vf4_i32m1(v381, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v383 = __riscv_vfcvt_f_x_v_f32m1(v382, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v384 = __riscv_vfmul_vf_f32m1(v383, v350, 4);
    float* v385 = v15 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v385, v384, 4);
    size_t v386 = (size_t) v369;
    size_t v387 = v386 * 4;
    const uint8_t* v388 = v8 + v387;
    const int8_t* v389 = (const int8_t*) v388;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v390 = __riscv_vle8_v_i8mf4(v389, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v391 = __riscv_vand_vx_u8mf4(v7, v372, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v392 = __riscv_vmsne_vx_u8mf4_b32(v391, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v393 = __riscv_vneg_v_i8mf4(v390, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v394 = __riscv_vmerge_vvm_i8mf4(v390, v393, v392, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v395 = __riscv_vsext_vf4_i32m1(v394, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v396 = __riscv_vfcvt_f_x_v_f32m1(v395, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v397 = __riscv_vfmul_vf_f32m1(v396, v350, 4);
    float* v398 = v15 + 68;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v398, v397, 4);
    const uint8_t v399 = v12[20];
    int v400 = (int) v399;
    const uint8_t v401 = v12[21];
    int v402 = (int) v401;
    int v403 = v357 << 6;
    int v404 = v403 & 256;
    int v405 = v400 | v404;
    int v406 = v357 << 5;
    int v407 = v406 & 256;
    int v408 = v402 | v407;
    const uint8_t v409 = v12[83];
    int v410 = (int) v409;
    uint8_t v411 = (uint8_t) v410;
    size_t v412 = (size_t) v405;
    size_t v413 = v412 * 4;
    const uint8_t* v414 = v8 + v413;
    const int8_t* v415 = (const int8_t*) v414;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v416 = __riscv_vle8_v_i8mf4(v415, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v417 = __riscv_vand_vx_u8mf4(v6, v411, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v418 = __riscv_vmsne_vx_u8mf4_b32(v417, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v419 = __riscv_vneg_v_i8mf4(v416, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v420 = __riscv_vmerge_vvm_i8mf4(v416, v419, v418, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v421 = __riscv_vsext_vf4_i32m1(v420, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v422 = __riscv_vfcvt_f_x_v_f32m1(v421, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v423 = __riscv_vfmul_vf_f32m1(v422, v350, 4);
    float* v424 = v15 + 72;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v424, v423, 4);
    size_t v425 = (size_t) v408;
    size_t v426 = v425 * 4;
    const uint8_t* v427 = v8 + v426;
    const int8_t* v428 = (const int8_t*) v427;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v429 = __riscv_vle8_v_i8mf4(v428, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v430 = __riscv_vand_vx_u8mf4(v7, v411, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v431 = __riscv_vmsne_vx_u8mf4_b32(v430, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v432 = __riscv_vneg_v_i8mf4(v429, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v433 = __riscv_vmerge_vvm_i8mf4(v429, v432, v431, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v434 = __riscv_vsext_vf4_i32m1(v433, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v435 = __riscv_vfcvt_f_x_v_f32m1(v434, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v436 = __riscv_vfmul_vf_f32m1(v435, v350, 4);
    float* v437 = v15 + 76;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v437, v436, 4);
    const uint8_t v438 = v12[22];
    int v439 = (int) v438;
    const uint8_t v440 = v12[23];
    int v441 = (int) v440;
    int v442 = v357 << 4;
    int v443 = v442 & 256;
    int v444 = v439 | v443;
    int v445 = v357 << 3;
    int v446 = v445 & 256;
    int v447 = v441 | v446;
    const uint8_t v448 = v12[84];
    int v449 = (int) v448;
    uint8_t v450 = (uint8_t) v449;
    size_t v451 = (size_t) v444;
    size_t v452 = v451 * 4;
    const uint8_t* v453 = v8 + v452;
    const int8_t* v454 = (const int8_t*) v453;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v455 = __riscv_vle8_v_i8mf4(v454, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v456 = __riscv_vand_vx_u8mf4(v6, v450, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v457 = __riscv_vmsne_vx_u8mf4_b32(v456, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v458 = __riscv_vneg_v_i8mf4(v455, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v459 = __riscv_vmerge_vvm_i8mf4(v455, v458, v457, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v460 = __riscv_vsext_vf4_i32m1(v459, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v461 = __riscv_vfcvt_f_x_v_f32m1(v460, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v462 = __riscv_vfmul_vf_f32m1(v461, v350, 4);
    float* v463 = v15 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v463, v462, 4);
    size_t v464 = (size_t) v447;
    size_t v465 = v464 * 4;
    const uint8_t* v466 = v8 + v465;
    const int8_t* v467 = (const int8_t*) v466;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v468 = __riscv_vle8_v_i8mf4(v467, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v469 = __riscv_vand_vx_u8mf4(v7, v450, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v470 = __riscv_vmsne_vx_u8mf4_b32(v469, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v471 = __riscv_vneg_v_i8mf4(v468, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v472 = __riscv_vmerge_vvm_i8mf4(v468, v471, v470, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v473 = __riscv_vsext_vf4_i32m1(v472, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v474 = __riscv_vfcvt_f_x_v_f32m1(v473, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v475 = __riscv_vfmul_vf_f32m1(v474, v350, 4);
    float* v476 = v15 + 84;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v476, v475, 4);
    const uint8_t v477 = v12[24];
    int v478 = (int) v477;
    const uint8_t v479 = v12[25];
    int v480 = (int) v479;
    int v481 = v357 << 2;
    int v482 = v481 & 256;
    int v483 = v478 | v482;
    int v484 = v357 << 1;
    int v485 = v484 & 256;
    int v486 = v480 | v485;
    const uint8_t v487 = v12[85];
    int v488 = (int) v487;
    uint8_t v489 = (uint8_t) v488;
    size_t v490 = (size_t) v483;
    size_t v491 = v490 * 4;
    const uint8_t* v492 = v8 + v491;
    const int8_t* v493 = (const int8_t*) v492;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v494 = __riscv_vle8_v_i8mf4(v493, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v495 = __riscv_vand_vx_u8mf4(v6, v489, 4);
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
    vfloat32m1_t v501 = __riscv_vfmul_vf_f32m1(v500, v350, 4);
    float* v502 = v15 + 88;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v502, v501, 4);
    size_t v503 = (size_t) v486;
    size_t v504 = v503 * 4;
    const uint8_t* v505 = v8 + v504;
    const int8_t* v506 = (const int8_t*) v505;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v507 = __riscv_vle8_v_i8mf4(v506, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v508 = __riscv_vand_vx_u8mf4(v7, v489, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v509 = __riscv_vmsne_vx_u8mf4_b32(v508, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v510 = __riscv_vneg_v_i8mf4(v507, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v511 = __riscv_vmerge_vvm_i8mf4(v507, v510, v509, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v512 = __riscv_vsext_vf4_i32m1(v511, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v513 = __riscv_vfcvt_f_x_v_f32m1(v512, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v514 = __riscv_vfmul_vf_f32m1(v513, v350, 4);
    float* v515 = v15 + 92;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v515, v514, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v516 = v12[26];
    int v517 = (int) v516;
    const uint8_t v518 = v12[27];
    int v519 = (int) v518;
    int v520 = v359 << 8;
    int v521 = v520 & 256;
    int v522 = v517 | v521;
    int v523 = v359 << 7;
    int v524 = v523 & 256;
    int v525 = v519 | v524;
    const uint8_t v526 = v12[86];
    int v527 = (int) v526;
    uint8_t v528 = (uint8_t) v527;
    size_t v529 = (size_t) v522;
    size_t v530 = v529 * 4;
    const uint8_t* v531 = v8 + v530;
    const int8_t* v532 = (const int8_t*) v531;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v533 = __riscv_vle8_v_i8mf4(v532, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v534 = __riscv_vand_vx_u8mf4(v6, v528, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v535 = __riscv_vmsne_vx_u8mf4_b32(v534, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v536 = __riscv_vneg_v_i8mf4(v533, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v537 = __riscv_vmerge_vvm_i8mf4(v533, v536, v535, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v538 = __riscv_vsext_vf4_i32m1(v537, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v539 = __riscv_vfcvt_f_x_v_f32m1(v538, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v540 = __riscv_vfmul_vf_f32m1(v539, v355, 4);
    float* v541 = v15 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v541, v540, 4);
    size_t v542 = (size_t) v525;
    size_t v543 = v542 * 4;
    const uint8_t* v544 = v8 + v543;
    const int8_t* v545 = (const int8_t*) v544;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v546 = __riscv_vle8_v_i8mf4(v545, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v547 = __riscv_vand_vx_u8mf4(v7, v528, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v548 = __riscv_vmsne_vx_u8mf4_b32(v547, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v549 = __riscv_vneg_v_i8mf4(v546, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v550 = __riscv_vmerge_vvm_i8mf4(v546, v549, v548, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v551 = __riscv_vsext_vf4_i32m1(v550, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v552 = __riscv_vfcvt_f_x_v_f32m1(v551, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v553 = __riscv_vfmul_vf_f32m1(v552, v355, 4);
    float* v554 = v15 + 100;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v554, v553, 4);
    const uint8_t v555 = v12[28];
    int v556 = (int) v555;
    const uint8_t v557 = v12[29];
    int v558 = (int) v557;
    int v559 = v359 << 6;
    int v560 = v559 & 256;
    int v561 = v556 | v560;
    int v562 = v359 << 5;
    int v563 = v562 & 256;
    int v564 = v558 | v563;
    const uint8_t v565 = v12[87];
    int v566 = (int) v565;
    uint8_t v567 = (uint8_t) v566;
    size_t v568 = (size_t) v561;
    size_t v569 = v568 * 4;
    const uint8_t* v570 = v8 + v569;
    const int8_t* v571 = (const int8_t*) v570;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v572 = __riscv_vle8_v_i8mf4(v571, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v573 = __riscv_vand_vx_u8mf4(v6, v567, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v574 = __riscv_vmsne_vx_u8mf4_b32(v573, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v575 = __riscv_vneg_v_i8mf4(v572, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v576 = __riscv_vmerge_vvm_i8mf4(v572, v575, v574, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v577 = __riscv_vsext_vf4_i32m1(v576, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v578 = __riscv_vfcvt_f_x_v_f32m1(v577, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v579 = __riscv_vfmul_vf_f32m1(v578, v355, 4);
    float* v580 = v15 + 104;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v580, v579, 4);
    size_t v581 = (size_t) v564;
    size_t v582 = v581 * 4;
    const uint8_t* v583 = v8 + v582;
    const int8_t* v584 = (const int8_t*) v583;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v585 = __riscv_vle8_v_i8mf4(v584, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v586 = __riscv_vand_vx_u8mf4(v7, v567, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v587 = __riscv_vmsne_vx_u8mf4_b32(v586, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v588 = __riscv_vneg_v_i8mf4(v585, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v589 = __riscv_vmerge_vvm_i8mf4(v585, v588, v587, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v590 = __riscv_vsext_vf4_i32m1(v589, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v591 = __riscv_vfcvt_f_x_v_f32m1(v590, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v592 = __riscv_vfmul_vf_f32m1(v591, v355, 4);
    float* v593 = v15 + 108;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v593, v592, 4);
    const uint8_t v594 = v12[30];
    int v595 = (int) v594;
    const uint8_t v596 = v12[31];
    int v597 = (int) v596;
    int v598 = v359 << 4;
    int v599 = v598 & 256;
    int v600 = v595 | v599;
    int v601 = v359 << 3;
    int v602 = v601 & 256;
    int v603 = v597 | v602;
    const uint8_t v604 = v12[88];
    int v605 = (int) v604;
    uint8_t v606 = (uint8_t) v605;
    size_t v607 = (size_t) v600;
    size_t v608 = v607 * 4;
    const uint8_t* v609 = v8 + v608;
    const int8_t* v610 = (const int8_t*) v609;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v611 = __riscv_vle8_v_i8mf4(v610, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v612 = __riscv_vand_vx_u8mf4(v6, v606, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v613 = __riscv_vmsne_vx_u8mf4_b32(v612, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v614 = __riscv_vneg_v_i8mf4(v611, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v615 = __riscv_vmerge_vvm_i8mf4(v611, v614, v613, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v616 = __riscv_vsext_vf4_i32m1(v615, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v617 = __riscv_vfcvt_f_x_v_f32m1(v616, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v618 = __riscv_vfmul_vf_f32m1(v617, v355, 4);
    float* v619 = v15 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v619, v618, 4);
    size_t v620 = (size_t) v603;
    size_t v621 = v620 * 4;
    const uint8_t* v622 = v8 + v621;
    const int8_t* v623 = (const int8_t*) v622;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v624 = __riscv_vle8_v_i8mf4(v623, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v625 = __riscv_vand_vx_u8mf4(v7, v606, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v626 = __riscv_vmsne_vx_u8mf4_b32(v625, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v627 = __riscv_vneg_v_i8mf4(v624, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v628 = __riscv_vmerge_vvm_i8mf4(v624, v627, v626, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v629 = __riscv_vsext_vf4_i32m1(v628, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v630 = __riscv_vfcvt_f_x_v_f32m1(v629, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v631 = __riscv_vfmul_vf_f32m1(v630, v355, 4);
    float* v632 = v15 + 116;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v632, v631, 4);
    const uint8_t v633 = v12[32];
    int v634 = (int) v633;
    const uint8_t v635 = v12[33];
    int v636 = (int) v635;
    int v637 = v359 << 2;
    int v638 = v637 & 256;
    int v639 = v634 | v638;
    int v640 = v359 << 1;
    int v641 = v640 & 256;
    int v642 = v636 | v641;
    const uint8_t v643 = v12[89];
    int v644 = (int) v643;
    uint8_t v645 = (uint8_t) v644;
    size_t v646 = (size_t) v639;
    size_t v647 = v646 * 4;
    const uint8_t* v648 = v8 + v647;
    const int8_t* v649 = (const int8_t*) v648;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v650 = __riscv_vle8_v_i8mf4(v649, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v651 = __riscv_vand_vx_u8mf4(v6, v645, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v652 = __riscv_vmsne_vx_u8mf4_b32(v651, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v653 = __riscv_vneg_v_i8mf4(v650, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v654 = __riscv_vmerge_vvm_i8mf4(v650, v653, v652, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v655 = __riscv_vsext_vf4_i32m1(v654, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v656 = __riscv_vfcvt_f_x_v_f32m1(v655, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v657 = __riscv_vfmul_vf_f32m1(v656, v355, 4);
    float* v658 = v15 + 120;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v658, v657, 4);
    size_t v659 = (size_t) v642;
    size_t v660 = v659 * 4;
    const uint8_t* v661 = v8 + v660;
    const int8_t* v662 = (const int8_t*) v661;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v663 = __riscv_vle8_v_i8mf4(v662, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v664 = __riscv_vand_vx_u8mf4(v7, v645, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v665 = __riscv_vmsne_vx_u8mf4_b32(v664, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v666 = __riscv_vneg_v_i8mf4(v663, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v667 = __riscv_vmerge_vvm_i8mf4(v663, v666, v665, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v668 = __riscv_vsext_vf4_i32m1(v667, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v669 = __riscv_vfcvt_f_x_v_f32m1(v668, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v670 = __riscv_vfmul_vf_f32m1(v669, v355, 4);
    float* v671 = v15 + 124;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v671, v670, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=group_scale
    const uint8_t v672 = v12[108];
    int v673 = (int) v672;
    int v674 = v673 & 15;
    int v675 = 2 * v674;
    int v676 = 1 + v675;
    float v677 = (float) v676;
    float v678 = v16 * v677;
    int v679 = v673 >> 4;
    int v680 = 2 * v679;
    int v681 = 1 + v680;
    float v682 = (float) v681;
    float v683 = v16 * v682;
    const uint8_t v684 = v12[70];
    int v685 = (int) v684;
    const uint8_t v686 = v12[71];
    int v687 = (int) v686;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v688 = v12[34];
    int v689 = (int) v688;
    const uint8_t v690 = v12[35];
    int v691 = (int) v690;
    int v692 = v685 << 8;
    int v693 = v692 & 256;
    int v694 = v689 | v693;
    int v695 = v685 << 7;
    int v696 = v695 & 256;
    int v697 = v691 | v696;
    const uint8_t v698 = v12[90];
    int v699 = (int) v698;
    uint8_t v700 = (uint8_t) v699;
    size_t v701 = (size_t) v694;
    size_t v702 = v701 * 4;
    const uint8_t* v703 = v8 + v702;
    const int8_t* v704 = (const int8_t*) v703;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v705 = __riscv_vle8_v_i8mf4(v704, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v706 = __riscv_vand_vx_u8mf4(v6, v700, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v707 = __riscv_vmsne_vx_u8mf4_b32(v706, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v708 = __riscv_vneg_v_i8mf4(v705, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v709 = __riscv_vmerge_vvm_i8mf4(v705, v708, v707, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v710 = __riscv_vsext_vf4_i32m1(v709, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v711 = __riscv_vfcvt_f_x_v_f32m1(v710, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v712 = __riscv_vfmul_vf_f32m1(v711, v678, 4);
    float* v713 = v15 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v713, v712, 4);
    size_t v714 = (size_t) v697;
    size_t v715 = v714 * 4;
    const uint8_t* v716 = v8 + v715;
    const int8_t* v717 = (const int8_t*) v716;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v718 = __riscv_vle8_v_i8mf4(v717, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v719 = __riscv_vand_vx_u8mf4(v7, v700, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v720 = __riscv_vmsne_vx_u8mf4_b32(v719, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v721 = __riscv_vneg_v_i8mf4(v718, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v722 = __riscv_vmerge_vvm_i8mf4(v718, v721, v720, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v723 = __riscv_vsext_vf4_i32m1(v722, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v724 = __riscv_vfcvt_f_x_v_f32m1(v723, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v725 = __riscv_vfmul_vf_f32m1(v724, v678, 4);
    float* v726 = v15 + 132;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v726, v725, 4);
    const uint8_t v727 = v12[36];
    int v728 = (int) v727;
    const uint8_t v729 = v12[37];
    int v730 = (int) v729;
    int v731 = v685 << 6;
    int v732 = v731 & 256;
    int v733 = v728 | v732;
    int v734 = v685 << 5;
    int v735 = v734 & 256;
    int v736 = v730 | v735;
    const uint8_t v737 = v12[91];
    int v738 = (int) v737;
    uint8_t v739 = (uint8_t) v738;
    size_t v740 = (size_t) v733;
    size_t v741 = v740 * 4;
    const uint8_t* v742 = v8 + v741;
    const int8_t* v743 = (const int8_t*) v742;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v744 = __riscv_vle8_v_i8mf4(v743, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v745 = __riscv_vand_vx_u8mf4(v6, v739, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v746 = __riscv_vmsne_vx_u8mf4_b32(v745, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v747 = __riscv_vneg_v_i8mf4(v744, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v748 = __riscv_vmerge_vvm_i8mf4(v744, v747, v746, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v749 = __riscv_vsext_vf4_i32m1(v748, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v750 = __riscv_vfcvt_f_x_v_f32m1(v749, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v751 = __riscv_vfmul_vf_f32m1(v750, v678, 4);
    float* v752 = v15 + 136;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v752, v751, 4);
    size_t v753 = (size_t) v736;
    size_t v754 = v753 * 4;
    const uint8_t* v755 = v8 + v754;
    const int8_t* v756 = (const int8_t*) v755;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v757 = __riscv_vle8_v_i8mf4(v756, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v758 = __riscv_vand_vx_u8mf4(v7, v739, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v759 = __riscv_vmsne_vx_u8mf4_b32(v758, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v760 = __riscv_vneg_v_i8mf4(v757, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v761 = __riscv_vmerge_vvm_i8mf4(v757, v760, v759, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v762 = __riscv_vsext_vf4_i32m1(v761, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v763 = __riscv_vfcvt_f_x_v_f32m1(v762, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v764 = __riscv_vfmul_vf_f32m1(v763, v678, 4);
    float* v765 = v15 + 140;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v765, v764, 4);
    const uint8_t v766 = v12[38];
    int v767 = (int) v766;
    const uint8_t v768 = v12[39];
    int v769 = (int) v768;
    int v770 = v685 << 4;
    int v771 = v770 & 256;
    int v772 = v767 | v771;
    int v773 = v685 << 3;
    int v774 = v773 & 256;
    int v775 = v769 | v774;
    const uint8_t v776 = v12[92];
    int v777 = (int) v776;
    uint8_t v778 = (uint8_t) v777;
    size_t v779 = (size_t) v772;
    size_t v780 = v779 * 4;
    const uint8_t* v781 = v8 + v780;
    const int8_t* v782 = (const int8_t*) v781;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v783 = __riscv_vle8_v_i8mf4(v782, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v784 = __riscv_vand_vx_u8mf4(v6, v778, 4);
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
    vfloat32m1_t v790 = __riscv_vfmul_vf_f32m1(v789, v678, 4);
    float* v791 = v15 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v791, v790, 4);
    size_t v792 = (size_t) v775;
    size_t v793 = v792 * 4;
    const uint8_t* v794 = v8 + v793;
    const int8_t* v795 = (const int8_t*) v794;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v796 = __riscv_vle8_v_i8mf4(v795, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v797 = __riscv_vand_vx_u8mf4(v7, v778, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v798 = __riscv_vmsne_vx_u8mf4_b32(v797, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v799 = __riscv_vneg_v_i8mf4(v796, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v800 = __riscv_vmerge_vvm_i8mf4(v796, v799, v798, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v801 = __riscv_vsext_vf4_i32m1(v800, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v802 = __riscv_vfcvt_f_x_v_f32m1(v801, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v803 = __riscv_vfmul_vf_f32m1(v802, v678, 4);
    float* v804 = v15 + 148;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v804, v803, 4);
    const uint8_t v805 = v12[40];
    int v806 = (int) v805;
    const uint8_t v807 = v12[41];
    int v808 = (int) v807;
    int v809 = v685 << 2;
    int v810 = v809 & 256;
    int v811 = v806 | v810;
    int v812 = v685 << 1;
    int v813 = v812 & 256;
    int v814 = v808 | v813;
    const uint8_t v815 = v12[93];
    int v816 = (int) v815;
    uint8_t v817 = (uint8_t) v816;
    size_t v818 = (size_t) v811;
    size_t v819 = v818 * 4;
    const uint8_t* v820 = v8 + v819;
    const int8_t* v821 = (const int8_t*) v820;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v822 = __riscv_vle8_v_i8mf4(v821, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v823 = __riscv_vand_vx_u8mf4(v6, v817, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v824 = __riscv_vmsne_vx_u8mf4_b32(v823, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v825 = __riscv_vneg_v_i8mf4(v822, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v826 = __riscv_vmerge_vvm_i8mf4(v822, v825, v824, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v827 = __riscv_vsext_vf4_i32m1(v826, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v828 = __riscv_vfcvt_f_x_v_f32m1(v827, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v829 = __riscv_vfmul_vf_f32m1(v828, v678, 4);
    float* v830 = v15 + 152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v830, v829, 4);
    size_t v831 = (size_t) v814;
    size_t v832 = v831 * 4;
    const uint8_t* v833 = v8 + v832;
    const int8_t* v834 = (const int8_t*) v833;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v835 = __riscv_vle8_v_i8mf4(v834, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v836 = __riscv_vand_vx_u8mf4(v7, v817, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v837 = __riscv_vmsne_vx_u8mf4_b32(v836, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v838 = __riscv_vneg_v_i8mf4(v835, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v839 = __riscv_vmerge_vvm_i8mf4(v835, v838, v837, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v840 = __riscv_vsext_vf4_i32m1(v839, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v841 = __riscv_vfcvt_f_x_v_f32m1(v840, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v842 = __riscv_vfmul_vf_f32m1(v841, v678, 4);
    float* v843 = v15 + 156;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v843, v842, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v844 = v12[42];
    int v845 = (int) v844;
    const uint8_t v846 = v12[43];
    int v847 = (int) v846;
    int v848 = v687 << 8;
    int v849 = v848 & 256;
    int v850 = v845 | v849;
    int v851 = v687 << 7;
    int v852 = v851 & 256;
    int v853 = v847 | v852;
    const uint8_t v854 = v12[94];
    int v855 = (int) v854;
    uint8_t v856 = (uint8_t) v855;
    size_t v857 = (size_t) v850;
    size_t v858 = v857 * 4;
    const uint8_t* v859 = v8 + v858;
    const int8_t* v860 = (const int8_t*) v859;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v861 = __riscv_vle8_v_i8mf4(v860, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v862 = __riscv_vand_vx_u8mf4(v6, v856, 4);
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
    vfloat32m1_t v868 = __riscv_vfmul_vf_f32m1(v867, v683, 4);
    float* v869 = v15 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v869, v868, 4);
    size_t v870 = (size_t) v853;
    size_t v871 = v870 * 4;
    const uint8_t* v872 = v8 + v871;
    const int8_t* v873 = (const int8_t*) v872;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v874 = __riscv_vle8_v_i8mf4(v873, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v875 = __riscv_vand_vx_u8mf4(v7, v856, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v876 = __riscv_vmsne_vx_u8mf4_b32(v875, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v877 = __riscv_vneg_v_i8mf4(v874, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v878 = __riscv_vmerge_vvm_i8mf4(v874, v877, v876, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v879 = __riscv_vsext_vf4_i32m1(v878, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v880 = __riscv_vfcvt_f_x_v_f32m1(v879, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v881 = __riscv_vfmul_vf_f32m1(v880, v683, 4);
    float* v882 = v15 + 164;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v882, v881, 4);
    const uint8_t v883 = v12[44];
    int v884 = (int) v883;
    const uint8_t v885 = v12[45];
    int v886 = (int) v885;
    int v887 = v687 << 6;
    int v888 = v887 & 256;
    int v889 = v884 | v888;
    int v890 = v687 << 5;
    int v891 = v890 & 256;
    int v892 = v886 | v891;
    const uint8_t v893 = v12[95];
    int v894 = (int) v893;
    uint8_t v895 = (uint8_t) v894;
    size_t v896 = (size_t) v889;
    size_t v897 = v896 * 4;
    const uint8_t* v898 = v8 + v897;
    const int8_t* v899 = (const int8_t*) v898;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v900 = __riscv_vle8_v_i8mf4(v899, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v901 = __riscv_vand_vx_u8mf4(v6, v895, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v902 = __riscv_vmsne_vx_u8mf4_b32(v901, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v903 = __riscv_vneg_v_i8mf4(v900, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v904 = __riscv_vmerge_vvm_i8mf4(v900, v903, v902, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v905 = __riscv_vsext_vf4_i32m1(v904, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v906 = __riscv_vfcvt_f_x_v_f32m1(v905, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v907 = __riscv_vfmul_vf_f32m1(v906, v683, 4);
    float* v908 = v15 + 168;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v908, v907, 4);
    size_t v909 = (size_t) v892;
    size_t v910 = v909 * 4;
    const uint8_t* v911 = v8 + v910;
    const int8_t* v912 = (const int8_t*) v911;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v913 = __riscv_vle8_v_i8mf4(v912, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v914 = __riscv_vand_vx_u8mf4(v7, v895, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v915 = __riscv_vmsne_vx_u8mf4_b32(v914, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v916 = __riscv_vneg_v_i8mf4(v913, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v917 = __riscv_vmerge_vvm_i8mf4(v913, v916, v915, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v918 = __riscv_vsext_vf4_i32m1(v917, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v919 = __riscv_vfcvt_f_x_v_f32m1(v918, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v920 = __riscv_vfmul_vf_f32m1(v919, v683, 4);
    float* v921 = v15 + 172;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v921, v920, 4);
    const uint8_t v922 = v12[46];
    int v923 = (int) v922;
    const uint8_t v924 = v12[47];
    int v925 = (int) v924;
    int v926 = v687 << 4;
    int v927 = v926 & 256;
    int v928 = v923 | v927;
    int v929 = v687 << 3;
    int v930 = v929 & 256;
    int v931 = v925 | v930;
    const uint8_t v932 = v12[96];
    int v933 = (int) v932;
    uint8_t v934 = (uint8_t) v933;
    size_t v935 = (size_t) v928;
    size_t v936 = v935 * 4;
    const uint8_t* v937 = v8 + v936;
    const int8_t* v938 = (const int8_t*) v937;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v939 = __riscv_vle8_v_i8mf4(v938, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v940 = __riscv_vand_vx_u8mf4(v6, v934, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v941 = __riscv_vmsne_vx_u8mf4_b32(v940, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v942 = __riscv_vneg_v_i8mf4(v939, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v943 = __riscv_vmerge_vvm_i8mf4(v939, v942, v941, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v944 = __riscv_vsext_vf4_i32m1(v943, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v945 = __riscv_vfcvt_f_x_v_f32m1(v944, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v946 = __riscv_vfmul_vf_f32m1(v945, v683, 4);
    float* v947 = v15 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v947, v946, 4);
    size_t v948 = (size_t) v931;
    size_t v949 = v948 * 4;
    const uint8_t* v950 = v8 + v949;
    const int8_t* v951 = (const int8_t*) v950;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v952 = __riscv_vle8_v_i8mf4(v951, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v953 = __riscv_vand_vx_u8mf4(v7, v934, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v954 = __riscv_vmsne_vx_u8mf4_b32(v953, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v955 = __riscv_vneg_v_i8mf4(v952, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v956 = __riscv_vmerge_vvm_i8mf4(v952, v955, v954, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v957 = __riscv_vsext_vf4_i32m1(v956, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v958 = __riscv_vfcvt_f_x_v_f32m1(v957, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v959 = __riscv_vfmul_vf_f32m1(v958, v683, 4);
    float* v960 = v15 + 180;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v960, v959, 4);
    const uint8_t v961 = v12[48];
    int v962 = (int) v961;
    const uint8_t v963 = v12[49];
    int v964 = (int) v963;
    int v965 = v687 << 2;
    int v966 = v965 & 256;
    int v967 = v962 | v966;
    int v968 = v687 << 1;
    int v969 = v968 & 256;
    int v970 = v964 | v969;
    const uint8_t v971 = v12[97];
    int v972 = (int) v971;
    uint8_t v973 = (uint8_t) v972;
    size_t v974 = (size_t) v967;
    size_t v975 = v974 * 4;
    const uint8_t* v976 = v8 + v975;
    const int8_t* v977 = (const int8_t*) v976;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v978 = __riscv_vle8_v_i8mf4(v977, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v979 = __riscv_vand_vx_u8mf4(v6, v973, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v980 = __riscv_vmsne_vx_u8mf4_b32(v979, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v981 = __riscv_vneg_v_i8mf4(v978, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v982 = __riscv_vmerge_vvm_i8mf4(v978, v981, v980, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v983 = __riscv_vsext_vf4_i32m1(v982, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v984 = __riscv_vfcvt_f_x_v_f32m1(v983, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v985 = __riscv_vfmul_vf_f32m1(v984, v683, 4);
    float* v986 = v15 + 184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v986, v985, 4);
    size_t v987 = (size_t) v970;
    size_t v988 = v987 * 4;
    const uint8_t* v989 = v8 + v988;
    const int8_t* v990 = (const int8_t*) v989;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v991 = __riscv_vle8_v_i8mf4(v990, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v992 = __riscv_vand_vx_u8mf4(v7, v973, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v993 = __riscv_vmsne_vx_u8mf4_b32(v992, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v994 = __riscv_vneg_v_i8mf4(v991, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v995 = __riscv_vmerge_vvm_i8mf4(v991, v994, v993, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v996 = __riscv_vsext_vf4_i32m1(v995, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v997 = __riscv_vfcvt_f_x_v_f32m1(v996, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v998 = __riscv_vfmul_vf_f32m1(v997, v683, 4);
    float* v999 = v15 + 188;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v999, v998, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=group_scale
    const uint8_t v1000 = v12[109];
    int v1001 = (int) v1000;
    int v1002 = v1001 & 15;
    int v1003 = 2 * v1002;
    int v1004 = 1 + v1003;
    float v1005 = (float) v1004;
    float v1006 = v16 * v1005;
    int v1007 = v1001 >> 4;
    int v1008 = 2 * v1007;
    int v1009 = 1 + v1008;
    float v1010 = (float) v1009;
    float v1011 = v16 * v1010;
    const uint8_t v1012 = v12[72];
    int v1013 = (int) v1012;
    const uint8_t v1014 = v12[73];
    int v1015 = (int) v1014;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v1016 = v12[50];
    int v1017 = (int) v1016;
    const uint8_t v1018 = v12[51];
    int v1019 = (int) v1018;
    int v1020 = v1013 << 8;
    int v1021 = v1020 & 256;
    int v1022 = v1017 | v1021;
    int v1023 = v1013 << 7;
    int v1024 = v1023 & 256;
    int v1025 = v1019 | v1024;
    const uint8_t v1026 = v12[98];
    int v1027 = (int) v1026;
    uint8_t v1028 = (uint8_t) v1027;
    size_t v1029 = (size_t) v1022;
    size_t v1030 = v1029 * 4;
    const uint8_t* v1031 = v8 + v1030;
    const int8_t* v1032 = (const int8_t*) v1031;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1033 = __riscv_vle8_v_i8mf4(v1032, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1034 = __riscv_vand_vx_u8mf4(v6, v1028, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1035 = __riscv_vmsne_vx_u8mf4_b32(v1034, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1036 = __riscv_vneg_v_i8mf4(v1033, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1037 = __riscv_vmerge_vvm_i8mf4(v1033, v1036, v1035, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1038 = __riscv_vsext_vf4_i32m1(v1037, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1039 = __riscv_vfcvt_f_x_v_f32m1(v1038, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1040 = __riscv_vfmul_vf_f32m1(v1039, v1006, 4);
    float* v1041 = v15 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1041, v1040, 4);
    size_t v1042 = (size_t) v1025;
    size_t v1043 = v1042 * 4;
    const uint8_t* v1044 = v8 + v1043;
    const int8_t* v1045 = (const int8_t*) v1044;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1046 = __riscv_vle8_v_i8mf4(v1045, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1047 = __riscv_vand_vx_u8mf4(v7, v1028, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1048 = __riscv_vmsne_vx_u8mf4_b32(v1047, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1049 = __riscv_vneg_v_i8mf4(v1046, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1050 = __riscv_vmerge_vvm_i8mf4(v1046, v1049, v1048, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1051 = __riscv_vsext_vf4_i32m1(v1050, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1052 = __riscv_vfcvt_f_x_v_f32m1(v1051, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1053 = __riscv_vfmul_vf_f32m1(v1052, v1006, 4);
    float* v1054 = v15 + 196;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1054, v1053, 4);
    const uint8_t v1055 = v12[52];
    int v1056 = (int) v1055;
    const uint8_t v1057 = v12[53];
    int v1058 = (int) v1057;
    int v1059 = v1013 << 6;
    int v1060 = v1059 & 256;
    int v1061 = v1056 | v1060;
    int v1062 = v1013 << 5;
    int v1063 = v1062 & 256;
    int v1064 = v1058 | v1063;
    const uint8_t v1065 = v12[99];
    int v1066 = (int) v1065;
    uint8_t v1067 = (uint8_t) v1066;
    size_t v1068 = (size_t) v1061;
    size_t v1069 = v1068 * 4;
    const uint8_t* v1070 = v8 + v1069;
    const int8_t* v1071 = (const int8_t*) v1070;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1072 = __riscv_vle8_v_i8mf4(v1071, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1073 = __riscv_vand_vx_u8mf4(v6, v1067, 4);
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
    vfloat32m1_t v1079 = __riscv_vfmul_vf_f32m1(v1078, v1006, 4);
    float* v1080 = v15 + 200;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1080, v1079, 4);
    size_t v1081 = (size_t) v1064;
    size_t v1082 = v1081 * 4;
    const uint8_t* v1083 = v8 + v1082;
    const int8_t* v1084 = (const int8_t*) v1083;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1085 = __riscv_vle8_v_i8mf4(v1084, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1086 = __riscv_vand_vx_u8mf4(v7, v1067, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1087 = __riscv_vmsne_vx_u8mf4_b32(v1086, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1088 = __riscv_vneg_v_i8mf4(v1085, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1089 = __riscv_vmerge_vvm_i8mf4(v1085, v1088, v1087, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1090 = __riscv_vsext_vf4_i32m1(v1089, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1091 = __riscv_vfcvt_f_x_v_f32m1(v1090, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1092 = __riscv_vfmul_vf_f32m1(v1091, v1006, 4);
    float* v1093 = v15 + 204;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1093, v1092, 4);
    const uint8_t v1094 = v12[54];
    int v1095 = (int) v1094;
    const uint8_t v1096 = v12[55];
    int v1097 = (int) v1096;
    int v1098 = v1013 << 4;
    int v1099 = v1098 & 256;
    int v1100 = v1095 | v1099;
    int v1101 = v1013 << 3;
    int v1102 = v1101 & 256;
    int v1103 = v1097 | v1102;
    const uint8_t v1104 = v12[100];
    int v1105 = (int) v1104;
    uint8_t v1106 = (uint8_t) v1105;
    size_t v1107 = (size_t) v1100;
    size_t v1108 = v1107 * 4;
    const uint8_t* v1109 = v8 + v1108;
    const int8_t* v1110 = (const int8_t*) v1109;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1111 = __riscv_vle8_v_i8mf4(v1110, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1112 = __riscv_vand_vx_u8mf4(v6, v1106, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1113 = __riscv_vmsne_vx_u8mf4_b32(v1112, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1114 = __riscv_vneg_v_i8mf4(v1111, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1115 = __riscv_vmerge_vvm_i8mf4(v1111, v1114, v1113, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1116 = __riscv_vsext_vf4_i32m1(v1115, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1117 = __riscv_vfcvt_f_x_v_f32m1(v1116, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1118 = __riscv_vfmul_vf_f32m1(v1117, v1006, 4);
    float* v1119 = v15 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1119, v1118, 4);
    size_t v1120 = (size_t) v1103;
    size_t v1121 = v1120 * 4;
    const uint8_t* v1122 = v8 + v1121;
    const int8_t* v1123 = (const int8_t*) v1122;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1124 = __riscv_vle8_v_i8mf4(v1123, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1125 = __riscv_vand_vx_u8mf4(v7, v1106, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1126 = __riscv_vmsne_vx_u8mf4_b32(v1125, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1127 = __riscv_vneg_v_i8mf4(v1124, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1128 = __riscv_vmerge_vvm_i8mf4(v1124, v1127, v1126, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1129 = __riscv_vsext_vf4_i32m1(v1128, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1130 = __riscv_vfcvt_f_x_v_f32m1(v1129, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1131 = __riscv_vfmul_vf_f32m1(v1130, v1006, 4);
    float* v1132 = v15 + 212;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1132, v1131, 4);
    const uint8_t v1133 = v12[56];
    int v1134 = (int) v1133;
    const uint8_t v1135 = v12[57];
    int v1136 = (int) v1135;
    int v1137 = v1013 << 2;
    int v1138 = v1137 & 256;
    int v1139 = v1134 | v1138;
    int v1140 = v1013 << 1;
    int v1141 = v1140 & 256;
    int v1142 = v1136 | v1141;
    const uint8_t v1143 = v12[101];
    int v1144 = (int) v1143;
    uint8_t v1145 = (uint8_t) v1144;
    size_t v1146 = (size_t) v1139;
    size_t v1147 = v1146 * 4;
    const uint8_t* v1148 = v8 + v1147;
    const int8_t* v1149 = (const int8_t*) v1148;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1150 = __riscv_vle8_v_i8mf4(v1149, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1151 = __riscv_vand_vx_u8mf4(v6, v1145, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1152 = __riscv_vmsne_vx_u8mf4_b32(v1151, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1153 = __riscv_vneg_v_i8mf4(v1150, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1154 = __riscv_vmerge_vvm_i8mf4(v1150, v1153, v1152, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1155 = __riscv_vsext_vf4_i32m1(v1154, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1156 = __riscv_vfcvt_f_x_v_f32m1(v1155, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1157 = __riscv_vfmul_vf_f32m1(v1156, v1006, 4);
    float* v1158 = v15 + 216;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1158, v1157, 4);
    size_t v1159 = (size_t) v1142;
    size_t v1160 = v1159 * 4;
    const uint8_t* v1161 = v8 + v1160;
    const int8_t* v1162 = (const int8_t*) v1161;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1163 = __riscv_vle8_v_i8mf4(v1162, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1164 = __riscv_vand_vx_u8mf4(v7, v1145, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1165 = __riscv_vmsne_vx_u8mf4_b32(v1164, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1166 = __riscv_vneg_v_i8mf4(v1163, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1167 = __riscv_vmerge_vvm_i8mf4(v1163, v1166, v1165, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1168 = __riscv_vsext_vf4_i32m1(v1167, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1169 = __riscv_vfcvt_f_x_v_f32m1(v1168, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1170 = __riscv_vfmul_vf_f32m1(v1169, v1006, 4);
    float* v1171 = v15 + 220;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1171, v1170, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_pass
    const uint8_t v1172 = v12[58];
    int v1173 = (int) v1172;
    const uint8_t v1174 = v12[59];
    int v1175 = (int) v1174;
    int v1176 = v1015 << 8;
    int v1177 = v1176 & 256;
    int v1178 = v1173 | v1177;
    int v1179 = v1015 << 7;
    int v1180 = v1179 & 256;
    int v1181 = v1175 | v1180;
    const uint8_t v1182 = v12[102];
    int v1183 = (int) v1182;
    uint8_t v1184 = (uint8_t) v1183;
    size_t v1185 = (size_t) v1178;
    size_t v1186 = v1185 * 4;
    const uint8_t* v1187 = v8 + v1186;
    const int8_t* v1188 = (const int8_t*) v1187;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1189 = __riscv_vle8_v_i8mf4(v1188, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1190 = __riscv_vand_vx_u8mf4(v6, v1184, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1191 = __riscv_vmsne_vx_u8mf4_b32(v1190, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1192 = __riscv_vneg_v_i8mf4(v1189, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1193 = __riscv_vmerge_vvm_i8mf4(v1189, v1192, v1191, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1194 = __riscv_vsext_vf4_i32m1(v1193, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1195 = __riscv_vfcvt_f_x_v_f32m1(v1194, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1196 = __riscv_vfmul_vf_f32m1(v1195, v1011, 4);
    float* v1197 = v15 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1197, v1196, 4);
    size_t v1198 = (size_t) v1181;
    size_t v1199 = v1198 * 4;
    const uint8_t* v1200 = v8 + v1199;
    const int8_t* v1201 = (const int8_t*) v1200;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1202 = __riscv_vle8_v_i8mf4(v1201, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1203 = __riscv_vand_vx_u8mf4(v7, v1184, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1204 = __riscv_vmsne_vx_u8mf4_b32(v1203, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1205 = __riscv_vneg_v_i8mf4(v1202, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1206 = __riscv_vmerge_vvm_i8mf4(v1202, v1205, v1204, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1207 = __riscv_vsext_vf4_i32m1(v1206, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1208 = __riscv_vfcvt_f_x_v_f32m1(v1207, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1209 = __riscv_vfmul_vf_f32m1(v1208, v1011, 4);
    float* v1210 = v15 + 228;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1210, v1209, 4);
    const uint8_t v1211 = v12[60];
    int v1212 = (int) v1211;
    const uint8_t v1213 = v12[61];
    int v1214 = (int) v1213;
    int v1215 = v1015 << 6;
    int v1216 = v1215 & 256;
    int v1217 = v1212 | v1216;
    int v1218 = v1015 << 5;
    int v1219 = v1218 & 256;
    int v1220 = v1214 | v1219;
    const uint8_t v1221 = v12[103];
    int v1222 = (int) v1221;
    uint8_t v1223 = (uint8_t) v1222;
    size_t v1224 = (size_t) v1217;
    size_t v1225 = v1224 * 4;
    const uint8_t* v1226 = v8 + v1225;
    const int8_t* v1227 = (const int8_t*) v1226;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1228 = __riscv_vle8_v_i8mf4(v1227, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1229 = __riscv_vand_vx_u8mf4(v6, v1223, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1230 = __riscv_vmsne_vx_u8mf4_b32(v1229, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1231 = __riscv_vneg_v_i8mf4(v1228, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1232 = __riscv_vmerge_vvm_i8mf4(v1228, v1231, v1230, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1233 = __riscv_vsext_vf4_i32m1(v1232, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1234 = __riscv_vfcvt_f_x_v_f32m1(v1233, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1235 = __riscv_vfmul_vf_f32m1(v1234, v1011, 4);
    float* v1236 = v15 + 232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1236, v1235, 4);
    size_t v1237 = (size_t) v1220;
    size_t v1238 = v1237 * 4;
    const uint8_t* v1239 = v8 + v1238;
    const int8_t* v1240 = (const int8_t*) v1239;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1241 = __riscv_vle8_v_i8mf4(v1240, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1242 = __riscv_vand_vx_u8mf4(v7, v1223, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1243 = __riscv_vmsne_vx_u8mf4_b32(v1242, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1244 = __riscv_vneg_v_i8mf4(v1241, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1245 = __riscv_vmerge_vvm_i8mf4(v1241, v1244, v1243, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1246 = __riscv_vsext_vf4_i32m1(v1245, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1247 = __riscv_vfcvt_f_x_v_f32m1(v1246, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1248 = __riscv_vfmul_vf_f32m1(v1247, v1011, 4);
    float* v1249 = v15 + 236;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1249, v1248, 4);
    const uint8_t v1250 = v12[62];
    int v1251 = (int) v1250;
    const uint8_t v1252 = v12[63];
    int v1253 = (int) v1252;
    int v1254 = v1015 << 4;
    int v1255 = v1254 & 256;
    int v1256 = v1251 | v1255;
    int v1257 = v1015 << 3;
    int v1258 = v1257 & 256;
    int v1259 = v1253 | v1258;
    const uint8_t v1260 = v12[104];
    int v1261 = (int) v1260;
    uint8_t v1262 = (uint8_t) v1261;
    size_t v1263 = (size_t) v1256;
    size_t v1264 = v1263 * 4;
    const uint8_t* v1265 = v8 + v1264;
    const int8_t* v1266 = (const int8_t*) v1265;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1267 = __riscv_vle8_v_i8mf4(v1266, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1268 = __riscv_vand_vx_u8mf4(v6, v1262, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1269 = __riscv_vmsne_vx_u8mf4_b32(v1268, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1270 = __riscv_vneg_v_i8mf4(v1267, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1271 = __riscv_vmerge_vvm_i8mf4(v1267, v1270, v1269, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1272 = __riscv_vsext_vf4_i32m1(v1271, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1273 = __riscv_vfcvt_f_x_v_f32m1(v1272, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1274 = __riscv_vfmul_vf_f32m1(v1273, v1011, 4);
    float* v1275 = v15 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1275, v1274, 4);
    size_t v1276 = (size_t) v1259;
    size_t v1277 = v1276 * 4;
    const uint8_t* v1278 = v8 + v1277;
    const int8_t* v1279 = (const int8_t*) v1278;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1280 = __riscv_vle8_v_i8mf4(v1279, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1281 = __riscv_vand_vx_u8mf4(v7, v1262, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1282 = __riscv_vmsne_vx_u8mf4_b32(v1281, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1283 = __riscv_vneg_v_i8mf4(v1280, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1284 = __riscv_vmerge_vvm_i8mf4(v1280, v1283, v1282, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1285 = __riscv_vsext_vf4_i32m1(v1284, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1286 = __riscv_vfcvt_f_x_v_f32m1(v1285, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1287 = __riscv_vfmul_vf_f32m1(v1286, v1011, 4);
    float* v1288 = v15 + 244;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1288, v1287, 4);
    const uint8_t v1289 = v12[64];
    int v1290 = (int) v1289;
    const uint8_t v1291 = v12[65];
    int v1292 = (int) v1291;
    int v1293 = v1015 << 2;
    int v1294 = v1293 & 256;
    int v1295 = v1290 | v1294;
    int v1296 = v1015 << 1;
    int v1297 = v1296 & 256;
    int v1298 = v1292 | v1297;
    const uint8_t v1299 = v12[105];
    int v1300 = (int) v1299;
    uint8_t v1301 = (uint8_t) v1300;
    size_t v1302 = (size_t) v1295;
    size_t v1303 = v1302 * 4;
    const uint8_t* v1304 = v8 + v1303;
    const int8_t* v1305 = (const int8_t*) v1304;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1306 = __riscv_vle8_v_i8mf4(v1305, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1307 = __riscv_vand_vx_u8mf4(v6, v1301, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1308 = __riscv_vmsne_vx_u8mf4_b32(v1307, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1309 = __riscv_vneg_v_i8mf4(v1306, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1310 = __riscv_vmerge_vvm_i8mf4(v1306, v1309, v1308, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1311 = __riscv_vsext_vf4_i32m1(v1310, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1312 = __riscv_vfcvt_f_x_v_f32m1(v1311, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1313 = __riscv_vfmul_vf_f32m1(v1312, v1011, 4);
    float* v1314 = v15 + 248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1314, v1313, 4);
    size_t v1315 = (size_t) v1298;
    size_t v1316 = v1315 * 4;
    const uint8_t* v1317 = v8 + v1316;
    const int8_t* v1318 = (const int8_t*) v1317;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    vint8mf4_t v1319 = __riscv_vle8_v_i8mf4(v1318, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
    vuint8mf4_t v1320 = __riscv_vand_vx_u8mf4(v7, v1301, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8mf4_b32
    vbool32_t v1321 = __riscv_vmsne_vx_u8mf4_b32(v1320, 0, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vneg_v_i8mf4
    vint8mf4_t v1322 = __riscv_vneg_v_i8mf4(v1319, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8mf4
    vint8mf4_t v1323 = __riscv_vmerge_vvm_i8mf4(v1319, v1322, v1321, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m1
    vint32m1_t v1324 = __riscv_vsext_vf4_i32m1(v1323, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
    vfloat32m1_t v1325 = __riscv_vfcvt_f_x_v_f32m1(v1324, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
    vfloat32m1_t v1326 = __riscv_vfmul_vf_f32m1(v1325, v1011, 4);
    float* v1327 = v15 + 252;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
    __riscv_vse32_v_f32m1(v1327, v1326, 4);
  }
  return;
}


