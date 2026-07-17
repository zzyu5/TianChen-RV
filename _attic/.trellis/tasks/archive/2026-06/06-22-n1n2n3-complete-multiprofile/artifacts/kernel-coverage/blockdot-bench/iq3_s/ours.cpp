#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_ggml_vec_dot_iq3_s_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const uint32_t tcrv_iq3s_grid[512] = {0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU, 0x01010301U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU, 0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U, 0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U, 0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU, 0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U, 0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU, 0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU, 0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU, 0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU, 0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU, 0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U, 0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U, 0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U, 0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U, 0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U, 0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U, 0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U, 0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU, 0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U, 0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U, 0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U, 0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U, 0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U, 0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU, 0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U, 0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U, 0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U, 0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU, 0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U, 0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U, 0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU, 0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U, 0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U, 0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U, 0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U, 0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU, 0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U, 0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U, 0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU, 0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU, 0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U, 0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U, 0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U, 0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U, 0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U, 0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U, 0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU, 0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU, 0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U, 0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU, 0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U, 0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U, 0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U, 0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U, 0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U, 0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU, 0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU, 0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U, 0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U, 0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U, 0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U, 0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U, 0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U, 0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U, 0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U, 0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U, 0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U, 0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U, 0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U, 0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U, 0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U, 0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U, 0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U, 0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU, 0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U, 0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U, 0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U, 0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U, 0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U, 0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U, 0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U, 0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU, 0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U, 0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U, 0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U, 0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U, 0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU, 0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U, 0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U, 0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU, 0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU, 0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U, 0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U, 0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U, 0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU, 0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U, 0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U, 0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU, 0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U, 0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U, 0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U, 0x0f0d0703U, 0x0f0f0101U};
  static const uint8_t tcrv_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq3s_kmask, 8);
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_i32_view
  const int32_t* v9 = (const int32_t*) tcrv_iq3s_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v7; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 110;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v16 = (const float*) v14;
    const float v17 = v16[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v18 = v15 * v17;
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t* v21 = v12 + 66;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t* v23 = v12 + 74;
    const uint8_t* v24 = (const uint8_t*) v23;
    const uint8_t* v25 = v12 + 106;
    const uint8_t* v26 = (const uint8_t*) v25;
    const uint8_t* v27 = v14 + 4;
    const int8_t* v28 = (const int8_t*) v27;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v29;
    v29 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v30 = v26[0];
    int v31 = (int) v30;
    int v32 = v31 & 15;
    int v33 = v32 * 2;
    int v34 = v33 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v35 = v22[0];
    int v36 = (int) v35;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v37 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v38 = v24[0];
    int v39 = (int) v38;
    const uint8_t v40 = v20[0];
    int v41 = (int) v40;
    int v42 = v36 << 8;
    int v43 = v42 & 256;
    int v44 = v41 | v43;
    const uint8_t v45 = v20[1];
    int v46 = (int) v45;
    int v47 = v36 << 7;
    int v48 = v47 & 256;
    int v49 = v46 | v48;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v50[2];
    int v51 = v44 << 2;
    uint16_t v52 = (uint16_t) v51;
    v50[0] = v52;
    int v53 = v49 << 2;
    uint16_t v54 = (uint16_t) v53;
    v50[1] = v54;
    uint16_t* v55 = &v50[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v56 = __riscv_vle16_v_u16mf2(v55, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v57 = __riscv_vluxei16_v_i32m1(v9, v56, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v58 = __riscv_vreinterpret_v_i32m1_i8m1(v57);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v59 = __riscv_vle8_v_i8m1(v28, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v60 = __riscv_vmv_v_x_u8m1(v39, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v61 = __riscv_vand_vv_u8m1(v60, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v62 = __riscv_vmsne_vx_u8m1_b8(v61, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v63 = __riscv_vneg_v_i8m1(v58, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v64 = __riscv_vmerge_vvm_i8m1(v58, v63, v62, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v65 = __riscv_vwmul_vv_i16m2(v64, v59, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v66 = __riscv_vwredsum_vs_i16m2_i32m1(v65, v37, 8);
    const int8_t* v67 = v28 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v68 = v24[1];
    int v69 = (int) v68;
    const uint8_t v70 = v20[2];
    int v71 = (int) v70;
    int v72 = v36 << 6;
    int v73 = v72 & 256;
    int v74 = v71 | v73;
    const uint8_t v75 = v20[3];
    int v76 = (int) v75;
    int v77 = v36 << 5;
    int v78 = v77 & 256;
    int v79 = v76 | v78;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v80[2];
    int v81 = v74 << 2;
    uint16_t v82 = (uint16_t) v81;
    v80[0] = v82;
    int v83 = v79 << 2;
    uint16_t v84 = (uint16_t) v83;
    v80[1] = v84;
    uint16_t* v85 = &v80[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v86 = __riscv_vle16_v_u16mf2(v85, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v87 = __riscv_vluxei16_v_i32m1(v9, v86, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v88 = __riscv_vreinterpret_v_i32m1_i8m1(v87);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v89 = __riscv_vle8_v_i8m1(v67, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v90 = __riscv_vmv_v_x_u8m1(v69, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v91 = __riscv_vand_vv_u8m1(v90, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v92 = __riscv_vmsne_vx_u8m1_b8(v91, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v93 = __riscv_vneg_v_i8m1(v88, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v94 = __riscv_vmerge_vvm_i8m1(v88, v93, v92, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v95 = __riscv_vwmul_vv_i16m2(v94, v89, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v96 = __riscv_vwredsum_vs_i16m2_i32m1(v95, v66, 8);
    const int8_t* v97 = v67 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v98 = v24[2];
    int v99 = (int) v98;
    const uint8_t v100 = v20[4];
    int v101 = (int) v100;
    int v102 = v36 << 4;
    int v103 = v102 & 256;
    int v104 = v101 | v103;
    const uint8_t v105 = v20[5];
    int v106 = (int) v105;
    int v107 = v36 << 3;
    int v108 = v107 & 256;
    int v109 = v106 | v108;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v110[2];
    int v111 = v104 << 2;
    uint16_t v112 = (uint16_t) v111;
    v110[0] = v112;
    int v113 = v109 << 2;
    uint16_t v114 = (uint16_t) v113;
    v110[1] = v114;
    uint16_t* v115 = &v110[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v116 = __riscv_vle16_v_u16mf2(v115, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v117 = __riscv_vluxei16_v_i32m1(v9, v116, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v118 = __riscv_vreinterpret_v_i32m1_i8m1(v117);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v119 = __riscv_vle8_v_i8m1(v97, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v120 = __riscv_vmv_v_x_u8m1(v99, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v121 = __riscv_vand_vv_u8m1(v120, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v122 = __riscv_vmsne_vx_u8m1_b8(v121, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v123 = __riscv_vneg_v_i8m1(v118, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v124 = __riscv_vmerge_vvm_i8m1(v118, v123, v122, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v125 = __riscv_vwmul_vv_i16m2(v124, v119, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v126 = __riscv_vwredsum_vs_i16m2_i32m1(v125, v96, 8);
    const int8_t* v127 = v97 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v128 = v24[3];
    int v129 = (int) v128;
    const uint8_t v130 = v20[6];
    int v131 = (int) v130;
    int v132 = v36 << 2;
    int v133 = v132 & 256;
    int v134 = v131 | v133;
    const uint8_t v135 = v20[7];
    int v136 = (int) v135;
    int v137 = v36 << 1;
    int v138 = v137 & 256;
    int v139 = v136 | v138;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v140[2];
    int v141 = v134 << 2;
    uint16_t v142 = (uint16_t) v141;
    v140[0] = v142;
    int v143 = v139 << 2;
    uint16_t v144 = (uint16_t) v143;
    v140[1] = v144;
    uint16_t* v145 = &v140[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v146 = __riscv_vle16_v_u16mf2(v145, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v147 = __riscv_vluxei16_v_i32m1(v9, v146, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v148 = __riscv_vreinterpret_v_i32m1_i8m1(v147);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v149 = __riscv_vle8_v_i8m1(v127, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v150 = __riscv_vmv_v_x_u8m1(v129, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v151 = __riscv_vand_vv_u8m1(v150, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v152 = __riscv_vmsne_vx_u8m1_b8(v151, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v153 = __riscv_vneg_v_i8m1(v148, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v154 = __riscv_vmerge_vvm_i8m1(v148, v153, v152, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v155 = __riscv_vwmul_vv_i16m2(v154, v149, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v156 = __riscv_vwredsum_vs_i16m2_i32m1(v155, v126, 8);
    const int8_t* v157 = v127 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v158 = __riscv_vmv_x_s_i32m1_i32(v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v159 = v29;
    int32_t v160 = (int32_t) v34;
    int32_t v161 = v158 * v160;
    int32_t v162 = v159 + v161;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v162;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v163 = v26[0];
    int v164 = (int) v163;
    int v165 = v164 >> 4;
    int v166 = v165 * 2;
    int v167 = v166 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v168 = v22[1];
    int v169 = (int) v168;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v170 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v171 = v28 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v172 = v24[4];
    int v173 = (int) v172;
    const uint8_t v174 = v20[8];
    int v175 = (int) v174;
    int v176 = v169 << 8;
    int v177 = v176 & 256;
    int v178 = v175 | v177;
    const uint8_t v179 = v20[9];
    int v180 = (int) v179;
    int v181 = v169 << 7;
    int v182 = v181 & 256;
    int v183 = v180 | v182;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v184[2];
    int v185 = v178 << 2;
    uint16_t v186 = (uint16_t) v185;
    v184[0] = v186;
    int v187 = v183 << 2;
    uint16_t v188 = (uint16_t) v187;
    v184[1] = v188;
    uint16_t* v189 = &v184[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v190 = __riscv_vle16_v_u16mf2(v189, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v191 = __riscv_vluxei16_v_i32m1(v9, v190, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v192 = __riscv_vreinterpret_v_i32m1_i8m1(v191);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v193 = __riscv_vle8_v_i8m1(v171, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v194 = __riscv_vmv_v_x_u8m1(v173, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v195 = __riscv_vand_vv_u8m1(v194, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v196 = __riscv_vmsne_vx_u8m1_b8(v195, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v197 = __riscv_vneg_v_i8m1(v192, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v198 = __riscv_vmerge_vvm_i8m1(v192, v197, v196, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v199 = __riscv_vwmul_vv_i16m2(v198, v193, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v200 = __riscv_vwredsum_vs_i16m2_i32m1(v199, v170, 8);
    const int8_t* v201 = v171 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v202 = v24[5];
    int v203 = (int) v202;
    const uint8_t v204 = v20[10];
    int v205 = (int) v204;
    int v206 = v169 << 6;
    int v207 = v206 & 256;
    int v208 = v205 | v207;
    const uint8_t v209 = v20[11];
    int v210 = (int) v209;
    int v211 = v169 << 5;
    int v212 = v211 & 256;
    int v213 = v210 | v212;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v214[2];
    int v215 = v208 << 2;
    uint16_t v216 = (uint16_t) v215;
    v214[0] = v216;
    int v217 = v213 << 2;
    uint16_t v218 = (uint16_t) v217;
    v214[1] = v218;
    uint16_t* v219 = &v214[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v220 = __riscv_vle16_v_u16mf2(v219, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v221 = __riscv_vluxei16_v_i32m1(v9, v220, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v222 = __riscv_vreinterpret_v_i32m1_i8m1(v221);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v223 = __riscv_vle8_v_i8m1(v201, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v224 = __riscv_vmv_v_x_u8m1(v203, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v225 = __riscv_vand_vv_u8m1(v224, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v226 = __riscv_vmsne_vx_u8m1_b8(v225, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v227 = __riscv_vneg_v_i8m1(v222, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v228 = __riscv_vmerge_vvm_i8m1(v222, v227, v226, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v229 = __riscv_vwmul_vv_i16m2(v228, v223, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v230 = __riscv_vwredsum_vs_i16m2_i32m1(v229, v200, 8);
    const int8_t* v231 = v201 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v232 = v24[6];
    int v233 = (int) v232;
    const uint8_t v234 = v20[12];
    int v235 = (int) v234;
    int v236 = v169 << 4;
    int v237 = v236 & 256;
    int v238 = v235 | v237;
    const uint8_t v239 = v20[13];
    int v240 = (int) v239;
    int v241 = v169 << 3;
    int v242 = v241 & 256;
    int v243 = v240 | v242;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v244[2];
    int v245 = v238 << 2;
    uint16_t v246 = (uint16_t) v245;
    v244[0] = v246;
    int v247 = v243 << 2;
    uint16_t v248 = (uint16_t) v247;
    v244[1] = v248;
    uint16_t* v249 = &v244[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v250 = __riscv_vle16_v_u16mf2(v249, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v251 = __riscv_vluxei16_v_i32m1(v9, v250, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v252 = __riscv_vreinterpret_v_i32m1_i8m1(v251);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v253 = __riscv_vle8_v_i8m1(v231, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v254 = __riscv_vmv_v_x_u8m1(v233, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v255 = __riscv_vand_vv_u8m1(v254, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v256 = __riscv_vmsne_vx_u8m1_b8(v255, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v257 = __riscv_vneg_v_i8m1(v252, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v258 = __riscv_vmerge_vvm_i8m1(v252, v257, v256, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v259 = __riscv_vwmul_vv_i16m2(v258, v253, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v260 = __riscv_vwredsum_vs_i16m2_i32m1(v259, v230, 8);
    const int8_t* v261 = v231 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v262 = v24[7];
    int v263 = (int) v262;
    const uint8_t v264 = v20[14];
    int v265 = (int) v264;
    int v266 = v169 << 2;
    int v267 = v266 & 256;
    int v268 = v265 | v267;
    const uint8_t v269 = v20[15];
    int v270 = (int) v269;
    int v271 = v169 << 1;
    int v272 = v271 & 256;
    int v273 = v270 | v272;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v274[2];
    int v275 = v268 << 2;
    uint16_t v276 = (uint16_t) v275;
    v274[0] = v276;
    int v277 = v273 << 2;
    uint16_t v278 = (uint16_t) v277;
    v274[1] = v278;
    uint16_t* v279 = &v274[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v280 = __riscv_vle16_v_u16mf2(v279, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v281 = __riscv_vluxei16_v_i32m1(v9, v280, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v282 = __riscv_vreinterpret_v_i32m1_i8m1(v281);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v283 = __riscv_vle8_v_i8m1(v261, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v284 = __riscv_vmv_v_x_u8m1(v263, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v285 = __riscv_vand_vv_u8m1(v284, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v286 = __riscv_vmsne_vx_u8m1_b8(v285, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v287 = __riscv_vneg_v_i8m1(v282, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v288 = __riscv_vmerge_vvm_i8m1(v282, v287, v286, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v289 = __riscv_vwmul_vv_i16m2(v288, v283, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v290 = __riscv_vwredsum_vs_i16m2_i32m1(v289, v260, 8);
    const int8_t* v291 = v261 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v292 = __riscv_vmv_x_s_i32m1_i32(v290);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v293 = v29;
    int32_t v294 = (int32_t) v167;
    int32_t v295 = v292 * v294;
    int32_t v296 = v293 + v295;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v296;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v297 = v26[1];
    int v298 = (int) v297;
    int v299 = v298 & 15;
    int v300 = v299 * 2;
    int v301 = v300 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v302 = v22[2];
    int v303 = (int) v302;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v304 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v305 = v28 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v306 = v24[8];
    int v307 = (int) v306;
    const uint8_t v308 = v20[16];
    int v309 = (int) v308;
    int v310 = v303 << 8;
    int v311 = v310 & 256;
    int v312 = v309 | v311;
    const uint8_t v313 = v20[17];
    int v314 = (int) v313;
    int v315 = v303 << 7;
    int v316 = v315 & 256;
    int v317 = v314 | v316;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v318[2];
    int v319 = v312 << 2;
    uint16_t v320 = (uint16_t) v319;
    v318[0] = v320;
    int v321 = v317 << 2;
    uint16_t v322 = (uint16_t) v321;
    v318[1] = v322;
    uint16_t* v323 = &v318[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v324 = __riscv_vle16_v_u16mf2(v323, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v325 = __riscv_vluxei16_v_i32m1(v9, v324, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v326 = __riscv_vreinterpret_v_i32m1_i8m1(v325);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v327 = __riscv_vle8_v_i8m1(v305, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v328 = __riscv_vmv_v_x_u8m1(v307, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v329 = __riscv_vand_vv_u8m1(v328, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v330 = __riscv_vmsne_vx_u8m1_b8(v329, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v331 = __riscv_vneg_v_i8m1(v326, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v332 = __riscv_vmerge_vvm_i8m1(v326, v331, v330, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v333 = __riscv_vwmul_vv_i16m2(v332, v327, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v334 = __riscv_vwredsum_vs_i16m2_i32m1(v333, v304, 8);
    const int8_t* v335 = v305 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v336 = v24[9];
    int v337 = (int) v336;
    const uint8_t v338 = v20[18];
    int v339 = (int) v338;
    int v340 = v303 << 6;
    int v341 = v340 & 256;
    int v342 = v339 | v341;
    const uint8_t v343 = v20[19];
    int v344 = (int) v343;
    int v345 = v303 << 5;
    int v346 = v345 & 256;
    int v347 = v344 | v346;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v348[2];
    int v349 = v342 << 2;
    uint16_t v350 = (uint16_t) v349;
    v348[0] = v350;
    int v351 = v347 << 2;
    uint16_t v352 = (uint16_t) v351;
    v348[1] = v352;
    uint16_t* v353 = &v348[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v354 = __riscv_vle16_v_u16mf2(v353, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v355 = __riscv_vluxei16_v_i32m1(v9, v354, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v356 = __riscv_vreinterpret_v_i32m1_i8m1(v355);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v357 = __riscv_vle8_v_i8m1(v335, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v358 = __riscv_vmv_v_x_u8m1(v337, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v359 = __riscv_vand_vv_u8m1(v358, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v360 = __riscv_vmsne_vx_u8m1_b8(v359, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v361 = __riscv_vneg_v_i8m1(v356, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v362 = __riscv_vmerge_vvm_i8m1(v356, v361, v360, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v363 = __riscv_vwmul_vv_i16m2(v362, v357, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v364 = __riscv_vwredsum_vs_i16m2_i32m1(v363, v334, 8);
    const int8_t* v365 = v335 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v366 = v24[10];
    int v367 = (int) v366;
    const uint8_t v368 = v20[20];
    int v369 = (int) v368;
    int v370 = v303 << 4;
    int v371 = v370 & 256;
    int v372 = v369 | v371;
    const uint8_t v373 = v20[21];
    int v374 = (int) v373;
    int v375 = v303 << 3;
    int v376 = v375 & 256;
    int v377 = v374 | v376;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v378[2];
    int v379 = v372 << 2;
    uint16_t v380 = (uint16_t) v379;
    v378[0] = v380;
    int v381 = v377 << 2;
    uint16_t v382 = (uint16_t) v381;
    v378[1] = v382;
    uint16_t* v383 = &v378[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v384 = __riscv_vle16_v_u16mf2(v383, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v385 = __riscv_vluxei16_v_i32m1(v9, v384, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v386 = __riscv_vreinterpret_v_i32m1_i8m1(v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v387 = __riscv_vle8_v_i8m1(v365, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v388 = __riscv_vmv_v_x_u8m1(v367, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v389 = __riscv_vand_vv_u8m1(v388, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v390 = __riscv_vmsne_vx_u8m1_b8(v389, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v391 = __riscv_vneg_v_i8m1(v386, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v392 = __riscv_vmerge_vvm_i8m1(v386, v391, v390, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v393 = __riscv_vwmul_vv_i16m2(v392, v387, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v394 = __riscv_vwredsum_vs_i16m2_i32m1(v393, v364, 8);
    const int8_t* v395 = v365 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v396 = v24[11];
    int v397 = (int) v396;
    const uint8_t v398 = v20[22];
    int v399 = (int) v398;
    int v400 = v303 << 2;
    int v401 = v400 & 256;
    int v402 = v399 | v401;
    const uint8_t v403 = v20[23];
    int v404 = (int) v403;
    int v405 = v303 << 1;
    int v406 = v405 & 256;
    int v407 = v404 | v406;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v408[2];
    int v409 = v402 << 2;
    uint16_t v410 = (uint16_t) v409;
    v408[0] = v410;
    int v411 = v407 << 2;
    uint16_t v412 = (uint16_t) v411;
    v408[1] = v412;
    uint16_t* v413 = &v408[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v414 = __riscv_vle16_v_u16mf2(v413, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v415 = __riscv_vluxei16_v_i32m1(v9, v414, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v416 = __riscv_vreinterpret_v_i32m1_i8m1(v415);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v417 = __riscv_vle8_v_i8m1(v395, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v418 = __riscv_vmv_v_x_u8m1(v397, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v419 = __riscv_vand_vv_u8m1(v418, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v420 = __riscv_vmsne_vx_u8m1_b8(v419, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v421 = __riscv_vneg_v_i8m1(v416, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v422 = __riscv_vmerge_vvm_i8m1(v416, v421, v420, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v423 = __riscv_vwmul_vv_i16m2(v422, v417, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v424 = __riscv_vwredsum_vs_i16m2_i32m1(v423, v394, 8);
    const int8_t* v425 = v395 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v426 = __riscv_vmv_x_s_i32m1_i32(v424);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v427 = v29;
    int32_t v428 = (int32_t) v301;
    int32_t v429 = v426 * v428;
    int32_t v430 = v427 + v429;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v430;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v431 = v26[1];
    int v432 = (int) v431;
    int v433 = v432 >> 4;
    int v434 = v433 * 2;
    int v435 = v434 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v436 = v22[3];
    int v437 = (int) v436;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v438 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v439 = v28 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v440 = v24[12];
    int v441 = (int) v440;
    const uint8_t v442 = v20[24];
    int v443 = (int) v442;
    int v444 = v437 << 8;
    int v445 = v444 & 256;
    int v446 = v443 | v445;
    const uint8_t v447 = v20[25];
    int v448 = (int) v447;
    int v449 = v437 << 7;
    int v450 = v449 & 256;
    int v451 = v448 | v450;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v452[2];
    int v453 = v446 << 2;
    uint16_t v454 = (uint16_t) v453;
    v452[0] = v454;
    int v455 = v451 << 2;
    uint16_t v456 = (uint16_t) v455;
    v452[1] = v456;
    uint16_t* v457 = &v452[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v458 = __riscv_vle16_v_u16mf2(v457, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v459 = __riscv_vluxei16_v_i32m1(v9, v458, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v460 = __riscv_vreinterpret_v_i32m1_i8m1(v459);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v461 = __riscv_vle8_v_i8m1(v439, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v462 = __riscv_vmv_v_x_u8m1(v441, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v463 = __riscv_vand_vv_u8m1(v462, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v464 = __riscv_vmsne_vx_u8m1_b8(v463, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v465 = __riscv_vneg_v_i8m1(v460, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v466 = __riscv_vmerge_vvm_i8m1(v460, v465, v464, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v467 = __riscv_vwmul_vv_i16m2(v466, v461, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v468 = __riscv_vwredsum_vs_i16m2_i32m1(v467, v438, 8);
    const int8_t* v469 = v439 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v470 = v24[13];
    int v471 = (int) v470;
    const uint8_t v472 = v20[26];
    int v473 = (int) v472;
    int v474 = v437 << 6;
    int v475 = v474 & 256;
    int v476 = v473 | v475;
    const uint8_t v477 = v20[27];
    int v478 = (int) v477;
    int v479 = v437 << 5;
    int v480 = v479 & 256;
    int v481 = v478 | v480;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v482[2];
    int v483 = v476 << 2;
    uint16_t v484 = (uint16_t) v483;
    v482[0] = v484;
    int v485 = v481 << 2;
    uint16_t v486 = (uint16_t) v485;
    v482[1] = v486;
    uint16_t* v487 = &v482[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v488 = __riscv_vle16_v_u16mf2(v487, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v489 = __riscv_vluxei16_v_i32m1(v9, v488, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v490 = __riscv_vreinterpret_v_i32m1_i8m1(v489);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v491 = __riscv_vle8_v_i8m1(v469, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v492 = __riscv_vmv_v_x_u8m1(v471, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v493 = __riscv_vand_vv_u8m1(v492, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v494 = __riscv_vmsne_vx_u8m1_b8(v493, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v495 = __riscv_vneg_v_i8m1(v490, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v496 = __riscv_vmerge_vvm_i8m1(v490, v495, v494, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v497 = __riscv_vwmul_vv_i16m2(v496, v491, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v498 = __riscv_vwredsum_vs_i16m2_i32m1(v497, v468, 8);
    const int8_t* v499 = v469 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v500 = v24[14];
    int v501 = (int) v500;
    const uint8_t v502 = v20[28];
    int v503 = (int) v502;
    int v504 = v437 << 4;
    int v505 = v504 & 256;
    int v506 = v503 | v505;
    const uint8_t v507 = v20[29];
    int v508 = (int) v507;
    int v509 = v437 << 3;
    int v510 = v509 & 256;
    int v511 = v508 | v510;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v512[2];
    int v513 = v506 << 2;
    uint16_t v514 = (uint16_t) v513;
    v512[0] = v514;
    int v515 = v511 << 2;
    uint16_t v516 = (uint16_t) v515;
    v512[1] = v516;
    uint16_t* v517 = &v512[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v518 = __riscv_vle16_v_u16mf2(v517, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v519 = __riscv_vluxei16_v_i32m1(v9, v518, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v520 = __riscv_vreinterpret_v_i32m1_i8m1(v519);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v521 = __riscv_vle8_v_i8m1(v499, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v522 = __riscv_vmv_v_x_u8m1(v501, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v523 = __riscv_vand_vv_u8m1(v522, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v524 = __riscv_vmsne_vx_u8m1_b8(v523, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v525 = __riscv_vneg_v_i8m1(v520, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v526 = __riscv_vmerge_vvm_i8m1(v520, v525, v524, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v527 = __riscv_vwmul_vv_i16m2(v526, v521, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v528 = __riscv_vwredsum_vs_i16m2_i32m1(v527, v498, 8);
    const int8_t* v529 = v499 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v530 = v24[15];
    int v531 = (int) v530;
    const uint8_t v532 = v20[30];
    int v533 = (int) v532;
    int v534 = v437 << 2;
    int v535 = v534 & 256;
    int v536 = v533 | v535;
    const uint8_t v537 = v20[31];
    int v538 = (int) v537;
    int v539 = v437 << 1;
    int v540 = v539 & 256;
    int v541 = v538 | v540;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v542[2];
    int v543 = v536 << 2;
    uint16_t v544 = (uint16_t) v543;
    v542[0] = v544;
    int v545 = v541 << 2;
    uint16_t v546 = (uint16_t) v545;
    v542[1] = v546;
    uint16_t* v547 = &v542[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v548 = __riscv_vle16_v_u16mf2(v547, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v549 = __riscv_vluxei16_v_i32m1(v9, v548, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v550 = __riscv_vreinterpret_v_i32m1_i8m1(v549);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v551 = __riscv_vle8_v_i8m1(v529, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v552 = __riscv_vmv_v_x_u8m1(v531, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v553 = __riscv_vand_vv_u8m1(v552, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v554 = __riscv_vmsne_vx_u8m1_b8(v553, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v555 = __riscv_vneg_v_i8m1(v550, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v556 = __riscv_vmerge_vvm_i8m1(v550, v555, v554, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v557 = __riscv_vwmul_vv_i16m2(v556, v551, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v558 = __riscv_vwredsum_vs_i16m2_i32m1(v557, v528, 8);
    const int8_t* v559 = v529 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v560 = __riscv_vmv_x_s_i32m1_i32(v558);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v561 = v29;
    int32_t v562 = (int32_t) v435;
    int32_t v563 = v560 * v562;
    int32_t v564 = v561 + v563;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v564;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v565 = v26[2];
    int v566 = (int) v565;
    int v567 = v566 & 15;
    int v568 = v567 * 2;
    int v569 = v568 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v570 = v22[4];
    int v571 = (int) v570;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v572 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v573 = v28 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v574 = v24[16];
    int v575 = (int) v574;
    const uint8_t v576 = v20[32];
    int v577 = (int) v576;
    int v578 = v571 << 8;
    int v579 = v578 & 256;
    int v580 = v577 | v579;
    const uint8_t v581 = v20[33];
    int v582 = (int) v581;
    int v583 = v571 << 7;
    int v584 = v583 & 256;
    int v585 = v582 | v584;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v586[2];
    int v587 = v580 << 2;
    uint16_t v588 = (uint16_t) v587;
    v586[0] = v588;
    int v589 = v585 << 2;
    uint16_t v590 = (uint16_t) v589;
    v586[1] = v590;
    uint16_t* v591 = &v586[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v592 = __riscv_vle16_v_u16mf2(v591, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v593 = __riscv_vluxei16_v_i32m1(v9, v592, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v594 = __riscv_vreinterpret_v_i32m1_i8m1(v593);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v595 = __riscv_vle8_v_i8m1(v573, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v596 = __riscv_vmv_v_x_u8m1(v575, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v597 = __riscv_vand_vv_u8m1(v596, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v598 = __riscv_vmsne_vx_u8m1_b8(v597, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v599 = __riscv_vneg_v_i8m1(v594, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v600 = __riscv_vmerge_vvm_i8m1(v594, v599, v598, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v601 = __riscv_vwmul_vv_i16m2(v600, v595, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v602 = __riscv_vwredsum_vs_i16m2_i32m1(v601, v572, 8);
    const int8_t* v603 = v573 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v604 = v24[17];
    int v605 = (int) v604;
    const uint8_t v606 = v20[34];
    int v607 = (int) v606;
    int v608 = v571 << 6;
    int v609 = v608 & 256;
    int v610 = v607 | v609;
    const uint8_t v611 = v20[35];
    int v612 = (int) v611;
    int v613 = v571 << 5;
    int v614 = v613 & 256;
    int v615 = v612 | v614;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v616[2];
    int v617 = v610 << 2;
    uint16_t v618 = (uint16_t) v617;
    v616[0] = v618;
    int v619 = v615 << 2;
    uint16_t v620 = (uint16_t) v619;
    v616[1] = v620;
    uint16_t* v621 = &v616[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v622 = __riscv_vle16_v_u16mf2(v621, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v623 = __riscv_vluxei16_v_i32m1(v9, v622, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v624 = __riscv_vreinterpret_v_i32m1_i8m1(v623);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v625 = __riscv_vle8_v_i8m1(v603, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v626 = __riscv_vmv_v_x_u8m1(v605, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v627 = __riscv_vand_vv_u8m1(v626, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v628 = __riscv_vmsne_vx_u8m1_b8(v627, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v629 = __riscv_vneg_v_i8m1(v624, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v630 = __riscv_vmerge_vvm_i8m1(v624, v629, v628, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v631 = __riscv_vwmul_vv_i16m2(v630, v625, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v632 = __riscv_vwredsum_vs_i16m2_i32m1(v631, v602, 8);
    const int8_t* v633 = v603 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v634 = v24[18];
    int v635 = (int) v634;
    const uint8_t v636 = v20[36];
    int v637 = (int) v636;
    int v638 = v571 << 4;
    int v639 = v638 & 256;
    int v640 = v637 | v639;
    const uint8_t v641 = v20[37];
    int v642 = (int) v641;
    int v643 = v571 << 3;
    int v644 = v643 & 256;
    int v645 = v642 | v644;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v646[2];
    int v647 = v640 << 2;
    uint16_t v648 = (uint16_t) v647;
    v646[0] = v648;
    int v649 = v645 << 2;
    uint16_t v650 = (uint16_t) v649;
    v646[1] = v650;
    uint16_t* v651 = &v646[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v652 = __riscv_vle16_v_u16mf2(v651, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v653 = __riscv_vluxei16_v_i32m1(v9, v652, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v654 = __riscv_vreinterpret_v_i32m1_i8m1(v653);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v655 = __riscv_vle8_v_i8m1(v633, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v656 = __riscv_vmv_v_x_u8m1(v635, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v657 = __riscv_vand_vv_u8m1(v656, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v658 = __riscv_vmsne_vx_u8m1_b8(v657, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v659 = __riscv_vneg_v_i8m1(v654, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v660 = __riscv_vmerge_vvm_i8m1(v654, v659, v658, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v661 = __riscv_vwmul_vv_i16m2(v660, v655, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v662 = __riscv_vwredsum_vs_i16m2_i32m1(v661, v632, 8);
    const int8_t* v663 = v633 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v664 = v24[19];
    int v665 = (int) v664;
    const uint8_t v666 = v20[38];
    int v667 = (int) v666;
    int v668 = v571 << 2;
    int v669 = v668 & 256;
    int v670 = v667 | v669;
    const uint8_t v671 = v20[39];
    int v672 = (int) v671;
    int v673 = v571 << 1;
    int v674 = v673 & 256;
    int v675 = v672 | v674;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v676[2];
    int v677 = v670 << 2;
    uint16_t v678 = (uint16_t) v677;
    v676[0] = v678;
    int v679 = v675 << 2;
    uint16_t v680 = (uint16_t) v679;
    v676[1] = v680;
    uint16_t* v681 = &v676[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v682 = __riscv_vle16_v_u16mf2(v681, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v683 = __riscv_vluxei16_v_i32m1(v9, v682, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v684 = __riscv_vreinterpret_v_i32m1_i8m1(v683);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v685 = __riscv_vle8_v_i8m1(v663, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v686 = __riscv_vmv_v_x_u8m1(v665, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v687 = __riscv_vand_vv_u8m1(v686, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v688 = __riscv_vmsne_vx_u8m1_b8(v687, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v689 = __riscv_vneg_v_i8m1(v684, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v690 = __riscv_vmerge_vvm_i8m1(v684, v689, v688, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v691 = __riscv_vwmul_vv_i16m2(v690, v685, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v692 = __riscv_vwredsum_vs_i16m2_i32m1(v691, v662, 8);
    const int8_t* v693 = v663 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v694 = __riscv_vmv_x_s_i32m1_i32(v692);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v695 = v29;
    int32_t v696 = (int32_t) v569;
    int32_t v697 = v694 * v696;
    int32_t v698 = v695 + v697;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v698;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v699 = v26[2];
    int v700 = (int) v699;
    int v701 = v700 >> 4;
    int v702 = v701 * 2;
    int v703 = v702 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v704 = v22[5];
    int v705 = (int) v704;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v706 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v707 = v28 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v708 = v24[20];
    int v709 = (int) v708;
    const uint8_t v710 = v20[40];
    int v711 = (int) v710;
    int v712 = v705 << 8;
    int v713 = v712 & 256;
    int v714 = v711 | v713;
    const uint8_t v715 = v20[41];
    int v716 = (int) v715;
    int v717 = v705 << 7;
    int v718 = v717 & 256;
    int v719 = v716 | v718;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v720[2];
    int v721 = v714 << 2;
    uint16_t v722 = (uint16_t) v721;
    v720[0] = v722;
    int v723 = v719 << 2;
    uint16_t v724 = (uint16_t) v723;
    v720[1] = v724;
    uint16_t* v725 = &v720[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v726 = __riscv_vle16_v_u16mf2(v725, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v727 = __riscv_vluxei16_v_i32m1(v9, v726, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v728 = __riscv_vreinterpret_v_i32m1_i8m1(v727);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v729 = __riscv_vle8_v_i8m1(v707, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v730 = __riscv_vmv_v_x_u8m1(v709, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v731 = __riscv_vand_vv_u8m1(v730, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v732 = __riscv_vmsne_vx_u8m1_b8(v731, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v733 = __riscv_vneg_v_i8m1(v728, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v734 = __riscv_vmerge_vvm_i8m1(v728, v733, v732, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v735 = __riscv_vwmul_vv_i16m2(v734, v729, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v736 = __riscv_vwredsum_vs_i16m2_i32m1(v735, v706, 8);
    const int8_t* v737 = v707 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v738 = v24[21];
    int v739 = (int) v738;
    const uint8_t v740 = v20[42];
    int v741 = (int) v740;
    int v742 = v705 << 6;
    int v743 = v742 & 256;
    int v744 = v741 | v743;
    const uint8_t v745 = v20[43];
    int v746 = (int) v745;
    int v747 = v705 << 5;
    int v748 = v747 & 256;
    int v749 = v746 | v748;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v750[2];
    int v751 = v744 << 2;
    uint16_t v752 = (uint16_t) v751;
    v750[0] = v752;
    int v753 = v749 << 2;
    uint16_t v754 = (uint16_t) v753;
    v750[1] = v754;
    uint16_t* v755 = &v750[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v756 = __riscv_vle16_v_u16mf2(v755, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v757 = __riscv_vluxei16_v_i32m1(v9, v756, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v758 = __riscv_vreinterpret_v_i32m1_i8m1(v757);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v759 = __riscv_vle8_v_i8m1(v737, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v760 = __riscv_vmv_v_x_u8m1(v739, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v761 = __riscv_vand_vv_u8m1(v760, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v762 = __riscv_vmsne_vx_u8m1_b8(v761, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v763 = __riscv_vneg_v_i8m1(v758, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v764 = __riscv_vmerge_vvm_i8m1(v758, v763, v762, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v765 = __riscv_vwmul_vv_i16m2(v764, v759, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v766 = __riscv_vwredsum_vs_i16m2_i32m1(v765, v736, 8);
    const int8_t* v767 = v737 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v768 = v24[22];
    int v769 = (int) v768;
    const uint8_t v770 = v20[44];
    int v771 = (int) v770;
    int v772 = v705 << 4;
    int v773 = v772 & 256;
    int v774 = v771 | v773;
    const uint8_t v775 = v20[45];
    int v776 = (int) v775;
    int v777 = v705 << 3;
    int v778 = v777 & 256;
    int v779 = v776 | v778;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v780[2];
    int v781 = v774 << 2;
    uint16_t v782 = (uint16_t) v781;
    v780[0] = v782;
    int v783 = v779 << 2;
    uint16_t v784 = (uint16_t) v783;
    v780[1] = v784;
    uint16_t* v785 = &v780[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v786 = __riscv_vle16_v_u16mf2(v785, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v787 = __riscv_vluxei16_v_i32m1(v9, v786, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v788 = __riscv_vreinterpret_v_i32m1_i8m1(v787);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v789 = __riscv_vle8_v_i8m1(v767, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v790 = __riscv_vmv_v_x_u8m1(v769, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v791 = __riscv_vand_vv_u8m1(v790, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v792 = __riscv_vmsne_vx_u8m1_b8(v791, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v793 = __riscv_vneg_v_i8m1(v788, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v794 = __riscv_vmerge_vvm_i8m1(v788, v793, v792, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v795 = __riscv_vwmul_vv_i16m2(v794, v789, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v796 = __riscv_vwredsum_vs_i16m2_i32m1(v795, v766, 8);
    const int8_t* v797 = v767 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v798 = v24[23];
    int v799 = (int) v798;
    const uint8_t v800 = v20[46];
    int v801 = (int) v800;
    int v802 = v705 << 2;
    int v803 = v802 & 256;
    int v804 = v801 | v803;
    const uint8_t v805 = v20[47];
    int v806 = (int) v805;
    int v807 = v705 << 1;
    int v808 = v807 & 256;
    int v809 = v806 | v808;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v810[2];
    int v811 = v804 << 2;
    uint16_t v812 = (uint16_t) v811;
    v810[0] = v812;
    int v813 = v809 << 2;
    uint16_t v814 = (uint16_t) v813;
    v810[1] = v814;
    uint16_t* v815 = &v810[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v816 = __riscv_vle16_v_u16mf2(v815, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v817 = __riscv_vluxei16_v_i32m1(v9, v816, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v818 = __riscv_vreinterpret_v_i32m1_i8m1(v817);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v819 = __riscv_vle8_v_i8m1(v797, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v820 = __riscv_vmv_v_x_u8m1(v799, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v821 = __riscv_vand_vv_u8m1(v820, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v822 = __riscv_vmsne_vx_u8m1_b8(v821, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v823 = __riscv_vneg_v_i8m1(v818, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v824 = __riscv_vmerge_vvm_i8m1(v818, v823, v822, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v825 = __riscv_vwmul_vv_i16m2(v824, v819, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v826 = __riscv_vwredsum_vs_i16m2_i32m1(v825, v796, 8);
    const int8_t* v827 = v797 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v828 = __riscv_vmv_x_s_i32m1_i32(v826);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v829 = v29;
    int32_t v830 = (int32_t) v703;
    int32_t v831 = v828 * v830;
    int32_t v832 = v829 + v831;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v832;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v833 = v26[3];
    int v834 = (int) v833;
    int v835 = v834 & 15;
    int v836 = v835 * 2;
    int v837 = v836 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v838 = v22[6];
    int v839 = (int) v838;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v840 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v841 = v28 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v842 = v24[24];
    int v843 = (int) v842;
    const uint8_t v844 = v20[48];
    int v845 = (int) v844;
    int v846 = v839 << 8;
    int v847 = v846 & 256;
    int v848 = v845 | v847;
    const uint8_t v849 = v20[49];
    int v850 = (int) v849;
    int v851 = v839 << 7;
    int v852 = v851 & 256;
    int v853 = v850 | v852;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v854[2];
    int v855 = v848 << 2;
    uint16_t v856 = (uint16_t) v855;
    v854[0] = v856;
    int v857 = v853 << 2;
    uint16_t v858 = (uint16_t) v857;
    v854[1] = v858;
    uint16_t* v859 = &v854[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v860 = __riscv_vle16_v_u16mf2(v859, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v861 = __riscv_vluxei16_v_i32m1(v9, v860, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v862 = __riscv_vreinterpret_v_i32m1_i8m1(v861);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v863 = __riscv_vle8_v_i8m1(v841, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v864 = __riscv_vmv_v_x_u8m1(v843, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v865 = __riscv_vand_vv_u8m1(v864, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v866 = __riscv_vmsne_vx_u8m1_b8(v865, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v867 = __riscv_vneg_v_i8m1(v862, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v868 = __riscv_vmerge_vvm_i8m1(v862, v867, v866, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v869 = __riscv_vwmul_vv_i16m2(v868, v863, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v870 = __riscv_vwredsum_vs_i16m2_i32m1(v869, v840, 8);
    const int8_t* v871 = v841 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v872 = v24[25];
    int v873 = (int) v872;
    const uint8_t v874 = v20[50];
    int v875 = (int) v874;
    int v876 = v839 << 6;
    int v877 = v876 & 256;
    int v878 = v875 | v877;
    const uint8_t v879 = v20[51];
    int v880 = (int) v879;
    int v881 = v839 << 5;
    int v882 = v881 & 256;
    int v883 = v880 | v882;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v884[2];
    int v885 = v878 << 2;
    uint16_t v886 = (uint16_t) v885;
    v884[0] = v886;
    int v887 = v883 << 2;
    uint16_t v888 = (uint16_t) v887;
    v884[1] = v888;
    uint16_t* v889 = &v884[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v890 = __riscv_vle16_v_u16mf2(v889, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v891 = __riscv_vluxei16_v_i32m1(v9, v890, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v892 = __riscv_vreinterpret_v_i32m1_i8m1(v891);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v893 = __riscv_vle8_v_i8m1(v871, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v894 = __riscv_vmv_v_x_u8m1(v873, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v895 = __riscv_vand_vv_u8m1(v894, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v896 = __riscv_vmsne_vx_u8m1_b8(v895, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v897 = __riscv_vneg_v_i8m1(v892, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v898 = __riscv_vmerge_vvm_i8m1(v892, v897, v896, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v899 = __riscv_vwmul_vv_i16m2(v898, v893, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v900 = __riscv_vwredsum_vs_i16m2_i32m1(v899, v870, 8);
    const int8_t* v901 = v871 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v902 = v24[26];
    int v903 = (int) v902;
    const uint8_t v904 = v20[52];
    int v905 = (int) v904;
    int v906 = v839 << 4;
    int v907 = v906 & 256;
    int v908 = v905 | v907;
    const uint8_t v909 = v20[53];
    int v910 = (int) v909;
    int v911 = v839 << 3;
    int v912 = v911 & 256;
    int v913 = v910 | v912;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v914[2];
    int v915 = v908 << 2;
    uint16_t v916 = (uint16_t) v915;
    v914[0] = v916;
    int v917 = v913 << 2;
    uint16_t v918 = (uint16_t) v917;
    v914[1] = v918;
    uint16_t* v919 = &v914[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v920 = __riscv_vle16_v_u16mf2(v919, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v921 = __riscv_vluxei16_v_i32m1(v9, v920, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v922 = __riscv_vreinterpret_v_i32m1_i8m1(v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v923 = __riscv_vle8_v_i8m1(v901, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v924 = __riscv_vmv_v_x_u8m1(v903, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v925 = __riscv_vand_vv_u8m1(v924, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v926 = __riscv_vmsne_vx_u8m1_b8(v925, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v927 = __riscv_vneg_v_i8m1(v922, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v928 = __riscv_vmerge_vvm_i8m1(v922, v927, v926, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v929 = __riscv_vwmul_vv_i16m2(v928, v923, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v930 = __riscv_vwredsum_vs_i16m2_i32m1(v929, v900, 8);
    const int8_t* v931 = v901 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v932 = v24[27];
    int v933 = (int) v932;
    const uint8_t v934 = v20[54];
    int v935 = (int) v934;
    int v936 = v839 << 2;
    int v937 = v936 & 256;
    int v938 = v935 | v937;
    const uint8_t v939 = v20[55];
    int v940 = (int) v939;
    int v941 = v839 << 1;
    int v942 = v941 & 256;
    int v943 = v940 | v942;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v944[2];
    int v945 = v938 << 2;
    uint16_t v946 = (uint16_t) v945;
    v944[0] = v946;
    int v947 = v943 << 2;
    uint16_t v948 = (uint16_t) v947;
    v944[1] = v948;
    uint16_t* v949 = &v944[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v950 = __riscv_vle16_v_u16mf2(v949, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v951 = __riscv_vluxei16_v_i32m1(v9, v950, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v952 = __riscv_vreinterpret_v_i32m1_i8m1(v951);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v953 = __riscv_vle8_v_i8m1(v931, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v954 = __riscv_vmv_v_x_u8m1(v933, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v955 = __riscv_vand_vv_u8m1(v954, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v956 = __riscv_vmsne_vx_u8m1_b8(v955, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v957 = __riscv_vneg_v_i8m1(v952, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v958 = __riscv_vmerge_vvm_i8m1(v952, v957, v956, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v959 = __riscv_vwmul_vv_i16m2(v958, v953, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v960 = __riscv_vwredsum_vs_i16m2_i32m1(v959, v930, 8);
    const int8_t* v961 = v931 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v962 = __riscv_vmv_x_s_i32m1_i32(v960);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v963 = v29;
    int32_t v964 = (int32_t) v837;
    int32_t v965 = v962 * v964;
    int32_t v966 = v963 + v965;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v966;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v967 = v26[3];
    int v968 = (int) v967;
    int v969 = v968 >> 4;
    int v970 = v969 * 2;
    int v971 = v970 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v972 = v22[7];
    int v973 = (int) v972;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v974 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v975 = v28 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v976 = v24[28];
    int v977 = (int) v976;
    const uint8_t v978 = v20[56];
    int v979 = (int) v978;
    int v980 = v973 << 8;
    int v981 = v980 & 256;
    int v982 = v979 | v981;
    const uint8_t v983 = v20[57];
    int v984 = (int) v983;
    int v985 = v973 << 7;
    int v986 = v985 & 256;
    int v987 = v984 | v986;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v988[2];
    int v989 = v982 << 2;
    uint16_t v990 = (uint16_t) v989;
    v988[0] = v990;
    int v991 = v987 << 2;
    uint16_t v992 = (uint16_t) v991;
    v988[1] = v992;
    uint16_t* v993 = &v988[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v994 = __riscv_vle16_v_u16mf2(v993, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v995 = __riscv_vluxei16_v_i32m1(v9, v994, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v996 = __riscv_vreinterpret_v_i32m1_i8m1(v995);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v997 = __riscv_vle8_v_i8m1(v975, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v998 = __riscv_vmv_v_x_u8m1(v977, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v999 = __riscv_vand_vv_u8m1(v998, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1000 = __riscv_vmsne_vx_u8m1_b8(v999, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1001 = __riscv_vneg_v_i8m1(v996, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1002 = __riscv_vmerge_vvm_i8m1(v996, v1001, v1000, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1003 = __riscv_vwmul_vv_i16m2(v1002, v997, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1004 = __riscv_vwredsum_vs_i16m2_i32m1(v1003, v974, 8);
    const int8_t* v1005 = v975 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1006 = v24[29];
    int v1007 = (int) v1006;
    const uint8_t v1008 = v20[58];
    int v1009 = (int) v1008;
    int v1010 = v973 << 6;
    int v1011 = v1010 & 256;
    int v1012 = v1009 | v1011;
    const uint8_t v1013 = v20[59];
    int v1014 = (int) v1013;
    int v1015 = v973 << 5;
    int v1016 = v1015 & 256;
    int v1017 = v1014 | v1016;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1018[2];
    int v1019 = v1012 << 2;
    uint16_t v1020 = (uint16_t) v1019;
    v1018[0] = v1020;
    int v1021 = v1017 << 2;
    uint16_t v1022 = (uint16_t) v1021;
    v1018[1] = v1022;
    uint16_t* v1023 = &v1018[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1024 = __riscv_vle16_v_u16mf2(v1023, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1025 = __riscv_vluxei16_v_i32m1(v9, v1024, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1026 = __riscv_vreinterpret_v_i32m1_i8m1(v1025);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1027 = __riscv_vle8_v_i8m1(v1005, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1028 = __riscv_vmv_v_x_u8m1(v1007, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1029 = __riscv_vand_vv_u8m1(v1028, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1030 = __riscv_vmsne_vx_u8m1_b8(v1029, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1031 = __riscv_vneg_v_i8m1(v1026, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1032 = __riscv_vmerge_vvm_i8m1(v1026, v1031, v1030, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1033 = __riscv_vwmul_vv_i16m2(v1032, v1027, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1034 = __riscv_vwredsum_vs_i16m2_i32m1(v1033, v1004, 8);
    const int8_t* v1035 = v1005 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1036 = v24[30];
    int v1037 = (int) v1036;
    const uint8_t v1038 = v20[60];
    int v1039 = (int) v1038;
    int v1040 = v973 << 4;
    int v1041 = v1040 & 256;
    int v1042 = v1039 | v1041;
    const uint8_t v1043 = v20[61];
    int v1044 = (int) v1043;
    int v1045 = v973 << 3;
    int v1046 = v1045 & 256;
    int v1047 = v1044 | v1046;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1048[2];
    int v1049 = v1042 << 2;
    uint16_t v1050 = (uint16_t) v1049;
    v1048[0] = v1050;
    int v1051 = v1047 << 2;
    uint16_t v1052 = (uint16_t) v1051;
    v1048[1] = v1052;
    uint16_t* v1053 = &v1048[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1054 = __riscv_vle16_v_u16mf2(v1053, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1055 = __riscv_vluxei16_v_i32m1(v9, v1054, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1056 = __riscv_vreinterpret_v_i32m1_i8m1(v1055);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1057 = __riscv_vle8_v_i8m1(v1035, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1058 = __riscv_vmv_v_x_u8m1(v1037, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1059 = __riscv_vand_vv_u8m1(v1058, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1060 = __riscv_vmsne_vx_u8m1_b8(v1059, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1061 = __riscv_vneg_v_i8m1(v1056, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1062 = __riscv_vmerge_vvm_i8m1(v1056, v1061, v1060, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1063 = __riscv_vwmul_vv_i16m2(v1062, v1057, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1064 = __riscv_vwredsum_vs_i16m2_i32m1(v1063, v1034, 8);
    const int8_t* v1065 = v1035 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1066 = v24[31];
    int v1067 = (int) v1066;
    const uint8_t v1068 = v20[62];
    int v1069 = (int) v1068;
    int v1070 = v973 << 2;
    int v1071 = v1070 & 256;
    int v1072 = v1069 | v1071;
    const uint8_t v1073 = v20[63];
    int v1074 = (int) v1073;
    int v1075 = v973 << 1;
    int v1076 = v1075 & 256;
    int v1077 = v1074 | v1076;
    // tcrv_emitc.local_variable=idxoff source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v1078[2];
    int v1079 = v1072 << 2;
    uint16_t v1080 = (uint16_t) v1079;
    v1078[0] = v1080;
    int v1081 = v1077 << 2;
    uint16_t v1082 = (uint16_t) v1081;
    v1078[1] = v1082;
    uint16_t* v1083 = &v1078[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf2
    vuint16mf2_t v1084 = __riscv_vle16_v_u16mf2(v1083, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i32m1
    vint32m1_t v1085 = __riscv_vluxei16_v_i32m1(v9, v1084, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i32m1_i8m1
    vint8m1_t v1086 = __riscv_vreinterpret_v_i32m1_i8m1(v1085);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1087 = __riscv_vle8_v_i8m1(v1065, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1088 = __riscv_vmv_v_x_u8m1(v1067, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1089 = __riscv_vand_vv_u8m1(v1088, v8, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1090 = __riscv_vmsne_vx_u8m1_b8(v1089, 0, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1091 = __riscv_vneg_v_i8m1(v1086, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1092 = __riscv_vmerge_vvm_i8m1(v1086, v1091, v1090, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1093 = __riscv_vwmul_vv_i16m2(v1092, v1087, 8);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1094 = __riscv_vwredsum_vs_i16m2_i32m1(v1093, v1064, 8);
    const int8_t* v1095 = v1065 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1096 = __riscv_vmv_x_s_i32m1_i32(v1094);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1097 = v29;
    int32_t v1098 = (int32_t) v971;
    int32_t v1099 = v1096 * v1098;
    int32_t v1100 = v1097 + v1099;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v1100;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v1101 = v29;
    float v1102 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v1102 + v18 * (float) v1101;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v1103 = v6;
  v2[0] = v1103;
  return;
}


