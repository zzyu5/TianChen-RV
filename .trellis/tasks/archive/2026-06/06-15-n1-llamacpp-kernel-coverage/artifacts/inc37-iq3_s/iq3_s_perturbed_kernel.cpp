#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_PERT_kernel_ggml_vec_dot_iq3_s_q8_K_PERT(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const uint32_t tcrv_iq3s_grid[512] = {0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU, 0x01010302U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU, 0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U, 0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U, 0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU, 0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U, 0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU, 0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU, 0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU, 0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU, 0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU, 0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U, 0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U, 0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U, 0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U, 0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U, 0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U, 0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U, 0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU, 0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U, 0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U, 0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U, 0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U, 0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U, 0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU, 0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U, 0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U, 0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U, 0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU, 0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U, 0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U, 0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU, 0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U, 0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U, 0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U, 0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U, 0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU, 0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U, 0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U, 0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU, 0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU, 0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U, 0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U, 0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U, 0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U, 0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U, 0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U, 0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU, 0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU, 0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U, 0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU, 0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U, 0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U, 0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U, 0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U, 0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U, 0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU, 0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU, 0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U, 0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U, 0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U, 0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U, 0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U, 0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U, 0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U, 0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U, 0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U, 0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U, 0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U, 0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U, 0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U, 0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U, 0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U, 0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U, 0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU, 0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U, 0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U, 0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U, 0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U, 0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U, 0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U, 0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U, 0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU, 0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U, 0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U, 0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U, 0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U, 0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU, 0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U, 0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U, 0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU, 0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU, 0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U, 0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U, 0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U, 0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU, 0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U, 0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U, 0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU, 0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U, 0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U, 0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U, 0x0f0d0703U, 0x0f0f0101U};
  static const uint8_t tcrv_iq3s_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq3s_kmask, 4);
  const uint8_t* v9 = tcrv_iq3s_kmask + 4;
  vuint8m1_t v10 = __riscv_vle8_v_u8m1(v9, 4);
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_byte_view
  const int8_t* v11 = (const int8_t*) tcrv_iq3s_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v12 = 0; v12 < v7; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v13 = v12 * 110;
    const uint8_t* v14 = v3 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v15 = v12 * 292;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v17 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v18 = (const float*) v16;
    const float v19 = v18[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v20 = v17 * v19;
    const uint8_t* v21 = v14 + 2;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t* v23 = v14 + 66;
    const uint8_t* v24 = (const uint8_t*) v23;
    const uint8_t* v25 = v14 + 74;
    const uint8_t* v26 = (const uint8_t*) v25;
    const uint8_t* v27 = v14 + 106;
    const uint8_t* v28 = (const uint8_t*) v27;
    const uint8_t* v29 = v16 + 4;
    const int8_t* v30 = (const int8_t*) v29;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v31;
    v31 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v32 = v28[0];
    int v33 = (int) v32;
    int v34 = v33 & 15;
    int v35 = v34 * 2;
    int v36 = v35 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v37 = v24[0];
    int v38 = (int) v37;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v39 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v40 = v26[0];
    int v41 = (int) v40;
    const uint8_t v42 = v22[0];
    int v43 = (int) v42;
    int v44 = v38 << 8;
    int v45 = v44 & 256;
    int v46 = v43 | v45;
    const uint8_t v47 = v22[1];
    int v48 = (int) v47;
    int v49 = v38 << 7;
    int v50 = v49 & 256;
    int v51 = v48 | v50;
    const int8_t* v52 = v30 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v53 = __riscv_vsetvl_e8m1(4);
    size_t v54 = v46 * 4;
    const int8_t* v55 = v11 + v54;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v56 = __riscv_vle8_v_i8m1(v55, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v57 = __riscv_vle8_v_i8m1(v30, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v58 = __riscv_vmv_v_x_u8m1(v41, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v59 = __riscv_vand_vv_u8m1(v58, v8, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v60 = __riscv_vmsne_vx_u8m1_b8(v59, 0, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v61 = __riscv_vneg_v_i8m1(v56, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v62 = __riscv_vmerge_vvm_i8m1(v56, v61, v60, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v63 = __riscv_vwmul_vv_i16m2(v62, v57, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v64 = __riscv_vwredsum_vs_i16m2_i32m1(v63, v39, v53);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v65 = __riscv_vsetvl_e8m1(4);
    size_t v66 = v51 * 4;
    const int8_t* v67 = v11 + v66;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v68 = __riscv_vle8_v_i8m1(v67, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v69 = __riscv_vle8_v_i8m1(v52, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v70 = __riscv_vmv_v_x_u8m1(v41, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v71 = __riscv_vand_vv_u8m1(v70, v10, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v72 = __riscv_vmsne_vx_u8m1_b8(v71, 0, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v73 = __riscv_vneg_v_i8m1(v68, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v74 = __riscv_vmerge_vvm_i8m1(v68, v73, v72, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v75 = __riscv_vwmul_vv_i16m2(v74, v69, v65);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v76 = __riscv_vwredsum_vs_i16m2_i32m1(v75, v64, v65);
    const int8_t* v77 = v30 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v78 = v26[1];
    int v79 = (int) v78;
    const uint8_t v80 = v22[2];
    int v81 = (int) v80;
    int v82 = v38 << 6;
    int v83 = v82 & 256;
    int v84 = v81 | v83;
    const uint8_t v85 = v22[3];
    int v86 = (int) v85;
    int v87 = v38 << 5;
    int v88 = v87 & 256;
    int v89 = v86 | v88;
    const int8_t* v90 = v77 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v91 = __riscv_vsetvl_e8m1(4);
    size_t v92 = v84 * 4;
    const int8_t* v93 = v11 + v92;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v94 = __riscv_vle8_v_i8m1(v93, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v95 = __riscv_vle8_v_i8m1(v77, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v96 = __riscv_vmv_v_x_u8m1(v79, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v97 = __riscv_vand_vv_u8m1(v96, v8, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v98 = __riscv_vmsne_vx_u8m1_b8(v97, 0, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v99 = __riscv_vneg_v_i8m1(v94, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v100 = __riscv_vmerge_vvm_i8m1(v94, v99, v98, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v101 = __riscv_vwmul_vv_i16m2(v100, v95, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v102 = __riscv_vwredsum_vs_i16m2_i32m1(v101, v76, v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v103 = __riscv_vsetvl_e8m1(4);
    size_t v104 = v89 * 4;
    const int8_t* v105 = v11 + v104;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v106 = __riscv_vle8_v_i8m1(v105, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v107 = __riscv_vle8_v_i8m1(v90, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v108 = __riscv_vmv_v_x_u8m1(v79, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v109 = __riscv_vand_vv_u8m1(v108, v10, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v110 = __riscv_vmsne_vx_u8m1_b8(v109, 0, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v111 = __riscv_vneg_v_i8m1(v106, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v112 = __riscv_vmerge_vvm_i8m1(v106, v111, v110, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v113 = __riscv_vwmul_vv_i16m2(v112, v107, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v114 = __riscv_vwredsum_vs_i16m2_i32m1(v113, v102, v103);
    const int8_t* v115 = v77 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v116 = v26[2];
    int v117 = (int) v116;
    const uint8_t v118 = v22[4];
    int v119 = (int) v118;
    int v120 = v38 << 4;
    int v121 = v120 & 256;
    int v122 = v119 | v121;
    const uint8_t v123 = v22[5];
    int v124 = (int) v123;
    int v125 = v38 << 3;
    int v126 = v125 & 256;
    int v127 = v124 | v126;
    const int8_t* v128 = v115 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v129 = __riscv_vsetvl_e8m1(4);
    size_t v130 = v122 * 4;
    const int8_t* v131 = v11 + v130;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v132 = __riscv_vle8_v_i8m1(v131, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v133 = __riscv_vle8_v_i8m1(v115, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v134 = __riscv_vmv_v_x_u8m1(v117, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v135 = __riscv_vand_vv_u8m1(v134, v8, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v136 = __riscv_vmsne_vx_u8m1_b8(v135, 0, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v137 = __riscv_vneg_v_i8m1(v132, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v138 = __riscv_vmerge_vvm_i8m1(v132, v137, v136, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v139 = __riscv_vwmul_vv_i16m2(v138, v133, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v140 = __riscv_vwredsum_vs_i16m2_i32m1(v139, v114, v129);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v141 = __riscv_vsetvl_e8m1(4);
    size_t v142 = v127 * 4;
    const int8_t* v143 = v11 + v142;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v144 = __riscv_vle8_v_i8m1(v143, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v145 = __riscv_vle8_v_i8m1(v128, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v146 = __riscv_vmv_v_x_u8m1(v117, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v147 = __riscv_vand_vv_u8m1(v146, v10, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v148 = __riscv_vmsne_vx_u8m1_b8(v147, 0, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v149 = __riscv_vneg_v_i8m1(v144, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v150 = __riscv_vmerge_vvm_i8m1(v144, v149, v148, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v151 = __riscv_vwmul_vv_i16m2(v150, v145, v141);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v152 = __riscv_vwredsum_vs_i16m2_i32m1(v151, v140, v141);
    const int8_t* v153 = v115 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v154 = v26[3];
    int v155 = (int) v154;
    const uint8_t v156 = v22[6];
    int v157 = (int) v156;
    int v158 = v38 << 2;
    int v159 = v158 & 256;
    int v160 = v157 | v159;
    const uint8_t v161 = v22[7];
    int v162 = (int) v161;
    int v163 = v38 << 1;
    int v164 = v163 & 256;
    int v165 = v162 | v164;
    const int8_t* v166 = v153 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v167 = __riscv_vsetvl_e8m1(4);
    size_t v168 = v160 * 4;
    const int8_t* v169 = v11 + v168;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v170 = __riscv_vle8_v_i8m1(v169, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v171 = __riscv_vle8_v_i8m1(v153, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v172 = __riscv_vmv_v_x_u8m1(v155, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v173 = __riscv_vand_vv_u8m1(v172, v8, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v174 = __riscv_vmsne_vx_u8m1_b8(v173, 0, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v175 = __riscv_vneg_v_i8m1(v170, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v176 = __riscv_vmerge_vvm_i8m1(v170, v175, v174, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v177 = __riscv_vwmul_vv_i16m2(v176, v171, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v178 = __riscv_vwredsum_vs_i16m2_i32m1(v177, v152, v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v179 = __riscv_vsetvl_e8m1(4);
    size_t v180 = v165 * 4;
    const int8_t* v181 = v11 + v180;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v182 = __riscv_vle8_v_i8m1(v181, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v183 = __riscv_vle8_v_i8m1(v166, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v184 = __riscv_vmv_v_x_u8m1(v155, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v185 = __riscv_vand_vv_u8m1(v184, v10, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v186 = __riscv_vmsne_vx_u8m1_b8(v185, 0, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v187 = __riscv_vneg_v_i8m1(v182, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v188 = __riscv_vmerge_vvm_i8m1(v182, v187, v186, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v189 = __riscv_vwmul_vv_i16m2(v188, v183, v179);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v190 = __riscv_vwredsum_vs_i16m2_i32m1(v189, v178, v179);
    const int8_t* v191 = v153 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v192 = __riscv_vmv_x_s_i32m1_i32(v190);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v193 = v31;
    int32_t v194 = (int32_t) v36;
    int32_t v195 = v192 * v194;
    int32_t v196 = v193 + v195;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v196;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v197 = v28[0];
    int v198 = (int) v197;
    int v199 = v198 >> 4;
    int v200 = v199 * 2;
    int v201 = v200 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v202 = v24[1];
    int v203 = (int) v202;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v204 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v205 = v30 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v206 = v26[4];
    int v207 = (int) v206;
    const uint8_t v208 = v22[8];
    int v209 = (int) v208;
    int v210 = v203 << 8;
    int v211 = v210 & 256;
    int v212 = v209 | v211;
    const uint8_t v213 = v22[9];
    int v214 = (int) v213;
    int v215 = v203 << 7;
    int v216 = v215 & 256;
    int v217 = v214 | v216;
    const int8_t* v218 = v205 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v219 = __riscv_vsetvl_e8m1(4);
    size_t v220 = v212 * 4;
    const int8_t* v221 = v11 + v220;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v222 = __riscv_vle8_v_i8m1(v221, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v223 = __riscv_vle8_v_i8m1(v205, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v224 = __riscv_vmv_v_x_u8m1(v207, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v225 = __riscv_vand_vv_u8m1(v224, v8, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v226 = __riscv_vmsne_vx_u8m1_b8(v225, 0, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v227 = __riscv_vneg_v_i8m1(v222, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v228 = __riscv_vmerge_vvm_i8m1(v222, v227, v226, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v229 = __riscv_vwmul_vv_i16m2(v228, v223, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v230 = __riscv_vwredsum_vs_i16m2_i32m1(v229, v204, v219);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v231 = __riscv_vsetvl_e8m1(4);
    size_t v232 = v217 * 4;
    const int8_t* v233 = v11 + v232;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v234 = __riscv_vle8_v_i8m1(v233, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v235 = __riscv_vle8_v_i8m1(v218, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v236 = __riscv_vmv_v_x_u8m1(v207, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v237 = __riscv_vand_vv_u8m1(v236, v10, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v238 = __riscv_vmsne_vx_u8m1_b8(v237, 0, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v239 = __riscv_vneg_v_i8m1(v234, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v240 = __riscv_vmerge_vvm_i8m1(v234, v239, v238, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v241 = __riscv_vwmul_vv_i16m2(v240, v235, v231);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v242 = __riscv_vwredsum_vs_i16m2_i32m1(v241, v230, v231);
    const int8_t* v243 = v205 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v244 = v26[5];
    int v245 = (int) v244;
    const uint8_t v246 = v22[10];
    int v247 = (int) v246;
    int v248 = v203 << 6;
    int v249 = v248 & 256;
    int v250 = v247 | v249;
    const uint8_t v251 = v22[11];
    int v252 = (int) v251;
    int v253 = v203 << 5;
    int v254 = v253 & 256;
    int v255 = v252 | v254;
    const int8_t* v256 = v243 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v257 = __riscv_vsetvl_e8m1(4);
    size_t v258 = v250 * 4;
    const int8_t* v259 = v11 + v258;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v260 = __riscv_vle8_v_i8m1(v259, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v261 = __riscv_vle8_v_i8m1(v243, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v262 = __riscv_vmv_v_x_u8m1(v245, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v263 = __riscv_vand_vv_u8m1(v262, v8, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v264 = __riscv_vmsne_vx_u8m1_b8(v263, 0, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v265 = __riscv_vneg_v_i8m1(v260, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v266 = __riscv_vmerge_vvm_i8m1(v260, v265, v264, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v267 = __riscv_vwmul_vv_i16m2(v266, v261, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v268 = __riscv_vwredsum_vs_i16m2_i32m1(v267, v242, v257);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v269 = __riscv_vsetvl_e8m1(4);
    size_t v270 = v255 * 4;
    const int8_t* v271 = v11 + v270;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v272 = __riscv_vle8_v_i8m1(v271, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v273 = __riscv_vle8_v_i8m1(v256, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v274 = __riscv_vmv_v_x_u8m1(v245, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v275 = __riscv_vand_vv_u8m1(v274, v10, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v276 = __riscv_vmsne_vx_u8m1_b8(v275, 0, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v277 = __riscv_vneg_v_i8m1(v272, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v278 = __riscv_vmerge_vvm_i8m1(v272, v277, v276, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v279 = __riscv_vwmul_vv_i16m2(v278, v273, v269);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v280 = __riscv_vwredsum_vs_i16m2_i32m1(v279, v268, v269);
    const int8_t* v281 = v243 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v282 = v26[6];
    int v283 = (int) v282;
    const uint8_t v284 = v22[12];
    int v285 = (int) v284;
    int v286 = v203 << 4;
    int v287 = v286 & 256;
    int v288 = v285 | v287;
    const uint8_t v289 = v22[13];
    int v290 = (int) v289;
    int v291 = v203 << 3;
    int v292 = v291 & 256;
    int v293 = v290 | v292;
    const int8_t* v294 = v281 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v295 = __riscv_vsetvl_e8m1(4);
    size_t v296 = v288 * 4;
    const int8_t* v297 = v11 + v296;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v298 = __riscv_vle8_v_i8m1(v297, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v299 = __riscv_vle8_v_i8m1(v281, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v300 = __riscv_vmv_v_x_u8m1(v283, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v301 = __riscv_vand_vv_u8m1(v300, v8, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v302 = __riscv_vmsne_vx_u8m1_b8(v301, 0, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v303 = __riscv_vneg_v_i8m1(v298, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v304 = __riscv_vmerge_vvm_i8m1(v298, v303, v302, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v305 = __riscv_vwmul_vv_i16m2(v304, v299, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v306 = __riscv_vwredsum_vs_i16m2_i32m1(v305, v280, v295);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v307 = __riscv_vsetvl_e8m1(4);
    size_t v308 = v293 * 4;
    const int8_t* v309 = v11 + v308;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v310 = __riscv_vle8_v_i8m1(v309, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v311 = __riscv_vle8_v_i8m1(v294, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v312 = __riscv_vmv_v_x_u8m1(v283, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v313 = __riscv_vand_vv_u8m1(v312, v10, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v314 = __riscv_vmsne_vx_u8m1_b8(v313, 0, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v315 = __riscv_vneg_v_i8m1(v310, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v316 = __riscv_vmerge_vvm_i8m1(v310, v315, v314, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v317 = __riscv_vwmul_vv_i16m2(v316, v311, v307);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v318 = __riscv_vwredsum_vs_i16m2_i32m1(v317, v306, v307);
    const int8_t* v319 = v281 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v320 = v26[7];
    int v321 = (int) v320;
    const uint8_t v322 = v22[14];
    int v323 = (int) v322;
    int v324 = v203 << 2;
    int v325 = v324 & 256;
    int v326 = v323 | v325;
    const uint8_t v327 = v22[15];
    int v328 = (int) v327;
    int v329 = v203 << 1;
    int v330 = v329 & 256;
    int v331 = v328 | v330;
    const int8_t* v332 = v319 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v333 = __riscv_vsetvl_e8m1(4);
    size_t v334 = v326 * 4;
    const int8_t* v335 = v11 + v334;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v336 = __riscv_vle8_v_i8m1(v335, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v337 = __riscv_vle8_v_i8m1(v319, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v338 = __riscv_vmv_v_x_u8m1(v321, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v339 = __riscv_vand_vv_u8m1(v338, v8, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v340 = __riscv_vmsne_vx_u8m1_b8(v339, 0, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v341 = __riscv_vneg_v_i8m1(v336, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v342 = __riscv_vmerge_vvm_i8m1(v336, v341, v340, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v343 = __riscv_vwmul_vv_i16m2(v342, v337, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v344 = __riscv_vwredsum_vs_i16m2_i32m1(v343, v318, v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v345 = __riscv_vsetvl_e8m1(4);
    size_t v346 = v331 * 4;
    const int8_t* v347 = v11 + v346;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v348 = __riscv_vle8_v_i8m1(v347, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v349 = __riscv_vle8_v_i8m1(v332, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v350 = __riscv_vmv_v_x_u8m1(v321, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v351 = __riscv_vand_vv_u8m1(v350, v10, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v352 = __riscv_vmsne_vx_u8m1_b8(v351, 0, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v353 = __riscv_vneg_v_i8m1(v348, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v354 = __riscv_vmerge_vvm_i8m1(v348, v353, v352, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v355 = __riscv_vwmul_vv_i16m2(v354, v349, v345);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v356 = __riscv_vwredsum_vs_i16m2_i32m1(v355, v344, v345);
    const int8_t* v357 = v319 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v358 = __riscv_vmv_x_s_i32m1_i32(v356);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v359 = v31;
    int32_t v360 = (int32_t) v201;
    int32_t v361 = v358 * v360;
    int32_t v362 = v359 + v361;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v362;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v363 = v28[1];
    int v364 = (int) v363;
    int v365 = v364 & 15;
    int v366 = v365 * 2;
    int v367 = v366 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v368 = v24[2];
    int v369 = (int) v368;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v370 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v371 = v30 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v372 = v26[8];
    int v373 = (int) v372;
    const uint8_t v374 = v22[16];
    int v375 = (int) v374;
    int v376 = v369 << 8;
    int v377 = v376 & 256;
    int v378 = v375 | v377;
    const uint8_t v379 = v22[17];
    int v380 = (int) v379;
    int v381 = v369 << 7;
    int v382 = v381 & 256;
    int v383 = v380 | v382;
    const int8_t* v384 = v371 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v385 = __riscv_vsetvl_e8m1(4);
    size_t v386 = v378 * 4;
    const int8_t* v387 = v11 + v386;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v388 = __riscv_vle8_v_i8m1(v387, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v389 = __riscv_vle8_v_i8m1(v371, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v390 = __riscv_vmv_v_x_u8m1(v373, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v391 = __riscv_vand_vv_u8m1(v390, v8, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v392 = __riscv_vmsne_vx_u8m1_b8(v391, 0, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v393 = __riscv_vneg_v_i8m1(v388, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v394 = __riscv_vmerge_vvm_i8m1(v388, v393, v392, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v395 = __riscv_vwmul_vv_i16m2(v394, v389, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v396 = __riscv_vwredsum_vs_i16m2_i32m1(v395, v370, v385);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v397 = __riscv_vsetvl_e8m1(4);
    size_t v398 = v383 * 4;
    const int8_t* v399 = v11 + v398;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v400 = __riscv_vle8_v_i8m1(v399, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v401 = __riscv_vle8_v_i8m1(v384, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v402 = __riscv_vmv_v_x_u8m1(v373, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v403 = __riscv_vand_vv_u8m1(v402, v10, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v404 = __riscv_vmsne_vx_u8m1_b8(v403, 0, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v405 = __riscv_vneg_v_i8m1(v400, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v406 = __riscv_vmerge_vvm_i8m1(v400, v405, v404, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v407 = __riscv_vwmul_vv_i16m2(v406, v401, v397);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v408 = __riscv_vwredsum_vs_i16m2_i32m1(v407, v396, v397);
    const int8_t* v409 = v371 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v410 = v26[9];
    int v411 = (int) v410;
    const uint8_t v412 = v22[18];
    int v413 = (int) v412;
    int v414 = v369 << 6;
    int v415 = v414 & 256;
    int v416 = v413 | v415;
    const uint8_t v417 = v22[19];
    int v418 = (int) v417;
    int v419 = v369 << 5;
    int v420 = v419 & 256;
    int v421 = v418 | v420;
    const int8_t* v422 = v409 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v423 = __riscv_vsetvl_e8m1(4);
    size_t v424 = v416 * 4;
    const int8_t* v425 = v11 + v424;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v426 = __riscv_vle8_v_i8m1(v425, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v427 = __riscv_vle8_v_i8m1(v409, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v428 = __riscv_vmv_v_x_u8m1(v411, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v429 = __riscv_vand_vv_u8m1(v428, v8, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v430 = __riscv_vmsne_vx_u8m1_b8(v429, 0, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v431 = __riscv_vneg_v_i8m1(v426, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v432 = __riscv_vmerge_vvm_i8m1(v426, v431, v430, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v433 = __riscv_vwmul_vv_i16m2(v432, v427, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v434 = __riscv_vwredsum_vs_i16m2_i32m1(v433, v408, v423);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v435 = __riscv_vsetvl_e8m1(4);
    size_t v436 = v421 * 4;
    const int8_t* v437 = v11 + v436;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v438 = __riscv_vle8_v_i8m1(v437, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v439 = __riscv_vle8_v_i8m1(v422, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v440 = __riscv_vmv_v_x_u8m1(v411, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v441 = __riscv_vand_vv_u8m1(v440, v10, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v442 = __riscv_vmsne_vx_u8m1_b8(v441, 0, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v443 = __riscv_vneg_v_i8m1(v438, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v444 = __riscv_vmerge_vvm_i8m1(v438, v443, v442, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v445 = __riscv_vwmul_vv_i16m2(v444, v439, v435);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v446 = __riscv_vwredsum_vs_i16m2_i32m1(v445, v434, v435);
    const int8_t* v447 = v409 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v448 = v26[10];
    int v449 = (int) v448;
    const uint8_t v450 = v22[20];
    int v451 = (int) v450;
    int v452 = v369 << 4;
    int v453 = v452 & 256;
    int v454 = v451 | v453;
    const uint8_t v455 = v22[21];
    int v456 = (int) v455;
    int v457 = v369 << 3;
    int v458 = v457 & 256;
    int v459 = v456 | v458;
    const int8_t* v460 = v447 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v461 = __riscv_vsetvl_e8m1(4);
    size_t v462 = v454 * 4;
    const int8_t* v463 = v11 + v462;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v464 = __riscv_vle8_v_i8m1(v463, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v465 = __riscv_vle8_v_i8m1(v447, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v466 = __riscv_vmv_v_x_u8m1(v449, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v467 = __riscv_vand_vv_u8m1(v466, v8, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v468 = __riscv_vmsne_vx_u8m1_b8(v467, 0, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v469 = __riscv_vneg_v_i8m1(v464, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v470 = __riscv_vmerge_vvm_i8m1(v464, v469, v468, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v471 = __riscv_vwmul_vv_i16m2(v470, v465, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v472 = __riscv_vwredsum_vs_i16m2_i32m1(v471, v446, v461);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v473 = __riscv_vsetvl_e8m1(4);
    size_t v474 = v459 * 4;
    const int8_t* v475 = v11 + v474;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v476 = __riscv_vle8_v_i8m1(v475, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v477 = __riscv_vle8_v_i8m1(v460, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v478 = __riscv_vmv_v_x_u8m1(v449, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v479 = __riscv_vand_vv_u8m1(v478, v10, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v480 = __riscv_vmsne_vx_u8m1_b8(v479, 0, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v481 = __riscv_vneg_v_i8m1(v476, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v482 = __riscv_vmerge_vvm_i8m1(v476, v481, v480, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v483 = __riscv_vwmul_vv_i16m2(v482, v477, v473);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v484 = __riscv_vwredsum_vs_i16m2_i32m1(v483, v472, v473);
    const int8_t* v485 = v447 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v486 = v26[11];
    int v487 = (int) v486;
    const uint8_t v488 = v22[22];
    int v489 = (int) v488;
    int v490 = v369 << 2;
    int v491 = v490 & 256;
    int v492 = v489 | v491;
    const uint8_t v493 = v22[23];
    int v494 = (int) v493;
    int v495 = v369 << 1;
    int v496 = v495 & 256;
    int v497 = v494 | v496;
    const int8_t* v498 = v485 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v499 = __riscv_vsetvl_e8m1(4);
    size_t v500 = v492 * 4;
    const int8_t* v501 = v11 + v500;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v502 = __riscv_vle8_v_i8m1(v501, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v503 = __riscv_vle8_v_i8m1(v485, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v504 = __riscv_vmv_v_x_u8m1(v487, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v505 = __riscv_vand_vv_u8m1(v504, v8, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v506 = __riscv_vmsne_vx_u8m1_b8(v505, 0, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v507 = __riscv_vneg_v_i8m1(v502, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v508 = __riscv_vmerge_vvm_i8m1(v502, v507, v506, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v509 = __riscv_vwmul_vv_i16m2(v508, v503, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v510 = __riscv_vwredsum_vs_i16m2_i32m1(v509, v484, v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v511 = __riscv_vsetvl_e8m1(4);
    size_t v512 = v497 * 4;
    const int8_t* v513 = v11 + v512;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v514 = __riscv_vle8_v_i8m1(v513, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v515 = __riscv_vle8_v_i8m1(v498, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v516 = __riscv_vmv_v_x_u8m1(v487, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v517 = __riscv_vand_vv_u8m1(v516, v10, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v518 = __riscv_vmsne_vx_u8m1_b8(v517, 0, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v519 = __riscv_vneg_v_i8m1(v514, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v520 = __riscv_vmerge_vvm_i8m1(v514, v519, v518, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v521 = __riscv_vwmul_vv_i16m2(v520, v515, v511);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v522 = __riscv_vwredsum_vs_i16m2_i32m1(v521, v510, v511);
    const int8_t* v523 = v485 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v524 = __riscv_vmv_x_s_i32m1_i32(v522);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v525 = v31;
    int32_t v526 = (int32_t) v367;
    int32_t v527 = v524 * v526;
    int32_t v528 = v525 + v527;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v528;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v529 = v28[1];
    int v530 = (int) v529;
    int v531 = v530 >> 4;
    int v532 = v531 * 2;
    int v533 = v532 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v534 = v24[3];
    int v535 = (int) v534;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v536 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v537 = v30 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v538 = v26[12];
    int v539 = (int) v538;
    const uint8_t v540 = v22[24];
    int v541 = (int) v540;
    int v542 = v535 << 8;
    int v543 = v542 & 256;
    int v544 = v541 | v543;
    const uint8_t v545 = v22[25];
    int v546 = (int) v545;
    int v547 = v535 << 7;
    int v548 = v547 & 256;
    int v549 = v546 | v548;
    const int8_t* v550 = v537 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v551 = __riscv_vsetvl_e8m1(4);
    size_t v552 = v544 * 4;
    const int8_t* v553 = v11 + v552;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v554 = __riscv_vle8_v_i8m1(v553, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v555 = __riscv_vle8_v_i8m1(v537, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v556 = __riscv_vmv_v_x_u8m1(v539, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v557 = __riscv_vand_vv_u8m1(v556, v8, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v558 = __riscv_vmsne_vx_u8m1_b8(v557, 0, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v559 = __riscv_vneg_v_i8m1(v554, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v560 = __riscv_vmerge_vvm_i8m1(v554, v559, v558, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v561 = __riscv_vwmul_vv_i16m2(v560, v555, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v562 = __riscv_vwredsum_vs_i16m2_i32m1(v561, v536, v551);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v563 = __riscv_vsetvl_e8m1(4);
    size_t v564 = v549 * 4;
    const int8_t* v565 = v11 + v564;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v566 = __riscv_vle8_v_i8m1(v565, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v567 = __riscv_vle8_v_i8m1(v550, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v568 = __riscv_vmv_v_x_u8m1(v539, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v569 = __riscv_vand_vv_u8m1(v568, v10, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v570 = __riscv_vmsne_vx_u8m1_b8(v569, 0, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v571 = __riscv_vneg_v_i8m1(v566, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v572 = __riscv_vmerge_vvm_i8m1(v566, v571, v570, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v573 = __riscv_vwmul_vv_i16m2(v572, v567, v563);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v574 = __riscv_vwredsum_vs_i16m2_i32m1(v573, v562, v563);
    const int8_t* v575 = v537 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v576 = v26[13];
    int v577 = (int) v576;
    const uint8_t v578 = v22[26];
    int v579 = (int) v578;
    int v580 = v535 << 6;
    int v581 = v580 & 256;
    int v582 = v579 | v581;
    const uint8_t v583 = v22[27];
    int v584 = (int) v583;
    int v585 = v535 << 5;
    int v586 = v585 & 256;
    int v587 = v584 | v586;
    const int8_t* v588 = v575 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v589 = __riscv_vsetvl_e8m1(4);
    size_t v590 = v582 * 4;
    const int8_t* v591 = v11 + v590;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v592 = __riscv_vle8_v_i8m1(v591, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v593 = __riscv_vle8_v_i8m1(v575, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v594 = __riscv_vmv_v_x_u8m1(v577, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v595 = __riscv_vand_vv_u8m1(v594, v8, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v596 = __riscv_vmsne_vx_u8m1_b8(v595, 0, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v597 = __riscv_vneg_v_i8m1(v592, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v598 = __riscv_vmerge_vvm_i8m1(v592, v597, v596, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v599 = __riscv_vwmul_vv_i16m2(v598, v593, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v600 = __riscv_vwredsum_vs_i16m2_i32m1(v599, v574, v589);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v601 = __riscv_vsetvl_e8m1(4);
    size_t v602 = v587 * 4;
    const int8_t* v603 = v11 + v602;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v604 = __riscv_vle8_v_i8m1(v603, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v605 = __riscv_vle8_v_i8m1(v588, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v606 = __riscv_vmv_v_x_u8m1(v577, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v607 = __riscv_vand_vv_u8m1(v606, v10, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v608 = __riscv_vmsne_vx_u8m1_b8(v607, 0, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v609 = __riscv_vneg_v_i8m1(v604, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v610 = __riscv_vmerge_vvm_i8m1(v604, v609, v608, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v611 = __riscv_vwmul_vv_i16m2(v610, v605, v601);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v612 = __riscv_vwredsum_vs_i16m2_i32m1(v611, v600, v601);
    const int8_t* v613 = v575 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v614 = v26[14];
    int v615 = (int) v614;
    const uint8_t v616 = v22[28];
    int v617 = (int) v616;
    int v618 = v535 << 4;
    int v619 = v618 & 256;
    int v620 = v617 | v619;
    const uint8_t v621 = v22[29];
    int v622 = (int) v621;
    int v623 = v535 << 3;
    int v624 = v623 & 256;
    int v625 = v622 | v624;
    const int8_t* v626 = v613 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v627 = __riscv_vsetvl_e8m1(4);
    size_t v628 = v620 * 4;
    const int8_t* v629 = v11 + v628;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v630 = __riscv_vle8_v_i8m1(v629, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v631 = __riscv_vle8_v_i8m1(v613, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v632 = __riscv_vmv_v_x_u8m1(v615, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v633 = __riscv_vand_vv_u8m1(v632, v8, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v634 = __riscv_vmsne_vx_u8m1_b8(v633, 0, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v635 = __riscv_vneg_v_i8m1(v630, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v636 = __riscv_vmerge_vvm_i8m1(v630, v635, v634, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v637 = __riscv_vwmul_vv_i16m2(v636, v631, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v638 = __riscv_vwredsum_vs_i16m2_i32m1(v637, v612, v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v639 = __riscv_vsetvl_e8m1(4);
    size_t v640 = v625 * 4;
    const int8_t* v641 = v11 + v640;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v642 = __riscv_vle8_v_i8m1(v641, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v643 = __riscv_vle8_v_i8m1(v626, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v644 = __riscv_vmv_v_x_u8m1(v615, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v645 = __riscv_vand_vv_u8m1(v644, v10, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v646 = __riscv_vmsne_vx_u8m1_b8(v645, 0, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v647 = __riscv_vneg_v_i8m1(v642, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v648 = __riscv_vmerge_vvm_i8m1(v642, v647, v646, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v649 = __riscv_vwmul_vv_i16m2(v648, v643, v639);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v650 = __riscv_vwredsum_vs_i16m2_i32m1(v649, v638, v639);
    const int8_t* v651 = v613 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v652 = v26[15];
    int v653 = (int) v652;
    const uint8_t v654 = v22[30];
    int v655 = (int) v654;
    int v656 = v535 << 2;
    int v657 = v656 & 256;
    int v658 = v655 | v657;
    const uint8_t v659 = v22[31];
    int v660 = (int) v659;
    int v661 = v535 << 1;
    int v662 = v661 & 256;
    int v663 = v660 | v662;
    const int8_t* v664 = v651 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v665 = __riscv_vsetvl_e8m1(4);
    size_t v666 = v658 * 4;
    const int8_t* v667 = v11 + v666;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v668 = __riscv_vle8_v_i8m1(v667, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v669 = __riscv_vle8_v_i8m1(v651, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v670 = __riscv_vmv_v_x_u8m1(v653, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v671 = __riscv_vand_vv_u8m1(v670, v8, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v672 = __riscv_vmsne_vx_u8m1_b8(v671, 0, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v673 = __riscv_vneg_v_i8m1(v668, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v674 = __riscv_vmerge_vvm_i8m1(v668, v673, v672, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v675 = __riscv_vwmul_vv_i16m2(v674, v669, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v676 = __riscv_vwredsum_vs_i16m2_i32m1(v675, v650, v665);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v677 = __riscv_vsetvl_e8m1(4);
    size_t v678 = v663 * 4;
    const int8_t* v679 = v11 + v678;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v680 = __riscv_vle8_v_i8m1(v679, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v681 = __riscv_vle8_v_i8m1(v664, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v682 = __riscv_vmv_v_x_u8m1(v653, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v683 = __riscv_vand_vv_u8m1(v682, v10, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v684 = __riscv_vmsne_vx_u8m1_b8(v683, 0, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v685 = __riscv_vneg_v_i8m1(v680, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v686 = __riscv_vmerge_vvm_i8m1(v680, v685, v684, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v687 = __riscv_vwmul_vv_i16m2(v686, v681, v677);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v688 = __riscv_vwredsum_vs_i16m2_i32m1(v687, v676, v677);
    const int8_t* v689 = v651 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v690 = __riscv_vmv_x_s_i32m1_i32(v688);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v691 = v31;
    int32_t v692 = (int32_t) v533;
    int32_t v693 = v690 * v692;
    int32_t v694 = v691 + v693;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v694;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v695 = v28[2];
    int v696 = (int) v695;
    int v697 = v696 & 15;
    int v698 = v697 * 2;
    int v699 = v698 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v700 = v24[4];
    int v701 = (int) v700;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v702 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v703 = v30 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v704 = v26[16];
    int v705 = (int) v704;
    const uint8_t v706 = v22[32];
    int v707 = (int) v706;
    int v708 = v701 << 8;
    int v709 = v708 & 256;
    int v710 = v707 | v709;
    const uint8_t v711 = v22[33];
    int v712 = (int) v711;
    int v713 = v701 << 7;
    int v714 = v713 & 256;
    int v715 = v712 | v714;
    const int8_t* v716 = v703 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v717 = __riscv_vsetvl_e8m1(4);
    size_t v718 = v710 * 4;
    const int8_t* v719 = v11 + v718;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v720 = __riscv_vle8_v_i8m1(v719, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v721 = __riscv_vle8_v_i8m1(v703, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v722 = __riscv_vmv_v_x_u8m1(v705, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v723 = __riscv_vand_vv_u8m1(v722, v8, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v724 = __riscv_vmsne_vx_u8m1_b8(v723, 0, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v725 = __riscv_vneg_v_i8m1(v720, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v726 = __riscv_vmerge_vvm_i8m1(v720, v725, v724, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v727 = __riscv_vwmul_vv_i16m2(v726, v721, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v728 = __riscv_vwredsum_vs_i16m2_i32m1(v727, v702, v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v729 = __riscv_vsetvl_e8m1(4);
    size_t v730 = v715 * 4;
    const int8_t* v731 = v11 + v730;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v732 = __riscv_vle8_v_i8m1(v731, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v733 = __riscv_vle8_v_i8m1(v716, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v734 = __riscv_vmv_v_x_u8m1(v705, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v735 = __riscv_vand_vv_u8m1(v734, v10, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v736 = __riscv_vmsne_vx_u8m1_b8(v735, 0, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v737 = __riscv_vneg_v_i8m1(v732, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v738 = __riscv_vmerge_vvm_i8m1(v732, v737, v736, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v739 = __riscv_vwmul_vv_i16m2(v738, v733, v729);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v740 = __riscv_vwredsum_vs_i16m2_i32m1(v739, v728, v729);
    const int8_t* v741 = v703 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v742 = v26[17];
    int v743 = (int) v742;
    const uint8_t v744 = v22[34];
    int v745 = (int) v744;
    int v746 = v701 << 6;
    int v747 = v746 & 256;
    int v748 = v745 | v747;
    const uint8_t v749 = v22[35];
    int v750 = (int) v749;
    int v751 = v701 << 5;
    int v752 = v751 & 256;
    int v753 = v750 | v752;
    const int8_t* v754 = v741 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v755 = __riscv_vsetvl_e8m1(4);
    size_t v756 = v748 * 4;
    const int8_t* v757 = v11 + v756;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v758 = __riscv_vle8_v_i8m1(v757, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v759 = __riscv_vle8_v_i8m1(v741, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v760 = __riscv_vmv_v_x_u8m1(v743, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v761 = __riscv_vand_vv_u8m1(v760, v8, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v762 = __riscv_vmsne_vx_u8m1_b8(v761, 0, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v763 = __riscv_vneg_v_i8m1(v758, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v764 = __riscv_vmerge_vvm_i8m1(v758, v763, v762, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v765 = __riscv_vwmul_vv_i16m2(v764, v759, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v766 = __riscv_vwredsum_vs_i16m2_i32m1(v765, v740, v755);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v767 = __riscv_vsetvl_e8m1(4);
    size_t v768 = v753 * 4;
    const int8_t* v769 = v11 + v768;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v770 = __riscv_vle8_v_i8m1(v769, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v771 = __riscv_vle8_v_i8m1(v754, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v772 = __riscv_vmv_v_x_u8m1(v743, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v773 = __riscv_vand_vv_u8m1(v772, v10, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v774 = __riscv_vmsne_vx_u8m1_b8(v773, 0, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v775 = __riscv_vneg_v_i8m1(v770, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v776 = __riscv_vmerge_vvm_i8m1(v770, v775, v774, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v777 = __riscv_vwmul_vv_i16m2(v776, v771, v767);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v778 = __riscv_vwredsum_vs_i16m2_i32m1(v777, v766, v767);
    const int8_t* v779 = v741 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v780 = v26[18];
    int v781 = (int) v780;
    const uint8_t v782 = v22[36];
    int v783 = (int) v782;
    int v784 = v701 << 4;
    int v785 = v784 & 256;
    int v786 = v783 | v785;
    const uint8_t v787 = v22[37];
    int v788 = (int) v787;
    int v789 = v701 << 3;
    int v790 = v789 & 256;
    int v791 = v788 | v790;
    const int8_t* v792 = v779 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v793 = __riscv_vsetvl_e8m1(4);
    size_t v794 = v786 * 4;
    const int8_t* v795 = v11 + v794;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v796 = __riscv_vle8_v_i8m1(v795, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v797 = __riscv_vle8_v_i8m1(v779, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v798 = __riscv_vmv_v_x_u8m1(v781, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v799 = __riscv_vand_vv_u8m1(v798, v8, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v800 = __riscv_vmsne_vx_u8m1_b8(v799, 0, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v801 = __riscv_vneg_v_i8m1(v796, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v802 = __riscv_vmerge_vvm_i8m1(v796, v801, v800, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v803 = __riscv_vwmul_vv_i16m2(v802, v797, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v804 = __riscv_vwredsum_vs_i16m2_i32m1(v803, v778, v793);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v805 = __riscv_vsetvl_e8m1(4);
    size_t v806 = v791 * 4;
    const int8_t* v807 = v11 + v806;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v808 = __riscv_vle8_v_i8m1(v807, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v809 = __riscv_vle8_v_i8m1(v792, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v810 = __riscv_vmv_v_x_u8m1(v781, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v811 = __riscv_vand_vv_u8m1(v810, v10, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v812 = __riscv_vmsne_vx_u8m1_b8(v811, 0, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v813 = __riscv_vneg_v_i8m1(v808, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v814 = __riscv_vmerge_vvm_i8m1(v808, v813, v812, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v815 = __riscv_vwmul_vv_i16m2(v814, v809, v805);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v816 = __riscv_vwredsum_vs_i16m2_i32m1(v815, v804, v805);
    const int8_t* v817 = v779 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v818 = v26[19];
    int v819 = (int) v818;
    const uint8_t v820 = v22[38];
    int v821 = (int) v820;
    int v822 = v701 << 2;
    int v823 = v822 & 256;
    int v824 = v821 | v823;
    const uint8_t v825 = v22[39];
    int v826 = (int) v825;
    int v827 = v701 << 1;
    int v828 = v827 & 256;
    int v829 = v826 | v828;
    const int8_t* v830 = v817 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v831 = __riscv_vsetvl_e8m1(4);
    size_t v832 = v824 * 4;
    const int8_t* v833 = v11 + v832;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v834 = __riscv_vle8_v_i8m1(v833, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v835 = __riscv_vle8_v_i8m1(v817, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v836 = __riscv_vmv_v_x_u8m1(v819, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v837 = __riscv_vand_vv_u8m1(v836, v8, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v838 = __riscv_vmsne_vx_u8m1_b8(v837, 0, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v839 = __riscv_vneg_v_i8m1(v834, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v840 = __riscv_vmerge_vvm_i8m1(v834, v839, v838, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v841 = __riscv_vwmul_vv_i16m2(v840, v835, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v842 = __riscv_vwredsum_vs_i16m2_i32m1(v841, v816, v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v843 = __riscv_vsetvl_e8m1(4);
    size_t v844 = v829 * 4;
    const int8_t* v845 = v11 + v844;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v846 = __riscv_vle8_v_i8m1(v845, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v847 = __riscv_vle8_v_i8m1(v830, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v848 = __riscv_vmv_v_x_u8m1(v819, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v849 = __riscv_vand_vv_u8m1(v848, v10, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v850 = __riscv_vmsne_vx_u8m1_b8(v849, 0, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v851 = __riscv_vneg_v_i8m1(v846, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v852 = __riscv_vmerge_vvm_i8m1(v846, v851, v850, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v853 = __riscv_vwmul_vv_i16m2(v852, v847, v843);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v854 = __riscv_vwredsum_vs_i16m2_i32m1(v853, v842, v843);
    const int8_t* v855 = v817 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v856 = __riscv_vmv_x_s_i32m1_i32(v854);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v857 = v31;
    int32_t v858 = (int32_t) v699;
    int32_t v859 = v856 * v858;
    int32_t v860 = v857 + v859;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v860;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v861 = v28[2];
    int v862 = (int) v861;
    int v863 = v862 >> 4;
    int v864 = v863 * 2;
    int v865 = v864 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v866 = v24[5];
    int v867 = (int) v866;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v868 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v869 = v30 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v870 = v26[20];
    int v871 = (int) v870;
    const uint8_t v872 = v22[40];
    int v873 = (int) v872;
    int v874 = v867 << 8;
    int v875 = v874 & 256;
    int v876 = v873 | v875;
    const uint8_t v877 = v22[41];
    int v878 = (int) v877;
    int v879 = v867 << 7;
    int v880 = v879 & 256;
    int v881 = v878 | v880;
    const int8_t* v882 = v869 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v883 = __riscv_vsetvl_e8m1(4);
    size_t v884 = v876 * 4;
    const int8_t* v885 = v11 + v884;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v886 = __riscv_vle8_v_i8m1(v885, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v887 = __riscv_vle8_v_i8m1(v869, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v888 = __riscv_vmv_v_x_u8m1(v871, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v889 = __riscv_vand_vv_u8m1(v888, v8, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v890 = __riscv_vmsne_vx_u8m1_b8(v889, 0, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v891 = __riscv_vneg_v_i8m1(v886, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v892 = __riscv_vmerge_vvm_i8m1(v886, v891, v890, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v893 = __riscv_vwmul_vv_i16m2(v892, v887, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v894 = __riscv_vwredsum_vs_i16m2_i32m1(v893, v868, v883);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v895 = __riscv_vsetvl_e8m1(4);
    size_t v896 = v881 * 4;
    const int8_t* v897 = v11 + v896;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v898 = __riscv_vle8_v_i8m1(v897, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v899 = __riscv_vle8_v_i8m1(v882, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v900 = __riscv_vmv_v_x_u8m1(v871, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v901 = __riscv_vand_vv_u8m1(v900, v10, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v902 = __riscv_vmsne_vx_u8m1_b8(v901, 0, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v903 = __riscv_vneg_v_i8m1(v898, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v904 = __riscv_vmerge_vvm_i8m1(v898, v903, v902, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v905 = __riscv_vwmul_vv_i16m2(v904, v899, v895);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v906 = __riscv_vwredsum_vs_i16m2_i32m1(v905, v894, v895);
    const int8_t* v907 = v869 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v908 = v26[21];
    int v909 = (int) v908;
    const uint8_t v910 = v22[42];
    int v911 = (int) v910;
    int v912 = v867 << 6;
    int v913 = v912 & 256;
    int v914 = v911 | v913;
    const uint8_t v915 = v22[43];
    int v916 = (int) v915;
    int v917 = v867 << 5;
    int v918 = v917 & 256;
    int v919 = v916 | v918;
    const int8_t* v920 = v907 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v921 = __riscv_vsetvl_e8m1(4);
    size_t v922 = v914 * 4;
    const int8_t* v923 = v11 + v922;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v924 = __riscv_vle8_v_i8m1(v923, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v925 = __riscv_vle8_v_i8m1(v907, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v926 = __riscv_vmv_v_x_u8m1(v909, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v927 = __riscv_vand_vv_u8m1(v926, v8, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v928 = __riscv_vmsne_vx_u8m1_b8(v927, 0, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v929 = __riscv_vneg_v_i8m1(v924, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v930 = __riscv_vmerge_vvm_i8m1(v924, v929, v928, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v931 = __riscv_vwmul_vv_i16m2(v930, v925, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v932 = __riscv_vwredsum_vs_i16m2_i32m1(v931, v906, v921);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v933 = __riscv_vsetvl_e8m1(4);
    size_t v934 = v919 * 4;
    const int8_t* v935 = v11 + v934;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v936 = __riscv_vle8_v_i8m1(v935, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v937 = __riscv_vle8_v_i8m1(v920, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v938 = __riscv_vmv_v_x_u8m1(v909, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v939 = __riscv_vand_vv_u8m1(v938, v10, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v940 = __riscv_vmsne_vx_u8m1_b8(v939, 0, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v941 = __riscv_vneg_v_i8m1(v936, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v942 = __riscv_vmerge_vvm_i8m1(v936, v941, v940, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v943 = __riscv_vwmul_vv_i16m2(v942, v937, v933);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v944 = __riscv_vwredsum_vs_i16m2_i32m1(v943, v932, v933);
    const int8_t* v945 = v907 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v946 = v26[22];
    int v947 = (int) v946;
    const uint8_t v948 = v22[44];
    int v949 = (int) v948;
    int v950 = v867 << 4;
    int v951 = v950 & 256;
    int v952 = v949 | v951;
    const uint8_t v953 = v22[45];
    int v954 = (int) v953;
    int v955 = v867 << 3;
    int v956 = v955 & 256;
    int v957 = v954 | v956;
    const int8_t* v958 = v945 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v959 = __riscv_vsetvl_e8m1(4);
    size_t v960 = v952 * 4;
    const int8_t* v961 = v11 + v960;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v962 = __riscv_vle8_v_i8m1(v961, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v963 = __riscv_vle8_v_i8m1(v945, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v964 = __riscv_vmv_v_x_u8m1(v947, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v965 = __riscv_vand_vv_u8m1(v964, v8, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v966 = __riscv_vmsne_vx_u8m1_b8(v965, 0, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v967 = __riscv_vneg_v_i8m1(v962, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v968 = __riscv_vmerge_vvm_i8m1(v962, v967, v966, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v969 = __riscv_vwmul_vv_i16m2(v968, v963, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v970 = __riscv_vwredsum_vs_i16m2_i32m1(v969, v944, v959);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v971 = __riscv_vsetvl_e8m1(4);
    size_t v972 = v957 * 4;
    const int8_t* v973 = v11 + v972;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v974 = __riscv_vle8_v_i8m1(v973, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v975 = __riscv_vle8_v_i8m1(v958, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v976 = __riscv_vmv_v_x_u8m1(v947, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v977 = __riscv_vand_vv_u8m1(v976, v10, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v978 = __riscv_vmsne_vx_u8m1_b8(v977, 0, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v979 = __riscv_vneg_v_i8m1(v974, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v980 = __riscv_vmerge_vvm_i8m1(v974, v979, v978, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v981 = __riscv_vwmul_vv_i16m2(v980, v975, v971);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v982 = __riscv_vwredsum_vs_i16m2_i32m1(v981, v970, v971);
    const int8_t* v983 = v945 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v984 = v26[23];
    int v985 = (int) v984;
    const uint8_t v986 = v22[46];
    int v987 = (int) v986;
    int v988 = v867 << 2;
    int v989 = v988 & 256;
    int v990 = v987 | v989;
    const uint8_t v991 = v22[47];
    int v992 = (int) v991;
    int v993 = v867 << 1;
    int v994 = v993 & 256;
    int v995 = v992 | v994;
    const int8_t* v996 = v983 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v997 = __riscv_vsetvl_e8m1(4);
    size_t v998 = v990 * 4;
    const int8_t* v999 = v11 + v998;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1000 = __riscv_vle8_v_i8m1(v999, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1001 = __riscv_vle8_v_i8m1(v983, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1002 = __riscv_vmv_v_x_u8m1(v985, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1003 = __riscv_vand_vv_u8m1(v1002, v8, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1004 = __riscv_vmsne_vx_u8m1_b8(v1003, 0, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1005 = __riscv_vneg_v_i8m1(v1000, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1006 = __riscv_vmerge_vvm_i8m1(v1000, v1005, v1004, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1007 = __riscv_vwmul_vv_i16m2(v1006, v1001, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1008 = __riscv_vwredsum_vs_i16m2_i32m1(v1007, v982, v997);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1009 = __riscv_vsetvl_e8m1(4);
    size_t v1010 = v995 * 4;
    const int8_t* v1011 = v11 + v1010;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1012 = __riscv_vle8_v_i8m1(v1011, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1013 = __riscv_vle8_v_i8m1(v996, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1014 = __riscv_vmv_v_x_u8m1(v985, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1015 = __riscv_vand_vv_u8m1(v1014, v10, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1016 = __riscv_vmsne_vx_u8m1_b8(v1015, 0, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1017 = __riscv_vneg_v_i8m1(v1012, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1018 = __riscv_vmerge_vvm_i8m1(v1012, v1017, v1016, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1019 = __riscv_vwmul_vv_i16m2(v1018, v1013, v1009);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1020 = __riscv_vwredsum_vs_i16m2_i32m1(v1019, v1008, v1009);
    const int8_t* v1021 = v983 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1022 = __riscv_vmv_x_s_i32m1_i32(v1020);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1023 = v31;
    int32_t v1024 = (int32_t) v865;
    int32_t v1025 = v1022 * v1024;
    int32_t v1026 = v1023 + v1025;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v1026;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v1027 = v28[3];
    int v1028 = (int) v1027;
    int v1029 = v1028 & 15;
    int v1030 = v1029 * 2;
    int v1031 = v1030 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v1032 = v24[6];
    int v1033 = (int) v1032;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v1034 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v1035 = v30 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1036 = v26[24];
    int v1037 = (int) v1036;
    const uint8_t v1038 = v22[48];
    int v1039 = (int) v1038;
    int v1040 = v1033 << 8;
    int v1041 = v1040 & 256;
    int v1042 = v1039 | v1041;
    const uint8_t v1043 = v22[49];
    int v1044 = (int) v1043;
    int v1045 = v1033 << 7;
    int v1046 = v1045 & 256;
    int v1047 = v1044 | v1046;
    const int8_t* v1048 = v1035 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1049 = __riscv_vsetvl_e8m1(4);
    size_t v1050 = v1042 * 4;
    const int8_t* v1051 = v11 + v1050;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1052 = __riscv_vle8_v_i8m1(v1051, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1053 = __riscv_vle8_v_i8m1(v1035, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1054 = __riscv_vmv_v_x_u8m1(v1037, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1055 = __riscv_vand_vv_u8m1(v1054, v8, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1056 = __riscv_vmsne_vx_u8m1_b8(v1055, 0, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1057 = __riscv_vneg_v_i8m1(v1052, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1058 = __riscv_vmerge_vvm_i8m1(v1052, v1057, v1056, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1059 = __riscv_vwmul_vv_i16m2(v1058, v1053, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1060 = __riscv_vwredsum_vs_i16m2_i32m1(v1059, v1034, v1049);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1061 = __riscv_vsetvl_e8m1(4);
    size_t v1062 = v1047 * 4;
    const int8_t* v1063 = v11 + v1062;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1064 = __riscv_vle8_v_i8m1(v1063, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1065 = __riscv_vle8_v_i8m1(v1048, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1066 = __riscv_vmv_v_x_u8m1(v1037, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1067 = __riscv_vand_vv_u8m1(v1066, v10, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1068 = __riscv_vmsne_vx_u8m1_b8(v1067, 0, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1069 = __riscv_vneg_v_i8m1(v1064, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1070 = __riscv_vmerge_vvm_i8m1(v1064, v1069, v1068, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1071 = __riscv_vwmul_vv_i16m2(v1070, v1065, v1061);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1072 = __riscv_vwredsum_vs_i16m2_i32m1(v1071, v1060, v1061);
    const int8_t* v1073 = v1035 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1074 = v26[25];
    int v1075 = (int) v1074;
    const uint8_t v1076 = v22[50];
    int v1077 = (int) v1076;
    int v1078 = v1033 << 6;
    int v1079 = v1078 & 256;
    int v1080 = v1077 | v1079;
    const uint8_t v1081 = v22[51];
    int v1082 = (int) v1081;
    int v1083 = v1033 << 5;
    int v1084 = v1083 & 256;
    int v1085 = v1082 | v1084;
    const int8_t* v1086 = v1073 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1087 = __riscv_vsetvl_e8m1(4);
    size_t v1088 = v1080 * 4;
    const int8_t* v1089 = v11 + v1088;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1090 = __riscv_vle8_v_i8m1(v1089, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1091 = __riscv_vle8_v_i8m1(v1073, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1092 = __riscv_vmv_v_x_u8m1(v1075, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1093 = __riscv_vand_vv_u8m1(v1092, v8, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1094 = __riscv_vmsne_vx_u8m1_b8(v1093, 0, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1095 = __riscv_vneg_v_i8m1(v1090, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1096 = __riscv_vmerge_vvm_i8m1(v1090, v1095, v1094, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1097 = __riscv_vwmul_vv_i16m2(v1096, v1091, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1098 = __riscv_vwredsum_vs_i16m2_i32m1(v1097, v1072, v1087);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1099 = __riscv_vsetvl_e8m1(4);
    size_t v1100 = v1085 * 4;
    const int8_t* v1101 = v11 + v1100;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1102 = __riscv_vle8_v_i8m1(v1101, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1103 = __riscv_vle8_v_i8m1(v1086, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1104 = __riscv_vmv_v_x_u8m1(v1075, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1105 = __riscv_vand_vv_u8m1(v1104, v10, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1106 = __riscv_vmsne_vx_u8m1_b8(v1105, 0, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1107 = __riscv_vneg_v_i8m1(v1102, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1108 = __riscv_vmerge_vvm_i8m1(v1102, v1107, v1106, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1109 = __riscv_vwmul_vv_i16m2(v1108, v1103, v1099);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1110 = __riscv_vwredsum_vs_i16m2_i32m1(v1109, v1098, v1099);
    const int8_t* v1111 = v1073 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1112 = v26[26];
    int v1113 = (int) v1112;
    const uint8_t v1114 = v22[52];
    int v1115 = (int) v1114;
    int v1116 = v1033 << 4;
    int v1117 = v1116 & 256;
    int v1118 = v1115 | v1117;
    const uint8_t v1119 = v22[53];
    int v1120 = (int) v1119;
    int v1121 = v1033 << 3;
    int v1122 = v1121 & 256;
    int v1123 = v1120 | v1122;
    const int8_t* v1124 = v1111 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1125 = __riscv_vsetvl_e8m1(4);
    size_t v1126 = v1118 * 4;
    const int8_t* v1127 = v11 + v1126;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1128 = __riscv_vle8_v_i8m1(v1127, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1129 = __riscv_vle8_v_i8m1(v1111, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1130 = __riscv_vmv_v_x_u8m1(v1113, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1131 = __riscv_vand_vv_u8m1(v1130, v8, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1132 = __riscv_vmsne_vx_u8m1_b8(v1131, 0, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1133 = __riscv_vneg_v_i8m1(v1128, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1134 = __riscv_vmerge_vvm_i8m1(v1128, v1133, v1132, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1135 = __riscv_vwmul_vv_i16m2(v1134, v1129, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1136 = __riscv_vwredsum_vs_i16m2_i32m1(v1135, v1110, v1125);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1137 = __riscv_vsetvl_e8m1(4);
    size_t v1138 = v1123 * 4;
    const int8_t* v1139 = v11 + v1138;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1140 = __riscv_vle8_v_i8m1(v1139, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1141 = __riscv_vle8_v_i8m1(v1124, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1142 = __riscv_vmv_v_x_u8m1(v1113, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1143 = __riscv_vand_vv_u8m1(v1142, v10, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1144 = __riscv_vmsne_vx_u8m1_b8(v1143, 0, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1145 = __riscv_vneg_v_i8m1(v1140, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1146 = __riscv_vmerge_vvm_i8m1(v1140, v1145, v1144, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1147 = __riscv_vwmul_vv_i16m2(v1146, v1141, v1137);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1148 = __riscv_vwredsum_vs_i16m2_i32m1(v1147, v1136, v1137);
    const int8_t* v1149 = v1111 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1150 = v26[27];
    int v1151 = (int) v1150;
    const uint8_t v1152 = v22[54];
    int v1153 = (int) v1152;
    int v1154 = v1033 << 2;
    int v1155 = v1154 & 256;
    int v1156 = v1153 | v1155;
    const uint8_t v1157 = v22[55];
    int v1158 = (int) v1157;
    int v1159 = v1033 << 1;
    int v1160 = v1159 & 256;
    int v1161 = v1158 | v1160;
    const int8_t* v1162 = v1149 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1163 = __riscv_vsetvl_e8m1(4);
    size_t v1164 = v1156 * 4;
    const int8_t* v1165 = v11 + v1164;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1166 = __riscv_vle8_v_i8m1(v1165, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1167 = __riscv_vle8_v_i8m1(v1149, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1168 = __riscv_vmv_v_x_u8m1(v1151, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1169 = __riscv_vand_vv_u8m1(v1168, v8, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1170 = __riscv_vmsne_vx_u8m1_b8(v1169, 0, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1171 = __riscv_vneg_v_i8m1(v1166, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1172 = __riscv_vmerge_vvm_i8m1(v1166, v1171, v1170, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1173 = __riscv_vwmul_vv_i16m2(v1172, v1167, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1174 = __riscv_vwredsum_vs_i16m2_i32m1(v1173, v1148, v1163);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1175 = __riscv_vsetvl_e8m1(4);
    size_t v1176 = v1161 * 4;
    const int8_t* v1177 = v11 + v1176;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1178 = __riscv_vle8_v_i8m1(v1177, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1179 = __riscv_vle8_v_i8m1(v1162, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1180 = __riscv_vmv_v_x_u8m1(v1151, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1181 = __riscv_vand_vv_u8m1(v1180, v10, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1182 = __riscv_vmsne_vx_u8m1_b8(v1181, 0, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1183 = __riscv_vneg_v_i8m1(v1178, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1184 = __riscv_vmerge_vvm_i8m1(v1178, v1183, v1182, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1185 = __riscv_vwmul_vv_i16m2(v1184, v1179, v1175);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1186 = __riscv_vwredsum_vs_i16m2_i32m1(v1185, v1174, v1175);
    const int8_t* v1187 = v1149 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1188 = __riscv_vmv_x_s_i32m1_i32(v1186);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1189 = v31;
    int32_t v1190 = (int32_t) v1031;
    int32_t v1191 = v1188 * v1190;
    int32_t v1192 = v1189 + v1191;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v1192;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scale
    const uint8_t v1193 = v28[3];
    int v1194 = (int) v1193;
    int v1195 = v1194 >> 4;
    int v1196 = v1195 * 2;
    int v1197 = v1196 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_plane_byte
    const uint8_t v1198 = v24[7];
    int v1199 = (int) v1198;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v1200 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v1201 = v30 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1202 = v26[28];
    int v1203 = (int) v1202;
    const uint8_t v1204 = v22[56];
    int v1205 = (int) v1204;
    int v1206 = v1199 << 8;
    int v1207 = v1206 & 256;
    int v1208 = v1205 | v1207;
    const uint8_t v1209 = v22[57];
    int v1210 = (int) v1209;
    int v1211 = v1199 << 7;
    int v1212 = v1211 & 256;
    int v1213 = v1210 | v1212;
    const int8_t* v1214 = v1201 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1215 = __riscv_vsetvl_e8m1(4);
    size_t v1216 = v1208 * 4;
    const int8_t* v1217 = v11 + v1216;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1218 = __riscv_vle8_v_i8m1(v1217, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1219 = __riscv_vle8_v_i8m1(v1201, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1220 = __riscv_vmv_v_x_u8m1(v1203, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1221 = __riscv_vand_vv_u8m1(v1220, v8, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1222 = __riscv_vmsne_vx_u8m1_b8(v1221, 0, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1223 = __riscv_vneg_v_i8m1(v1218, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1224 = __riscv_vmerge_vvm_i8m1(v1218, v1223, v1222, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1225 = __riscv_vwmul_vv_i16m2(v1224, v1219, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1226 = __riscv_vwredsum_vs_i16m2_i32m1(v1225, v1200, v1215);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1227 = __riscv_vsetvl_e8m1(4);
    size_t v1228 = v1213 * 4;
    const int8_t* v1229 = v11 + v1228;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1230 = __riscv_vle8_v_i8m1(v1229, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1231 = __riscv_vle8_v_i8m1(v1214, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1232 = __riscv_vmv_v_x_u8m1(v1203, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1233 = __riscv_vand_vv_u8m1(v1232, v10, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1234 = __riscv_vmsne_vx_u8m1_b8(v1233, 0, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1235 = __riscv_vneg_v_i8m1(v1230, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1236 = __riscv_vmerge_vvm_i8m1(v1230, v1235, v1234, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1237 = __riscv_vwmul_vv_i16m2(v1236, v1231, v1227);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1238 = __riscv_vwredsum_vs_i16m2_i32m1(v1237, v1226, v1227);
    const int8_t* v1239 = v1201 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1240 = v26[29];
    int v1241 = (int) v1240;
    const uint8_t v1242 = v22[58];
    int v1243 = (int) v1242;
    int v1244 = v1199 << 6;
    int v1245 = v1244 & 256;
    int v1246 = v1243 | v1245;
    const uint8_t v1247 = v22[59];
    int v1248 = (int) v1247;
    int v1249 = v1199 << 5;
    int v1250 = v1249 & 256;
    int v1251 = v1248 | v1250;
    const int8_t* v1252 = v1239 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1253 = __riscv_vsetvl_e8m1(4);
    size_t v1254 = v1246 * 4;
    const int8_t* v1255 = v11 + v1254;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1256 = __riscv_vle8_v_i8m1(v1255, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1257 = __riscv_vle8_v_i8m1(v1239, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1258 = __riscv_vmv_v_x_u8m1(v1241, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1259 = __riscv_vand_vv_u8m1(v1258, v8, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1260 = __riscv_vmsne_vx_u8m1_b8(v1259, 0, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1261 = __riscv_vneg_v_i8m1(v1256, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1262 = __riscv_vmerge_vvm_i8m1(v1256, v1261, v1260, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1263 = __riscv_vwmul_vv_i16m2(v1262, v1257, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1264 = __riscv_vwredsum_vs_i16m2_i32m1(v1263, v1238, v1253);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1265 = __riscv_vsetvl_e8m1(4);
    size_t v1266 = v1251 * 4;
    const int8_t* v1267 = v11 + v1266;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1268 = __riscv_vle8_v_i8m1(v1267, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1269 = __riscv_vle8_v_i8m1(v1252, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1270 = __riscv_vmv_v_x_u8m1(v1241, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1271 = __riscv_vand_vv_u8m1(v1270, v10, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1272 = __riscv_vmsne_vx_u8m1_b8(v1271, 0, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1273 = __riscv_vneg_v_i8m1(v1268, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1274 = __riscv_vmerge_vvm_i8m1(v1268, v1273, v1272, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1275 = __riscv_vwmul_vv_i16m2(v1274, v1269, v1265);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1276 = __riscv_vwredsum_vs_i16m2_i32m1(v1275, v1264, v1265);
    const int8_t* v1277 = v1239 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1278 = v26[30];
    int v1279 = (int) v1278;
    const uint8_t v1280 = v22[60];
    int v1281 = (int) v1280;
    int v1282 = v1199 << 4;
    int v1283 = v1282 & 256;
    int v1284 = v1281 | v1283;
    const uint8_t v1285 = v22[61];
    int v1286 = (int) v1285;
    int v1287 = v1199 << 3;
    int v1288 = v1287 & 256;
    int v1289 = v1286 | v1288;
    const int8_t* v1290 = v1277 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1291 = __riscv_vsetvl_e8m1(4);
    size_t v1292 = v1284 * 4;
    const int8_t* v1293 = v11 + v1292;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1294 = __riscv_vle8_v_i8m1(v1293, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1295 = __riscv_vle8_v_i8m1(v1277, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1296 = __riscv_vmv_v_x_u8m1(v1279, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1297 = __riscv_vand_vv_u8m1(v1296, v8, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1298 = __riscv_vmsne_vx_u8m1_b8(v1297, 0, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1299 = __riscv_vneg_v_i8m1(v1294, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1300 = __riscv_vmerge_vvm_i8m1(v1294, v1299, v1298, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1301 = __riscv_vwmul_vv_i16m2(v1300, v1295, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1302 = __riscv_vwredsum_vs_i16m2_i32m1(v1301, v1276, v1291);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1303 = __riscv_vsetvl_e8m1(4);
    size_t v1304 = v1289 * 4;
    const int8_t* v1305 = v11 + v1304;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1306 = __riscv_vle8_v_i8m1(v1305, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1307 = __riscv_vle8_v_i8m1(v1290, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1308 = __riscv_vmv_v_x_u8m1(v1279, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1309 = __riscv_vand_vv_u8m1(v1308, v10, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1310 = __riscv_vmsne_vx_u8m1_b8(v1309, 0, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1311 = __riscv_vneg_v_i8m1(v1306, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1312 = __riscv_vmerge_vvm_i8m1(v1306, v1311, v1310, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1313 = __riscv_vwmul_vv_i16m2(v1312, v1307, v1303);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1314 = __riscv_vwredsum_vs_i16m2_i32m1(v1313, v1302, v1303);
    const int8_t* v1315 = v1277 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v1316 = v26[31];
    int v1317 = (int) v1316;
    const uint8_t v1318 = v22[62];
    int v1319 = (int) v1318;
    int v1320 = v1199 << 2;
    int v1321 = v1320 & 256;
    int v1322 = v1319 | v1321;
    const uint8_t v1323 = v22[63];
    int v1324 = (int) v1323;
    int v1325 = v1199 << 1;
    int v1326 = v1325 & 256;
    int v1327 = v1324 | v1326;
    const int8_t* v1328 = v1315 + 4;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1329 = __riscv_vsetvl_e8m1(4);
    size_t v1330 = v1322 * 4;
    const int8_t* v1331 = v11 + v1330;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1332 = __riscv_vle8_v_i8m1(v1331, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1333 = __riscv_vle8_v_i8m1(v1315, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1334 = __riscv_vmv_v_x_u8m1(v1317, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1335 = __riscv_vand_vv_u8m1(v1334, v8, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1336 = __riscv_vmsne_vx_u8m1_b8(v1335, 0, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1337 = __riscv_vneg_v_i8m1(v1332, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1338 = __riscv_vmerge_vvm_i8m1(v1332, v1337, v1336, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1339 = __riscv_vwmul_vv_i16m2(v1338, v1333, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1340 = __riscv_vwredsum_vs_i16m2_i32m1(v1339, v1314, v1329);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v1341 = __riscv_vsetvl_e8m1(4);
    size_t v1342 = v1327 * 4;
    const int8_t* v1343 = v11 + v1342;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1344 = __riscv_vle8_v_i8m1(v1343, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v1345 = __riscv_vle8_v_i8m1(v1328, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v1346 = __riscv_vmv_v_x_u8m1(v1317, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v1347 = __riscv_vand_vv_u8m1(v1346, v10, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v1348 = __riscv_vmsne_vx_u8m1_b8(v1347, 0, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v1349 = __riscv_vneg_v_i8m1(v1344, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v1350 = __riscv_vmerge_vvm_i8m1(v1344, v1349, v1348, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v1351 = __riscv_vwmul_vv_i16m2(v1350, v1345, v1341);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v1352 = __riscv_vwredsum_vs_i16m2_i32m1(v1351, v1340, v1341);
    const int8_t* v1353 = v1315 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v1354 = __riscv_vmv_x_s_i32m1_i32(v1352);
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v1355 = v31;
    int32_t v1356 = (int32_t) v1197;
    int32_t v1357 = v1354 * v1356;
    int32_t v1358 = v1355 + v1357;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v31 = v1358;
    // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v1359 = v31;
    float v1360 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v1360 + v20 * (float) v1359;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq3_s_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v1361 = v6;
  v2[0] = v1361;
  return;
}


