#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_PERTURBED_kernel_ggml_vec_dot_iq2_xxs_q8_K_PERTURBED(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x080808080819081aULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
  static const uint8_t tcrv_iq2xxs_ksigns[128] = {0, 129, 130, 3, 132, 200, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
  static const uint8_t tcrv_iq2xxs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq2xxs_kmask, 8);
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_byte_view
  const int8_t* v9 = (const int8_t*) tcrv_iq2xxs_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v7; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 66;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v16 = (const float*) v14;
    const float v17 = v16[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v18 = v15 * v17;
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t* v21 = v14 + 4;
    const int8_t* v22 = (const int8_t*) v21;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v23;
    v23 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v24 = v20[4];
    uint32_t v25 = (uint32_t) v24;
    const uint8_t v26 = v20[5];
    uint32_t v27 = (uint32_t) v26;
    uint32_t v28 = v27 << 8u;
    uint32_t v29 = v25 | v28;
    const uint8_t v30 = v20[6];
    uint32_t v31 = (uint32_t) v30;
    uint32_t v32 = v31 << 16u;
    uint32_t v33 = v29 | v32;
    const uint8_t v34 = v20[7];
    uint32_t v35 = (uint32_t) v34;
    uint32_t v36 = v35 << 24u;
    uint32_t v37 = v33 | v36;
    uint32_t v38 = v37 >> 28u;
    int v39 = (int) v38;
    int v40 = v39 * 2;
    int v41 = v40 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v42 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v43 = v20[0];
    int v44 = (int) v43;
    uint32_t v45 = v37 >> 0u;
    uint32_t v46 = v45 & 127u;
    int v47 = (int) v46;
    const uint8_t v48 = tcrv_iq2xxs_ksigns[v47];
    int v49 = (int) v48;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v50 = __riscv_vsetvl_e8m1(8);
    size_t v51 = v44 * 8;
    const int8_t* v52 = v9 + v51;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v53 = __riscv_vle8_v_i8m1(v52, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v54 = __riscv_vle8_v_i8m1(v22, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v55 = __riscv_vmv_v_x_u8m1(v49, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v56 = __riscv_vand_vv_u8m1(v55, v8, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v57 = __riscv_vmsne_vx_u8m1_b8(v56, 0, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v58 = __riscv_vneg_v_i8m1(v53, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v59 = __riscv_vmerge_vvm_i8m1(v53, v58, v57, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v59, v54, v50);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v61 = __riscv_vwredsum_vs_i16m2_i32m1(v60, v42, v50);
    const int8_t* v62 = v22 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v63 = v20[1];
    int v64 = (int) v63;
    uint32_t v65 = v37 >> 7u;
    uint32_t v66 = v65 & 127u;
    int v67 = (int) v66;
    const uint8_t v68 = tcrv_iq2xxs_ksigns[v67];
    int v69 = (int) v68;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v70 = __riscv_vsetvl_e8m1(8);
    size_t v71 = v64 * 8;
    const int8_t* v72 = v9 + v71;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v73 = __riscv_vle8_v_i8m1(v72, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v74 = __riscv_vle8_v_i8m1(v62, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v75 = __riscv_vmv_v_x_u8m1(v69, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v76 = __riscv_vand_vv_u8m1(v75, v8, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v77 = __riscv_vmsne_vx_u8m1_b8(v76, 0, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v78 = __riscv_vneg_v_i8m1(v73, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v79 = __riscv_vmerge_vvm_i8m1(v73, v78, v77, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v80 = __riscv_vwmul_vv_i16m2(v79, v74, v70);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v81 = __riscv_vwredsum_vs_i16m2_i32m1(v80, v61, v70);
    const int8_t* v82 = v62 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v83 = v20[2];
    int v84 = (int) v83;
    uint32_t v85 = v37 >> 14u;
    uint32_t v86 = v85 & 127u;
    int v87 = (int) v86;
    const uint8_t v88 = tcrv_iq2xxs_ksigns[v87];
    int v89 = (int) v88;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v90 = __riscv_vsetvl_e8m1(8);
    size_t v91 = v84 * 8;
    const int8_t* v92 = v9 + v91;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v92, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v94 = __riscv_vle8_v_i8m1(v82, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v95 = __riscv_vmv_v_x_u8m1(v89, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v96 = __riscv_vand_vv_u8m1(v95, v8, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v97 = __riscv_vmsne_vx_u8m1_b8(v96, 0, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v98 = __riscv_vneg_v_i8m1(v93, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v99 = __riscv_vmerge_vvm_i8m1(v93, v98, v97, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v100 = __riscv_vwmul_vv_i16m2(v99, v94, v90);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v101 = __riscv_vwredsum_vs_i16m2_i32m1(v100, v81, v90);
    const int8_t* v102 = v82 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v103 = v20[3];
    int v104 = (int) v103;
    uint32_t v105 = v37 >> 21u;
    uint32_t v106 = v105 & 127u;
    int v107 = (int) v106;
    const uint8_t v108 = tcrv_iq2xxs_ksigns[v107];
    int v109 = (int) v108;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v110 = __riscv_vsetvl_e8m1(8);
    size_t v111 = v104 * 8;
    const int8_t* v112 = v9 + v111;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v113 = __riscv_vle8_v_i8m1(v112, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v114 = __riscv_vle8_v_i8m1(v102, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v115 = __riscv_vmv_v_x_u8m1(v109, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v116 = __riscv_vand_vv_u8m1(v115, v8, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v117 = __riscv_vmsne_vx_u8m1_b8(v116, 0, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v118 = __riscv_vneg_v_i8m1(v113, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v119 = __riscv_vmerge_vvm_i8m1(v113, v118, v117, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v120 = __riscv_vwmul_vv_i16m2(v119, v114, v110);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v121 = __riscv_vwredsum_vs_i16m2_i32m1(v120, v101, v110);
    const int8_t* v122 = v102 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v123 = __riscv_vmv_x_s_i32m1_i32(v121);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v124 = v23;
    int32_t v125 = (int32_t) v41;
    int32_t v126 = v123 * v125;
    int32_t v127 = v124 + v126;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v127;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v128 = v20 + 8;
    const uint8_t v129 = v128[4];
    uint32_t v130 = (uint32_t) v129;
    const uint8_t v131 = v128[5];
    uint32_t v132 = (uint32_t) v131;
    uint32_t v133 = v132 << 8u;
    uint32_t v134 = v130 | v133;
    const uint8_t v135 = v128[6];
    uint32_t v136 = (uint32_t) v135;
    uint32_t v137 = v136 << 16u;
    uint32_t v138 = v134 | v137;
    const uint8_t v139 = v128[7];
    uint32_t v140 = (uint32_t) v139;
    uint32_t v141 = v140 << 24u;
    uint32_t v142 = v138 | v141;
    uint32_t v143 = v142 >> 28u;
    int v144 = (int) v143;
    int v145 = v144 * 2;
    int v146 = v145 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v147 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v148 = v22 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v149 = v128[0];
    int v150 = (int) v149;
    uint32_t v151 = v142 >> 0u;
    uint32_t v152 = v151 & 127u;
    int v153 = (int) v152;
    const uint8_t v154 = tcrv_iq2xxs_ksigns[v153];
    int v155 = (int) v154;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v156 = __riscv_vsetvl_e8m1(8);
    size_t v157 = v150 * 8;
    const int8_t* v158 = v9 + v157;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v159 = __riscv_vle8_v_i8m1(v158, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v160 = __riscv_vle8_v_i8m1(v148, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v161 = __riscv_vmv_v_x_u8m1(v155, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v162 = __riscv_vand_vv_u8m1(v161, v8, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v163 = __riscv_vmsne_vx_u8m1_b8(v162, 0, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v164 = __riscv_vneg_v_i8m1(v159, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v165 = __riscv_vmerge_vvm_i8m1(v159, v164, v163, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v166 = __riscv_vwmul_vv_i16m2(v165, v160, v156);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v167 = __riscv_vwredsum_vs_i16m2_i32m1(v166, v147, v156);
    const int8_t* v168 = v148 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v169 = v128[1];
    int v170 = (int) v169;
    uint32_t v171 = v142 >> 7u;
    uint32_t v172 = v171 & 127u;
    int v173 = (int) v172;
    const uint8_t v174 = tcrv_iq2xxs_ksigns[v173];
    int v175 = (int) v174;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v176 = __riscv_vsetvl_e8m1(8);
    size_t v177 = v170 * 8;
    const int8_t* v178 = v9 + v177;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v179 = __riscv_vle8_v_i8m1(v178, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v180 = __riscv_vle8_v_i8m1(v168, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v181 = __riscv_vmv_v_x_u8m1(v175, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v182 = __riscv_vand_vv_u8m1(v181, v8, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v183 = __riscv_vmsne_vx_u8m1_b8(v182, 0, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v184 = __riscv_vneg_v_i8m1(v179, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v185 = __riscv_vmerge_vvm_i8m1(v179, v184, v183, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v186 = __riscv_vwmul_vv_i16m2(v185, v180, v176);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v187 = __riscv_vwredsum_vs_i16m2_i32m1(v186, v167, v176);
    const int8_t* v188 = v168 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v189 = v128[2];
    int v190 = (int) v189;
    uint32_t v191 = v142 >> 14u;
    uint32_t v192 = v191 & 127u;
    int v193 = (int) v192;
    const uint8_t v194 = tcrv_iq2xxs_ksigns[v193];
    int v195 = (int) v194;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v196 = __riscv_vsetvl_e8m1(8);
    size_t v197 = v190 * 8;
    const int8_t* v198 = v9 + v197;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v199 = __riscv_vle8_v_i8m1(v198, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v200 = __riscv_vle8_v_i8m1(v188, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v201 = __riscv_vmv_v_x_u8m1(v195, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v202 = __riscv_vand_vv_u8m1(v201, v8, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v203 = __riscv_vmsne_vx_u8m1_b8(v202, 0, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v204 = __riscv_vneg_v_i8m1(v199, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v205 = __riscv_vmerge_vvm_i8m1(v199, v204, v203, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v206 = __riscv_vwmul_vv_i16m2(v205, v200, v196);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v207 = __riscv_vwredsum_vs_i16m2_i32m1(v206, v187, v196);
    const int8_t* v208 = v188 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v209 = v128[3];
    int v210 = (int) v209;
    uint32_t v211 = v142 >> 21u;
    uint32_t v212 = v211 & 127u;
    int v213 = (int) v212;
    const uint8_t v214 = tcrv_iq2xxs_ksigns[v213];
    int v215 = (int) v214;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v216 = __riscv_vsetvl_e8m1(8);
    size_t v217 = v210 * 8;
    const int8_t* v218 = v9 + v217;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v219 = __riscv_vle8_v_i8m1(v218, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v220 = __riscv_vle8_v_i8m1(v208, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v221 = __riscv_vmv_v_x_u8m1(v215, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v222 = __riscv_vand_vv_u8m1(v221, v8, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v223 = __riscv_vmsne_vx_u8m1_b8(v222, 0, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v224 = __riscv_vneg_v_i8m1(v219, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v225 = __riscv_vmerge_vvm_i8m1(v219, v224, v223, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v226 = __riscv_vwmul_vv_i16m2(v225, v220, v216);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v227 = __riscv_vwredsum_vs_i16m2_i32m1(v226, v207, v216);
    const int8_t* v228 = v208 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v229 = __riscv_vmv_x_s_i32m1_i32(v227);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v230 = v23;
    int32_t v231 = (int32_t) v146;
    int32_t v232 = v229 * v231;
    int32_t v233 = v230 + v232;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v233;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v234 = v20 + 16;
    const uint8_t v235 = v234[4];
    uint32_t v236 = (uint32_t) v235;
    const uint8_t v237 = v234[5];
    uint32_t v238 = (uint32_t) v237;
    uint32_t v239 = v238 << 8u;
    uint32_t v240 = v236 | v239;
    const uint8_t v241 = v234[6];
    uint32_t v242 = (uint32_t) v241;
    uint32_t v243 = v242 << 16u;
    uint32_t v244 = v240 | v243;
    const uint8_t v245 = v234[7];
    uint32_t v246 = (uint32_t) v245;
    uint32_t v247 = v246 << 24u;
    uint32_t v248 = v244 | v247;
    uint32_t v249 = v248 >> 28u;
    int v250 = (int) v249;
    int v251 = v250 * 2;
    int v252 = v251 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v253 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v254 = v22 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v255 = v234[0];
    int v256 = (int) v255;
    uint32_t v257 = v248 >> 0u;
    uint32_t v258 = v257 & 127u;
    int v259 = (int) v258;
    const uint8_t v260 = tcrv_iq2xxs_ksigns[v259];
    int v261 = (int) v260;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v262 = __riscv_vsetvl_e8m1(8);
    size_t v263 = v256 * 8;
    const int8_t* v264 = v9 + v263;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v265 = __riscv_vle8_v_i8m1(v264, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v266 = __riscv_vle8_v_i8m1(v254, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v267 = __riscv_vmv_v_x_u8m1(v261, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v268 = __riscv_vand_vv_u8m1(v267, v8, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v269 = __riscv_vmsne_vx_u8m1_b8(v268, 0, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v270 = __riscv_vneg_v_i8m1(v265, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v271 = __riscv_vmerge_vvm_i8m1(v265, v270, v269, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v272 = __riscv_vwmul_vv_i16m2(v271, v266, v262);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v273 = __riscv_vwredsum_vs_i16m2_i32m1(v272, v253, v262);
    const int8_t* v274 = v254 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v275 = v234[1];
    int v276 = (int) v275;
    uint32_t v277 = v248 >> 7u;
    uint32_t v278 = v277 & 127u;
    int v279 = (int) v278;
    const uint8_t v280 = tcrv_iq2xxs_ksigns[v279];
    int v281 = (int) v280;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v282 = __riscv_vsetvl_e8m1(8);
    size_t v283 = v276 * 8;
    const int8_t* v284 = v9 + v283;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v285 = __riscv_vle8_v_i8m1(v284, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v286 = __riscv_vle8_v_i8m1(v274, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v287 = __riscv_vmv_v_x_u8m1(v281, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v288 = __riscv_vand_vv_u8m1(v287, v8, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v289 = __riscv_vmsne_vx_u8m1_b8(v288, 0, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v290 = __riscv_vneg_v_i8m1(v285, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v291 = __riscv_vmerge_vvm_i8m1(v285, v290, v289, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v292 = __riscv_vwmul_vv_i16m2(v291, v286, v282);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v293 = __riscv_vwredsum_vs_i16m2_i32m1(v292, v273, v282);
    const int8_t* v294 = v274 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v295 = v234[2];
    int v296 = (int) v295;
    uint32_t v297 = v248 >> 14u;
    uint32_t v298 = v297 & 127u;
    int v299 = (int) v298;
    const uint8_t v300 = tcrv_iq2xxs_ksigns[v299];
    int v301 = (int) v300;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v302 = __riscv_vsetvl_e8m1(8);
    size_t v303 = v296 * 8;
    const int8_t* v304 = v9 + v303;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v305 = __riscv_vle8_v_i8m1(v304, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v306 = __riscv_vle8_v_i8m1(v294, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v307 = __riscv_vmv_v_x_u8m1(v301, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v308 = __riscv_vand_vv_u8m1(v307, v8, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v309 = __riscv_vmsne_vx_u8m1_b8(v308, 0, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v310 = __riscv_vneg_v_i8m1(v305, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v311 = __riscv_vmerge_vvm_i8m1(v305, v310, v309, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v312 = __riscv_vwmul_vv_i16m2(v311, v306, v302);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v313 = __riscv_vwredsum_vs_i16m2_i32m1(v312, v293, v302);
    const int8_t* v314 = v294 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v315 = v234[3];
    int v316 = (int) v315;
    uint32_t v317 = v248 >> 21u;
    uint32_t v318 = v317 & 127u;
    int v319 = (int) v318;
    const uint8_t v320 = tcrv_iq2xxs_ksigns[v319];
    int v321 = (int) v320;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v322 = __riscv_vsetvl_e8m1(8);
    size_t v323 = v316 * 8;
    const int8_t* v324 = v9 + v323;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v325 = __riscv_vle8_v_i8m1(v324, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v326 = __riscv_vle8_v_i8m1(v314, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v327 = __riscv_vmv_v_x_u8m1(v321, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v328 = __riscv_vand_vv_u8m1(v327, v8, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v329 = __riscv_vmsne_vx_u8m1_b8(v328, 0, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v330 = __riscv_vneg_v_i8m1(v325, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v331 = __riscv_vmerge_vvm_i8m1(v325, v330, v329, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v332 = __riscv_vwmul_vv_i16m2(v331, v326, v322);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v333 = __riscv_vwredsum_vs_i16m2_i32m1(v332, v313, v322);
    const int8_t* v334 = v314 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v335 = __riscv_vmv_x_s_i32m1_i32(v333);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v336 = v23;
    int32_t v337 = (int32_t) v252;
    int32_t v338 = v335 * v337;
    int32_t v339 = v336 + v338;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v339;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v340 = v20 + 24;
    const uint8_t v341 = v340[4];
    uint32_t v342 = (uint32_t) v341;
    const uint8_t v343 = v340[5];
    uint32_t v344 = (uint32_t) v343;
    uint32_t v345 = v344 << 8u;
    uint32_t v346 = v342 | v345;
    const uint8_t v347 = v340[6];
    uint32_t v348 = (uint32_t) v347;
    uint32_t v349 = v348 << 16u;
    uint32_t v350 = v346 | v349;
    const uint8_t v351 = v340[7];
    uint32_t v352 = (uint32_t) v351;
    uint32_t v353 = v352 << 24u;
    uint32_t v354 = v350 | v353;
    uint32_t v355 = v354 >> 28u;
    int v356 = (int) v355;
    int v357 = v356 * 2;
    int v358 = v357 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v359 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v360 = v22 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v361 = v340[0];
    int v362 = (int) v361;
    uint32_t v363 = v354 >> 0u;
    uint32_t v364 = v363 & 127u;
    int v365 = (int) v364;
    const uint8_t v366 = tcrv_iq2xxs_ksigns[v365];
    int v367 = (int) v366;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v368 = __riscv_vsetvl_e8m1(8);
    size_t v369 = v362 * 8;
    const int8_t* v370 = v9 + v369;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v371 = __riscv_vle8_v_i8m1(v370, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v372 = __riscv_vle8_v_i8m1(v360, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v373 = __riscv_vmv_v_x_u8m1(v367, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v374 = __riscv_vand_vv_u8m1(v373, v8, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v375 = __riscv_vmsne_vx_u8m1_b8(v374, 0, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v376 = __riscv_vneg_v_i8m1(v371, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v377 = __riscv_vmerge_vvm_i8m1(v371, v376, v375, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v378 = __riscv_vwmul_vv_i16m2(v377, v372, v368);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v379 = __riscv_vwredsum_vs_i16m2_i32m1(v378, v359, v368);
    const int8_t* v380 = v360 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v381 = v340[1];
    int v382 = (int) v381;
    uint32_t v383 = v354 >> 7u;
    uint32_t v384 = v383 & 127u;
    int v385 = (int) v384;
    const uint8_t v386 = tcrv_iq2xxs_ksigns[v385];
    int v387 = (int) v386;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v388 = __riscv_vsetvl_e8m1(8);
    size_t v389 = v382 * 8;
    const int8_t* v390 = v9 + v389;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v391 = __riscv_vle8_v_i8m1(v390, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v392 = __riscv_vle8_v_i8m1(v380, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v393 = __riscv_vmv_v_x_u8m1(v387, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v394 = __riscv_vand_vv_u8m1(v393, v8, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v395 = __riscv_vmsne_vx_u8m1_b8(v394, 0, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v396 = __riscv_vneg_v_i8m1(v391, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v397 = __riscv_vmerge_vvm_i8m1(v391, v396, v395, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v398 = __riscv_vwmul_vv_i16m2(v397, v392, v388);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v399 = __riscv_vwredsum_vs_i16m2_i32m1(v398, v379, v388);
    const int8_t* v400 = v380 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v401 = v340[2];
    int v402 = (int) v401;
    uint32_t v403 = v354 >> 14u;
    uint32_t v404 = v403 & 127u;
    int v405 = (int) v404;
    const uint8_t v406 = tcrv_iq2xxs_ksigns[v405];
    int v407 = (int) v406;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v408 = __riscv_vsetvl_e8m1(8);
    size_t v409 = v402 * 8;
    const int8_t* v410 = v9 + v409;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v411 = __riscv_vle8_v_i8m1(v410, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v412 = __riscv_vle8_v_i8m1(v400, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v413 = __riscv_vmv_v_x_u8m1(v407, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v414 = __riscv_vand_vv_u8m1(v413, v8, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v415 = __riscv_vmsne_vx_u8m1_b8(v414, 0, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v416 = __riscv_vneg_v_i8m1(v411, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v417 = __riscv_vmerge_vvm_i8m1(v411, v416, v415, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v418 = __riscv_vwmul_vv_i16m2(v417, v412, v408);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v419 = __riscv_vwredsum_vs_i16m2_i32m1(v418, v399, v408);
    const int8_t* v420 = v400 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v421 = v340[3];
    int v422 = (int) v421;
    uint32_t v423 = v354 >> 21u;
    uint32_t v424 = v423 & 127u;
    int v425 = (int) v424;
    const uint8_t v426 = tcrv_iq2xxs_ksigns[v425];
    int v427 = (int) v426;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v428 = __riscv_vsetvl_e8m1(8);
    size_t v429 = v422 * 8;
    const int8_t* v430 = v9 + v429;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v431 = __riscv_vle8_v_i8m1(v430, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v432 = __riscv_vle8_v_i8m1(v420, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v433 = __riscv_vmv_v_x_u8m1(v427, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v434 = __riscv_vand_vv_u8m1(v433, v8, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v435 = __riscv_vmsne_vx_u8m1_b8(v434, 0, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v436 = __riscv_vneg_v_i8m1(v431, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v437 = __riscv_vmerge_vvm_i8m1(v431, v436, v435, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v438 = __riscv_vwmul_vv_i16m2(v437, v432, v428);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v439 = __riscv_vwredsum_vs_i16m2_i32m1(v438, v419, v428);
    const int8_t* v440 = v420 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v441 = __riscv_vmv_x_s_i32m1_i32(v439);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v442 = v23;
    int32_t v443 = (int32_t) v358;
    int32_t v444 = v441 * v443;
    int32_t v445 = v442 + v444;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v445;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v446 = v20 + 32;
    const uint8_t v447 = v446[4];
    uint32_t v448 = (uint32_t) v447;
    const uint8_t v449 = v446[5];
    uint32_t v450 = (uint32_t) v449;
    uint32_t v451 = v450 << 8u;
    uint32_t v452 = v448 | v451;
    const uint8_t v453 = v446[6];
    uint32_t v454 = (uint32_t) v453;
    uint32_t v455 = v454 << 16u;
    uint32_t v456 = v452 | v455;
    const uint8_t v457 = v446[7];
    uint32_t v458 = (uint32_t) v457;
    uint32_t v459 = v458 << 24u;
    uint32_t v460 = v456 | v459;
    uint32_t v461 = v460 >> 28u;
    int v462 = (int) v461;
    int v463 = v462 * 2;
    int v464 = v463 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v465 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v466 = v22 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v467 = v446[0];
    int v468 = (int) v467;
    uint32_t v469 = v460 >> 0u;
    uint32_t v470 = v469 & 127u;
    int v471 = (int) v470;
    const uint8_t v472 = tcrv_iq2xxs_ksigns[v471];
    int v473 = (int) v472;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v474 = __riscv_vsetvl_e8m1(8);
    size_t v475 = v468 * 8;
    const int8_t* v476 = v9 + v475;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v477 = __riscv_vle8_v_i8m1(v476, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v478 = __riscv_vle8_v_i8m1(v466, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v479 = __riscv_vmv_v_x_u8m1(v473, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v480 = __riscv_vand_vv_u8m1(v479, v8, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v481 = __riscv_vmsne_vx_u8m1_b8(v480, 0, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v482 = __riscv_vneg_v_i8m1(v477, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v483 = __riscv_vmerge_vvm_i8m1(v477, v482, v481, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v484 = __riscv_vwmul_vv_i16m2(v483, v478, v474);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v485 = __riscv_vwredsum_vs_i16m2_i32m1(v484, v465, v474);
    const int8_t* v486 = v466 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v487 = v446[1];
    int v488 = (int) v487;
    uint32_t v489 = v460 >> 7u;
    uint32_t v490 = v489 & 127u;
    int v491 = (int) v490;
    const uint8_t v492 = tcrv_iq2xxs_ksigns[v491];
    int v493 = (int) v492;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v494 = __riscv_vsetvl_e8m1(8);
    size_t v495 = v488 * 8;
    const int8_t* v496 = v9 + v495;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v497 = __riscv_vle8_v_i8m1(v496, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v498 = __riscv_vle8_v_i8m1(v486, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v499 = __riscv_vmv_v_x_u8m1(v493, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v500 = __riscv_vand_vv_u8m1(v499, v8, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v501 = __riscv_vmsne_vx_u8m1_b8(v500, 0, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v502 = __riscv_vneg_v_i8m1(v497, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v503 = __riscv_vmerge_vvm_i8m1(v497, v502, v501, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v504 = __riscv_vwmul_vv_i16m2(v503, v498, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v505 = __riscv_vwredsum_vs_i16m2_i32m1(v504, v485, v494);
    const int8_t* v506 = v486 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v507 = v446[2];
    int v508 = (int) v507;
    uint32_t v509 = v460 >> 14u;
    uint32_t v510 = v509 & 127u;
    int v511 = (int) v510;
    const uint8_t v512 = tcrv_iq2xxs_ksigns[v511];
    int v513 = (int) v512;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v514 = __riscv_vsetvl_e8m1(8);
    size_t v515 = v508 * 8;
    const int8_t* v516 = v9 + v515;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v517 = __riscv_vle8_v_i8m1(v516, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v518 = __riscv_vle8_v_i8m1(v506, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v519 = __riscv_vmv_v_x_u8m1(v513, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v520 = __riscv_vand_vv_u8m1(v519, v8, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v521 = __riscv_vmsne_vx_u8m1_b8(v520, 0, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v522 = __riscv_vneg_v_i8m1(v517, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v523 = __riscv_vmerge_vvm_i8m1(v517, v522, v521, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v524 = __riscv_vwmul_vv_i16m2(v523, v518, v514);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v525 = __riscv_vwredsum_vs_i16m2_i32m1(v524, v505, v514);
    const int8_t* v526 = v506 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v527 = v446[3];
    int v528 = (int) v527;
    uint32_t v529 = v460 >> 21u;
    uint32_t v530 = v529 & 127u;
    int v531 = (int) v530;
    const uint8_t v532 = tcrv_iq2xxs_ksigns[v531];
    int v533 = (int) v532;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v534 = __riscv_vsetvl_e8m1(8);
    size_t v535 = v528 * 8;
    const int8_t* v536 = v9 + v535;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v537 = __riscv_vle8_v_i8m1(v536, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v538 = __riscv_vle8_v_i8m1(v526, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v539 = __riscv_vmv_v_x_u8m1(v533, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v540 = __riscv_vand_vv_u8m1(v539, v8, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v541 = __riscv_vmsne_vx_u8m1_b8(v540, 0, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v542 = __riscv_vneg_v_i8m1(v537, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v543 = __riscv_vmerge_vvm_i8m1(v537, v542, v541, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v544 = __riscv_vwmul_vv_i16m2(v543, v538, v534);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v545 = __riscv_vwredsum_vs_i16m2_i32m1(v544, v525, v534);
    const int8_t* v546 = v526 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v547 = __riscv_vmv_x_s_i32m1_i32(v545);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v548 = v23;
    int32_t v549 = (int32_t) v464;
    int32_t v550 = v547 * v549;
    int32_t v551 = v548 + v550;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v551;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v552 = v20 + 40;
    const uint8_t v553 = v552[4];
    uint32_t v554 = (uint32_t) v553;
    const uint8_t v555 = v552[5];
    uint32_t v556 = (uint32_t) v555;
    uint32_t v557 = v556 << 8u;
    uint32_t v558 = v554 | v557;
    const uint8_t v559 = v552[6];
    uint32_t v560 = (uint32_t) v559;
    uint32_t v561 = v560 << 16u;
    uint32_t v562 = v558 | v561;
    const uint8_t v563 = v552[7];
    uint32_t v564 = (uint32_t) v563;
    uint32_t v565 = v564 << 24u;
    uint32_t v566 = v562 | v565;
    uint32_t v567 = v566 >> 28u;
    int v568 = (int) v567;
    int v569 = v568 * 2;
    int v570 = v569 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v571 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v572 = v22 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v573 = v552[0];
    int v574 = (int) v573;
    uint32_t v575 = v566 >> 0u;
    uint32_t v576 = v575 & 127u;
    int v577 = (int) v576;
    const uint8_t v578 = tcrv_iq2xxs_ksigns[v577];
    int v579 = (int) v578;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v580 = __riscv_vsetvl_e8m1(8);
    size_t v581 = v574 * 8;
    const int8_t* v582 = v9 + v581;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v583 = __riscv_vle8_v_i8m1(v582, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v584 = __riscv_vle8_v_i8m1(v572, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v585 = __riscv_vmv_v_x_u8m1(v579, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v586 = __riscv_vand_vv_u8m1(v585, v8, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v587 = __riscv_vmsne_vx_u8m1_b8(v586, 0, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v588 = __riscv_vneg_v_i8m1(v583, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v589 = __riscv_vmerge_vvm_i8m1(v583, v588, v587, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v590 = __riscv_vwmul_vv_i16m2(v589, v584, v580);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v591 = __riscv_vwredsum_vs_i16m2_i32m1(v590, v571, v580);
    const int8_t* v592 = v572 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v593 = v552[1];
    int v594 = (int) v593;
    uint32_t v595 = v566 >> 7u;
    uint32_t v596 = v595 & 127u;
    int v597 = (int) v596;
    const uint8_t v598 = tcrv_iq2xxs_ksigns[v597];
    int v599 = (int) v598;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v600 = __riscv_vsetvl_e8m1(8);
    size_t v601 = v594 * 8;
    const int8_t* v602 = v9 + v601;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v603 = __riscv_vle8_v_i8m1(v602, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v604 = __riscv_vle8_v_i8m1(v592, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v605 = __riscv_vmv_v_x_u8m1(v599, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v606 = __riscv_vand_vv_u8m1(v605, v8, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v607 = __riscv_vmsne_vx_u8m1_b8(v606, 0, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v608 = __riscv_vneg_v_i8m1(v603, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v609 = __riscv_vmerge_vvm_i8m1(v603, v608, v607, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v610 = __riscv_vwmul_vv_i16m2(v609, v604, v600);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v611 = __riscv_vwredsum_vs_i16m2_i32m1(v610, v591, v600);
    const int8_t* v612 = v592 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v613 = v552[2];
    int v614 = (int) v613;
    uint32_t v615 = v566 >> 14u;
    uint32_t v616 = v615 & 127u;
    int v617 = (int) v616;
    const uint8_t v618 = tcrv_iq2xxs_ksigns[v617];
    int v619 = (int) v618;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v620 = __riscv_vsetvl_e8m1(8);
    size_t v621 = v614 * 8;
    const int8_t* v622 = v9 + v621;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v623 = __riscv_vle8_v_i8m1(v622, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v624 = __riscv_vle8_v_i8m1(v612, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v625 = __riscv_vmv_v_x_u8m1(v619, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v626 = __riscv_vand_vv_u8m1(v625, v8, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v627 = __riscv_vmsne_vx_u8m1_b8(v626, 0, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v628 = __riscv_vneg_v_i8m1(v623, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v629 = __riscv_vmerge_vvm_i8m1(v623, v628, v627, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v630 = __riscv_vwmul_vv_i16m2(v629, v624, v620);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v631 = __riscv_vwredsum_vs_i16m2_i32m1(v630, v611, v620);
    const int8_t* v632 = v612 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v633 = v552[3];
    int v634 = (int) v633;
    uint32_t v635 = v566 >> 21u;
    uint32_t v636 = v635 & 127u;
    int v637 = (int) v636;
    const uint8_t v638 = tcrv_iq2xxs_ksigns[v637];
    int v639 = (int) v638;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v640 = __riscv_vsetvl_e8m1(8);
    size_t v641 = v634 * 8;
    const int8_t* v642 = v9 + v641;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v643 = __riscv_vle8_v_i8m1(v642, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v644 = __riscv_vle8_v_i8m1(v632, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v645 = __riscv_vmv_v_x_u8m1(v639, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v646 = __riscv_vand_vv_u8m1(v645, v8, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v647 = __riscv_vmsne_vx_u8m1_b8(v646, 0, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v648 = __riscv_vneg_v_i8m1(v643, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v649 = __riscv_vmerge_vvm_i8m1(v643, v648, v647, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v650 = __riscv_vwmul_vv_i16m2(v649, v644, v640);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v651 = __riscv_vwredsum_vs_i16m2_i32m1(v650, v631, v640);
    const int8_t* v652 = v632 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v653 = __riscv_vmv_x_s_i32m1_i32(v651);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v654 = v23;
    int32_t v655 = (int32_t) v570;
    int32_t v656 = v653 * v655;
    int32_t v657 = v654 + v656;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v657;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v658 = v20 + 48;
    const uint8_t v659 = v658[4];
    uint32_t v660 = (uint32_t) v659;
    const uint8_t v661 = v658[5];
    uint32_t v662 = (uint32_t) v661;
    uint32_t v663 = v662 << 8u;
    uint32_t v664 = v660 | v663;
    const uint8_t v665 = v658[6];
    uint32_t v666 = (uint32_t) v665;
    uint32_t v667 = v666 << 16u;
    uint32_t v668 = v664 | v667;
    const uint8_t v669 = v658[7];
    uint32_t v670 = (uint32_t) v669;
    uint32_t v671 = v670 << 24u;
    uint32_t v672 = v668 | v671;
    uint32_t v673 = v672 >> 28u;
    int v674 = (int) v673;
    int v675 = v674 * 2;
    int v676 = v675 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v677 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v678 = v22 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v679 = v658[0];
    int v680 = (int) v679;
    uint32_t v681 = v672 >> 0u;
    uint32_t v682 = v681 & 127u;
    int v683 = (int) v682;
    const uint8_t v684 = tcrv_iq2xxs_ksigns[v683];
    int v685 = (int) v684;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v686 = __riscv_vsetvl_e8m1(8);
    size_t v687 = v680 * 8;
    const int8_t* v688 = v9 + v687;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v689 = __riscv_vle8_v_i8m1(v688, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v690 = __riscv_vle8_v_i8m1(v678, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v691 = __riscv_vmv_v_x_u8m1(v685, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v692 = __riscv_vand_vv_u8m1(v691, v8, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v693 = __riscv_vmsne_vx_u8m1_b8(v692, 0, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v694 = __riscv_vneg_v_i8m1(v689, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v695 = __riscv_vmerge_vvm_i8m1(v689, v694, v693, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v696 = __riscv_vwmul_vv_i16m2(v695, v690, v686);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v697 = __riscv_vwredsum_vs_i16m2_i32m1(v696, v677, v686);
    const int8_t* v698 = v678 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v699 = v658[1];
    int v700 = (int) v699;
    uint32_t v701 = v672 >> 7u;
    uint32_t v702 = v701 & 127u;
    int v703 = (int) v702;
    const uint8_t v704 = tcrv_iq2xxs_ksigns[v703];
    int v705 = (int) v704;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v706 = __riscv_vsetvl_e8m1(8);
    size_t v707 = v700 * 8;
    const int8_t* v708 = v9 + v707;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v709 = __riscv_vle8_v_i8m1(v708, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v710 = __riscv_vle8_v_i8m1(v698, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v711 = __riscv_vmv_v_x_u8m1(v705, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v712 = __riscv_vand_vv_u8m1(v711, v8, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v713 = __riscv_vmsne_vx_u8m1_b8(v712, 0, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v714 = __riscv_vneg_v_i8m1(v709, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v715 = __riscv_vmerge_vvm_i8m1(v709, v714, v713, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v716 = __riscv_vwmul_vv_i16m2(v715, v710, v706);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v717 = __riscv_vwredsum_vs_i16m2_i32m1(v716, v697, v706);
    const int8_t* v718 = v698 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v719 = v658[2];
    int v720 = (int) v719;
    uint32_t v721 = v672 >> 14u;
    uint32_t v722 = v721 & 127u;
    int v723 = (int) v722;
    const uint8_t v724 = tcrv_iq2xxs_ksigns[v723];
    int v725 = (int) v724;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v726 = __riscv_vsetvl_e8m1(8);
    size_t v727 = v720 * 8;
    const int8_t* v728 = v9 + v727;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v729 = __riscv_vle8_v_i8m1(v728, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v730 = __riscv_vle8_v_i8m1(v718, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v731 = __riscv_vmv_v_x_u8m1(v725, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v732 = __riscv_vand_vv_u8m1(v731, v8, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v733 = __riscv_vmsne_vx_u8m1_b8(v732, 0, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v734 = __riscv_vneg_v_i8m1(v729, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v735 = __riscv_vmerge_vvm_i8m1(v729, v734, v733, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v736 = __riscv_vwmul_vv_i16m2(v735, v730, v726);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v737 = __riscv_vwredsum_vs_i16m2_i32m1(v736, v717, v726);
    const int8_t* v738 = v718 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v739 = v658[3];
    int v740 = (int) v739;
    uint32_t v741 = v672 >> 21u;
    uint32_t v742 = v741 & 127u;
    int v743 = (int) v742;
    const uint8_t v744 = tcrv_iq2xxs_ksigns[v743];
    int v745 = (int) v744;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v746 = __riscv_vsetvl_e8m1(8);
    size_t v747 = v740 * 8;
    const int8_t* v748 = v9 + v747;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v749 = __riscv_vle8_v_i8m1(v748, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v750 = __riscv_vle8_v_i8m1(v738, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v751 = __riscv_vmv_v_x_u8m1(v745, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v752 = __riscv_vand_vv_u8m1(v751, v8, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v753 = __riscv_vmsne_vx_u8m1_b8(v752, 0, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v754 = __riscv_vneg_v_i8m1(v749, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v755 = __riscv_vmerge_vvm_i8m1(v749, v754, v753, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v756 = __riscv_vwmul_vv_i16m2(v755, v750, v746);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v757 = __riscv_vwredsum_vs_i16m2_i32m1(v756, v737, v746);
    const int8_t* v758 = v738 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v759 = __riscv_vmv_x_s_i32m1_i32(v757);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v760 = v23;
    int32_t v761 = (int32_t) v676;
    int32_t v762 = v759 * v761;
    int32_t v763 = v760 + v762;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v763;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v764 = v20 + 56;
    const uint8_t v765 = v764[4];
    uint32_t v766 = (uint32_t) v765;
    const uint8_t v767 = v764[5];
    uint32_t v768 = (uint32_t) v767;
    uint32_t v769 = v768 << 8u;
    uint32_t v770 = v766 | v769;
    const uint8_t v771 = v764[6];
    uint32_t v772 = (uint32_t) v771;
    uint32_t v773 = v772 << 16u;
    uint32_t v774 = v770 | v773;
    const uint8_t v775 = v764[7];
    uint32_t v776 = (uint32_t) v775;
    uint32_t v777 = v776 << 24u;
    uint32_t v778 = v774 | v777;
    uint32_t v779 = v778 >> 28u;
    int v780 = (int) v779;
    int v781 = v780 * 2;
    int v782 = v781 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v783 = __riscv_vmv_v_x_i32m1(0, 1);
    const int8_t* v784 = v22 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v785 = v764[0];
    int v786 = (int) v785;
    uint32_t v787 = v778 >> 0u;
    uint32_t v788 = v787 & 127u;
    int v789 = (int) v788;
    const uint8_t v790 = tcrv_iq2xxs_ksigns[v789];
    int v791 = (int) v790;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v792 = __riscv_vsetvl_e8m1(8);
    size_t v793 = v786 * 8;
    const int8_t* v794 = v9 + v793;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v795 = __riscv_vle8_v_i8m1(v794, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v796 = __riscv_vle8_v_i8m1(v784, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v797 = __riscv_vmv_v_x_u8m1(v791, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v798 = __riscv_vand_vv_u8m1(v797, v8, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v799 = __riscv_vmsne_vx_u8m1_b8(v798, 0, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v800 = __riscv_vneg_v_i8m1(v795, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v801 = __riscv_vmerge_vvm_i8m1(v795, v800, v799, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v802 = __riscv_vwmul_vv_i16m2(v801, v796, v792);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v803 = __riscv_vwredsum_vs_i16m2_i32m1(v802, v783, v792);
    const int8_t* v804 = v784 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v805 = v764[1];
    int v806 = (int) v805;
    uint32_t v807 = v778 >> 7u;
    uint32_t v808 = v807 & 127u;
    int v809 = (int) v808;
    const uint8_t v810 = tcrv_iq2xxs_ksigns[v809];
    int v811 = (int) v810;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v812 = __riscv_vsetvl_e8m1(8);
    size_t v813 = v806 * 8;
    const int8_t* v814 = v9 + v813;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v815 = __riscv_vle8_v_i8m1(v814, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v816 = __riscv_vle8_v_i8m1(v804, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v817 = __riscv_vmv_v_x_u8m1(v811, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v818 = __riscv_vand_vv_u8m1(v817, v8, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v819 = __riscv_vmsne_vx_u8m1_b8(v818, 0, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v820 = __riscv_vneg_v_i8m1(v815, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v821 = __riscv_vmerge_vvm_i8m1(v815, v820, v819, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v822 = __riscv_vwmul_vv_i16m2(v821, v816, v812);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v823 = __riscv_vwredsum_vs_i16m2_i32m1(v822, v803, v812);
    const int8_t* v824 = v804 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v825 = v764[2];
    int v826 = (int) v825;
    uint32_t v827 = v778 >> 14u;
    uint32_t v828 = v827 & 127u;
    int v829 = (int) v828;
    const uint8_t v830 = tcrv_iq2xxs_ksigns[v829];
    int v831 = (int) v830;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v832 = __riscv_vsetvl_e8m1(8);
    size_t v833 = v826 * 8;
    const int8_t* v834 = v9 + v833;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v835 = __riscv_vle8_v_i8m1(v834, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v836 = __riscv_vle8_v_i8m1(v824, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v837 = __riscv_vmv_v_x_u8m1(v831, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v838 = __riscv_vand_vv_u8m1(v837, v8, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v839 = __riscv_vmsne_vx_u8m1_b8(v838, 0, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v840 = __riscv_vneg_v_i8m1(v835, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v841 = __riscv_vmerge_vvm_i8m1(v835, v840, v839, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v842 = __riscv_vwmul_vv_i16m2(v841, v836, v832);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v843 = __riscv_vwredsum_vs_i16m2_i32m1(v842, v823, v832);
    const int8_t* v844 = v824 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v845 = v764[3];
    int v846 = (int) v845;
    uint32_t v847 = v778 >> 21u;
    uint32_t v848 = v847 & 127u;
    int v849 = (int) v848;
    const uint8_t v850 = tcrv_iq2xxs_ksigns[v849];
    int v851 = (int) v850;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v852 = __riscv_vsetvl_e8m1(8);
    size_t v853 = v846 * 8;
    const int8_t* v854 = v9 + v853;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v855 = __riscv_vle8_v_i8m1(v854, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v856 = __riscv_vle8_v_i8m1(v844, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v857 = __riscv_vmv_v_x_u8m1(v851, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v858 = __riscv_vand_vv_u8m1(v857, v8, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v859 = __riscv_vmsne_vx_u8m1_b8(v858, 0, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v860 = __riscv_vneg_v_i8m1(v855, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v861 = __riscv_vmerge_vvm_i8m1(v855, v860, v859, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v862 = __riscv_vwmul_vv_i16m2(v861, v856, v852);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v863 = __riscv_vwredsum_vs_i16m2_i32m1(v862, v843, v852);
    const int8_t* v864 = v844 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v865 = __riscv_vmv_x_s_i32m1_i32(v863);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v866 = v23;
    int32_t v867 = (int32_t) v782;
    int32_t v868 = v865 * v867;
    int32_t v869 = v866 + v868;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v869;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v870 = v23;
    float v871 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v871 + v18 * (float) v870;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v872 = v6;
  float v873 = 0.125f * v872;
  v2[0] = v873;
  return;
}


