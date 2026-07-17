#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
  static const int8_t tcrv_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_i64_view
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=signs_table_i64_view
  const int64_t* v8 = (const int64_t*) tcrv_iq2xxs_signs64;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 66;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v15 = (const float*) v13;
    const float v16 = v15[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v17 = v14 * v16;
    const uint8_t* v18 = v11 + 2;
    const uint8_t* v19 = (const uint8_t*) v18;
    const uint8_t* v20 = v13 + 4;
    const int8_t* v21 = (const int8_t*) v20;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v22;
    v22 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t v23 = v19[4];
    uint32_t v24 = (uint32_t) v23;
    const uint8_t v25 = v19[5];
    uint32_t v26 = (uint32_t) v25;
    uint32_t v27 = v26 << 8u;
    uint32_t v28 = v24 | v27;
    const uint8_t v29 = v19[6];
    uint32_t v30 = (uint32_t) v29;
    uint32_t v31 = v30 << 16u;
    uint32_t v32 = v28 | v31;
    const uint8_t v33 = v19[7];
    uint32_t v34 = (uint32_t) v33;
    uint32_t v35 = v34 << 24u;
    uint32_t v36 = v32 | v35;
    uint32_t v37 = v36 >> 28u;
    int v38 = (int) v37;
    int v39 = v38 * 2;
    int v40 = v39 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v41[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v42[4];
    const uint8_t v43 = v19[0];
    int v44 = (int) v43;
    int v45 = v44 * 8;
    uint16_t v46 = (uint16_t) v45;
    v41[0] = v46;
    uint32_t v47 = v36 >> 0u;
    uint32_t v48 = v47 & 127u;
    int v49 = (int) v48;
    int v50 = v49 * 8;
    uint16_t v51 = (uint16_t) v50;
    v42[0] = v51;
    const uint8_t v52 = v19[1];
    int v53 = (int) v52;
    int v54 = v53 * 8;
    uint16_t v55 = (uint16_t) v54;
    v41[1] = v55;
    uint32_t v56 = v36 >> 7u;
    uint32_t v57 = v56 & 127u;
    int v58 = (int) v57;
    int v59 = v58 * 8;
    uint16_t v60 = (uint16_t) v59;
    v42[1] = v60;
    const uint8_t v61 = v19[2];
    int v62 = (int) v61;
    int v63 = v62 * 8;
    uint16_t v64 = (uint16_t) v63;
    v41[2] = v64;
    uint32_t v65 = v36 >> 14u;
    uint32_t v66 = v65 & 127u;
    int v67 = (int) v66;
    int v68 = v67 * 8;
    uint16_t v69 = (uint16_t) v68;
    v42[2] = v69;
    const uint8_t v70 = v19[3];
    int v71 = (int) v70;
    int v72 = v71 * 8;
    uint16_t v73 = (uint16_t) v72;
    v41[3] = v73;
    uint32_t v74 = v36 >> 21u;
    uint32_t v75 = v74 & 127u;
    int v76 = (int) v75;
    int v77 = v76 * 8;
    uint16_t v78 = (uint16_t) v77;
    v42[3] = v78;
    uint16_t* v79 = &v41[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v80 = __riscv_vle16_v_u16mf4(v79, 4);
    uint16_t* v81 = &v42[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v82 = __riscv_vle16_v_u16mf4(v81, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v83 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v80, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v84 = __riscv_vreinterpret_v_i64m1_i8m1(v83);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v85 = __riscv_vluxei16_v_i64m1(v8, v82, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v86 = __riscv_vreinterpret_v_i64m1_i8m1(v85);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v87 = __riscv_vle8_v_i8m1(v21, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v88 = __riscv_vmul_vv_i8m1(v84, v86, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v89 = __riscv_vwmul_vv_i16m2(v88, v87, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v90 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v91 = __riscv_vwredsum_vs_i16m2_i32m1(v89, v90, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v92 = __riscv_vmv_x_s_i32m1_i32(v91);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v93 = v22;
    int32_t v94 = (int32_t) v40;
    int32_t v95 = v92 * v94;
    int32_t v96 = v93 + v95;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v96;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v97 = v19 + 8;
    const uint8_t v98 = v97[4];
    uint32_t v99 = (uint32_t) v98;
    const uint8_t v100 = v97[5];
    uint32_t v101 = (uint32_t) v100;
    uint32_t v102 = v101 << 8u;
    uint32_t v103 = v99 | v102;
    const uint8_t v104 = v97[6];
    uint32_t v105 = (uint32_t) v104;
    uint32_t v106 = v105 << 16u;
    uint32_t v107 = v103 | v106;
    const uint8_t v108 = v97[7];
    uint32_t v109 = (uint32_t) v108;
    uint32_t v110 = v109 << 24u;
    uint32_t v111 = v107 | v110;
    uint32_t v112 = v111 >> 28u;
    int v113 = (int) v112;
    int v114 = v113 * 2;
    int v115 = v114 + 1;
    const int8_t* v116 = v21 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v117[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v118[4];
    const uint8_t v119 = v97[0];
    int v120 = (int) v119;
    int v121 = v120 * 8;
    uint16_t v122 = (uint16_t) v121;
    v117[0] = v122;
    uint32_t v123 = v111 >> 0u;
    uint32_t v124 = v123 & 127u;
    int v125 = (int) v124;
    int v126 = v125 * 8;
    uint16_t v127 = (uint16_t) v126;
    v118[0] = v127;
    const uint8_t v128 = v97[1];
    int v129 = (int) v128;
    int v130 = v129 * 8;
    uint16_t v131 = (uint16_t) v130;
    v117[1] = v131;
    uint32_t v132 = v111 >> 7u;
    uint32_t v133 = v132 & 127u;
    int v134 = (int) v133;
    int v135 = v134 * 8;
    uint16_t v136 = (uint16_t) v135;
    v118[1] = v136;
    const uint8_t v137 = v97[2];
    int v138 = (int) v137;
    int v139 = v138 * 8;
    uint16_t v140 = (uint16_t) v139;
    v117[2] = v140;
    uint32_t v141 = v111 >> 14u;
    uint32_t v142 = v141 & 127u;
    int v143 = (int) v142;
    int v144 = v143 * 8;
    uint16_t v145 = (uint16_t) v144;
    v118[2] = v145;
    const uint8_t v146 = v97[3];
    int v147 = (int) v146;
    int v148 = v147 * 8;
    uint16_t v149 = (uint16_t) v148;
    v117[3] = v149;
    uint32_t v150 = v111 >> 21u;
    uint32_t v151 = v150 & 127u;
    int v152 = (int) v151;
    int v153 = v152 * 8;
    uint16_t v154 = (uint16_t) v153;
    v118[3] = v154;
    uint16_t* v155 = &v117[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v156 = __riscv_vle16_v_u16mf4(v155, 4);
    uint16_t* v157 = &v118[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v158 = __riscv_vle16_v_u16mf4(v157, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v159 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v156, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v160 = __riscv_vreinterpret_v_i64m1_i8m1(v159);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v161 = __riscv_vluxei16_v_i64m1(v8, v158, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v162 = __riscv_vreinterpret_v_i64m1_i8m1(v161);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v163 = __riscv_vle8_v_i8m1(v116, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v164 = __riscv_vmul_vv_i8m1(v160, v162, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v165 = __riscv_vwmul_vv_i16m2(v164, v163, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v166 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v167 = __riscv_vwredsum_vs_i16m2_i32m1(v165, v166, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v168 = __riscv_vmv_x_s_i32m1_i32(v167);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v169 = v22;
    int32_t v170 = (int32_t) v115;
    int32_t v171 = v168 * v170;
    int32_t v172 = v169 + v171;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v172;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v173 = v19 + 16;
    const uint8_t v174 = v173[4];
    uint32_t v175 = (uint32_t) v174;
    const uint8_t v176 = v173[5];
    uint32_t v177 = (uint32_t) v176;
    uint32_t v178 = v177 << 8u;
    uint32_t v179 = v175 | v178;
    const uint8_t v180 = v173[6];
    uint32_t v181 = (uint32_t) v180;
    uint32_t v182 = v181 << 16u;
    uint32_t v183 = v179 | v182;
    const uint8_t v184 = v173[7];
    uint32_t v185 = (uint32_t) v184;
    uint32_t v186 = v185 << 24u;
    uint32_t v187 = v183 | v186;
    uint32_t v188 = v187 >> 28u;
    int v189 = (int) v188;
    int v190 = v189 * 2;
    int v191 = v190 + 1;
    const int8_t* v192 = v21 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v193[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v194[4];
    const uint8_t v195 = v173[0];
    int v196 = (int) v195;
    int v197 = v196 * 8;
    uint16_t v198 = (uint16_t) v197;
    v193[0] = v198;
    uint32_t v199 = v187 >> 0u;
    uint32_t v200 = v199 & 127u;
    int v201 = (int) v200;
    int v202 = v201 * 8;
    uint16_t v203 = (uint16_t) v202;
    v194[0] = v203;
    const uint8_t v204 = v173[1];
    int v205 = (int) v204;
    int v206 = v205 * 8;
    uint16_t v207 = (uint16_t) v206;
    v193[1] = v207;
    uint32_t v208 = v187 >> 7u;
    uint32_t v209 = v208 & 127u;
    int v210 = (int) v209;
    int v211 = v210 * 8;
    uint16_t v212 = (uint16_t) v211;
    v194[1] = v212;
    const uint8_t v213 = v173[2];
    int v214 = (int) v213;
    int v215 = v214 * 8;
    uint16_t v216 = (uint16_t) v215;
    v193[2] = v216;
    uint32_t v217 = v187 >> 14u;
    uint32_t v218 = v217 & 127u;
    int v219 = (int) v218;
    int v220 = v219 * 8;
    uint16_t v221 = (uint16_t) v220;
    v194[2] = v221;
    const uint8_t v222 = v173[3];
    int v223 = (int) v222;
    int v224 = v223 * 8;
    uint16_t v225 = (uint16_t) v224;
    v193[3] = v225;
    uint32_t v226 = v187 >> 21u;
    uint32_t v227 = v226 & 127u;
    int v228 = (int) v227;
    int v229 = v228 * 8;
    uint16_t v230 = (uint16_t) v229;
    v194[3] = v230;
    uint16_t* v231 = &v193[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v232 = __riscv_vle16_v_u16mf4(v231, 4);
    uint16_t* v233 = &v194[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v234 = __riscv_vle16_v_u16mf4(v233, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v235 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v232, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v236 = __riscv_vreinterpret_v_i64m1_i8m1(v235);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v237 = __riscv_vluxei16_v_i64m1(v8, v234, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v238 = __riscv_vreinterpret_v_i64m1_i8m1(v237);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v239 = __riscv_vle8_v_i8m1(v192, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v240 = __riscv_vmul_vv_i8m1(v236, v238, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v241 = __riscv_vwmul_vv_i16m2(v240, v239, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v242 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v243 = __riscv_vwredsum_vs_i16m2_i32m1(v241, v242, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v244 = __riscv_vmv_x_s_i32m1_i32(v243);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v245 = v22;
    int32_t v246 = (int32_t) v191;
    int32_t v247 = v244 * v246;
    int32_t v248 = v245 + v247;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v248;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v249 = v19 + 24;
    const uint8_t v250 = v249[4];
    uint32_t v251 = (uint32_t) v250;
    const uint8_t v252 = v249[5];
    uint32_t v253 = (uint32_t) v252;
    uint32_t v254 = v253 << 8u;
    uint32_t v255 = v251 | v254;
    const uint8_t v256 = v249[6];
    uint32_t v257 = (uint32_t) v256;
    uint32_t v258 = v257 << 16u;
    uint32_t v259 = v255 | v258;
    const uint8_t v260 = v249[7];
    uint32_t v261 = (uint32_t) v260;
    uint32_t v262 = v261 << 24u;
    uint32_t v263 = v259 | v262;
    uint32_t v264 = v263 >> 28u;
    int v265 = (int) v264;
    int v266 = v265 * 2;
    int v267 = v266 + 1;
    const int8_t* v268 = v21 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v269[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v270[4];
    const uint8_t v271 = v249[0];
    int v272 = (int) v271;
    int v273 = v272 * 8;
    uint16_t v274 = (uint16_t) v273;
    v269[0] = v274;
    uint32_t v275 = v263 >> 0u;
    uint32_t v276 = v275 & 127u;
    int v277 = (int) v276;
    int v278 = v277 * 8;
    uint16_t v279 = (uint16_t) v278;
    v270[0] = v279;
    const uint8_t v280 = v249[1];
    int v281 = (int) v280;
    int v282 = v281 * 8;
    uint16_t v283 = (uint16_t) v282;
    v269[1] = v283;
    uint32_t v284 = v263 >> 7u;
    uint32_t v285 = v284 & 127u;
    int v286 = (int) v285;
    int v287 = v286 * 8;
    uint16_t v288 = (uint16_t) v287;
    v270[1] = v288;
    const uint8_t v289 = v249[2];
    int v290 = (int) v289;
    int v291 = v290 * 8;
    uint16_t v292 = (uint16_t) v291;
    v269[2] = v292;
    uint32_t v293 = v263 >> 14u;
    uint32_t v294 = v293 & 127u;
    int v295 = (int) v294;
    int v296 = v295 * 8;
    uint16_t v297 = (uint16_t) v296;
    v270[2] = v297;
    const uint8_t v298 = v249[3];
    int v299 = (int) v298;
    int v300 = v299 * 8;
    uint16_t v301 = (uint16_t) v300;
    v269[3] = v301;
    uint32_t v302 = v263 >> 21u;
    uint32_t v303 = v302 & 127u;
    int v304 = (int) v303;
    int v305 = v304 * 8;
    uint16_t v306 = (uint16_t) v305;
    v270[3] = v306;
    uint16_t* v307 = &v269[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v308 = __riscv_vle16_v_u16mf4(v307, 4);
    uint16_t* v309 = &v270[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v310 = __riscv_vle16_v_u16mf4(v309, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v311 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v308, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v312 = __riscv_vreinterpret_v_i64m1_i8m1(v311);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v313 = __riscv_vluxei16_v_i64m1(v8, v310, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v314 = __riscv_vreinterpret_v_i64m1_i8m1(v313);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v315 = __riscv_vle8_v_i8m1(v268, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v316 = __riscv_vmul_vv_i8m1(v312, v314, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v317 = __riscv_vwmul_vv_i16m2(v316, v315, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v318 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v319 = __riscv_vwredsum_vs_i16m2_i32m1(v317, v318, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v320 = __riscv_vmv_x_s_i32m1_i32(v319);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v321 = v22;
    int32_t v322 = (int32_t) v267;
    int32_t v323 = v320 * v322;
    int32_t v324 = v321 + v323;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v324;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v325 = v19 + 32;
    const uint8_t v326 = v325[4];
    uint32_t v327 = (uint32_t) v326;
    const uint8_t v328 = v325[5];
    uint32_t v329 = (uint32_t) v328;
    uint32_t v330 = v329 << 8u;
    uint32_t v331 = v327 | v330;
    const uint8_t v332 = v325[6];
    uint32_t v333 = (uint32_t) v332;
    uint32_t v334 = v333 << 16u;
    uint32_t v335 = v331 | v334;
    const uint8_t v336 = v325[7];
    uint32_t v337 = (uint32_t) v336;
    uint32_t v338 = v337 << 24u;
    uint32_t v339 = v335 | v338;
    uint32_t v340 = v339 >> 28u;
    int v341 = (int) v340;
    int v342 = v341 * 2;
    int v343 = v342 + 1;
    const int8_t* v344 = v21 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v345[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v346[4];
    const uint8_t v347 = v325[0];
    int v348 = (int) v347;
    int v349 = v348 * 8;
    uint16_t v350 = (uint16_t) v349;
    v345[0] = v350;
    uint32_t v351 = v339 >> 0u;
    uint32_t v352 = v351 & 127u;
    int v353 = (int) v352;
    int v354 = v353 * 8;
    uint16_t v355 = (uint16_t) v354;
    v346[0] = v355;
    const uint8_t v356 = v325[1];
    int v357 = (int) v356;
    int v358 = v357 * 8;
    uint16_t v359 = (uint16_t) v358;
    v345[1] = v359;
    uint32_t v360 = v339 >> 7u;
    uint32_t v361 = v360 & 127u;
    int v362 = (int) v361;
    int v363 = v362 * 8;
    uint16_t v364 = (uint16_t) v363;
    v346[1] = v364;
    const uint8_t v365 = v325[2];
    int v366 = (int) v365;
    int v367 = v366 * 8;
    uint16_t v368 = (uint16_t) v367;
    v345[2] = v368;
    uint32_t v369 = v339 >> 14u;
    uint32_t v370 = v369 & 127u;
    int v371 = (int) v370;
    int v372 = v371 * 8;
    uint16_t v373 = (uint16_t) v372;
    v346[2] = v373;
    const uint8_t v374 = v325[3];
    int v375 = (int) v374;
    int v376 = v375 * 8;
    uint16_t v377 = (uint16_t) v376;
    v345[3] = v377;
    uint32_t v378 = v339 >> 21u;
    uint32_t v379 = v378 & 127u;
    int v380 = (int) v379;
    int v381 = v380 * 8;
    uint16_t v382 = (uint16_t) v381;
    v346[3] = v382;
    uint16_t* v383 = &v345[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v384 = __riscv_vle16_v_u16mf4(v383, 4);
    uint16_t* v385 = &v346[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v386 = __riscv_vle16_v_u16mf4(v385, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v387 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v384, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v388 = __riscv_vreinterpret_v_i64m1_i8m1(v387);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v389 = __riscv_vluxei16_v_i64m1(v8, v386, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v390 = __riscv_vreinterpret_v_i64m1_i8m1(v389);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v391 = __riscv_vle8_v_i8m1(v344, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v392 = __riscv_vmul_vv_i8m1(v388, v390, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v393 = __riscv_vwmul_vv_i16m2(v392, v391, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v394 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v395 = __riscv_vwredsum_vs_i16m2_i32m1(v393, v394, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v396 = __riscv_vmv_x_s_i32m1_i32(v395);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v397 = v22;
    int32_t v398 = (int32_t) v343;
    int32_t v399 = v396 * v398;
    int32_t v400 = v397 + v399;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v400;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v401 = v19 + 40;
    const uint8_t v402 = v401[4];
    uint32_t v403 = (uint32_t) v402;
    const uint8_t v404 = v401[5];
    uint32_t v405 = (uint32_t) v404;
    uint32_t v406 = v405 << 8u;
    uint32_t v407 = v403 | v406;
    const uint8_t v408 = v401[6];
    uint32_t v409 = (uint32_t) v408;
    uint32_t v410 = v409 << 16u;
    uint32_t v411 = v407 | v410;
    const uint8_t v412 = v401[7];
    uint32_t v413 = (uint32_t) v412;
    uint32_t v414 = v413 << 24u;
    uint32_t v415 = v411 | v414;
    uint32_t v416 = v415 >> 28u;
    int v417 = (int) v416;
    int v418 = v417 * 2;
    int v419 = v418 + 1;
    const int8_t* v420 = v21 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v421[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v422[4];
    const uint8_t v423 = v401[0];
    int v424 = (int) v423;
    int v425 = v424 * 8;
    uint16_t v426 = (uint16_t) v425;
    v421[0] = v426;
    uint32_t v427 = v415 >> 0u;
    uint32_t v428 = v427 & 127u;
    int v429 = (int) v428;
    int v430 = v429 * 8;
    uint16_t v431 = (uint16_t) v430;
    v422[0] = v431;
    const uint8_t v432 = v401[1];
    int v433 = (int) v432;
    int v434 = v433 * 8;
    uint16_t v435 = (uint16_t) v434;
    v421[1] = v435;
    uint32_t v436 = v415 >> 7u;
    uint32_t v437 = v436 & 127u;
    int v438 = (int) v437;
    int v439 = v438 * 8;
    uint16_t v440 = (uint16_t) v439;
    v422[1] = v440;
    const uint8_t v441 = v401[2];
    int v442 = (int) v441;
    int v443 = v442 * 8;
    uint16_t v444 = (uint16_t) v443;
    v421[2] = v444;
    uint32_t v445 = v415 >> 14u;
    uint32_t v446 = v445 & 127u;
    int v447 = (int) v446;
    int v448 = v447 * 8;
    uint16_t v449 = (uint16_t) v448;
    v422[2] = v449;
    const uint8_t v450 = v401[3];
    int v451 = (int) v450;
    int v452 = v451 * 8;
    uint16_t v453 = (uint16_t) v452;
    v421[3] = v453;
    uint32_t v454 = v415 >> 21u;
    uint32_t v455 = v454 & 127u;
    int v456 = (int) v455;
    int v457 = v456 * 8;
    uint16_t v458 = (uint16_t) v457;
    v422[3] = v458;
    uint16_t* v459 = &v421[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v460 = __riscv_vle16_v_u16mf4(v459, 4);
    uint16_t* v461 = &v422[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v462 = __riscv_vle16_v_u16mf4(v461, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v463 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v460, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v464 = __riscv_vreinterpret_v_i64m1_i8m1(v463);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v465 = __riscv_vluxei16_v_i64m1(v8, v462, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v466 = __riscv_vreinterpret_v_i64m1_i8m1(v465);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v467 = __riscv_vle8_v_i8m1(v420, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v468 = __riscv_vmul_vv_i8m1(v464, v466, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v469 = __riscv_vwmul_vv_i16m2(v468, v467, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v470 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v471 = __riscv_vwredsum_vs_i16m2_i32m1(v469, v470, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v472 = __riscv_vmv_x_s_i32m1_i32(v471);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v473 = v22;
    int32_t v474 = (int32_t) v419;
    int32_t v475 = v472 * v474;
    int32_t v476 = v473 + v475;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v476;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v477 = v19 + 48;
    const uint8_t v478 = v477[4];
    uint32_t v479 = (uint32_t) v478;
    const uint8_t v480 = v477[5];
    uint32_t v481 = (uint32_t) v480;
    uint32_t v482 = v481 << 8u;
    uint32_t v483 = v479 | v482;
    const uint8_t v484 = v477[6];
    uint32_t v485 = (uint32_t) v484;
    uint32_t v486 = v485 << 16u;
    uint32_t v487 = v483 | v486;
    const uint8_t v488 = v477[7];
    uint32_t v489 = (uint32_t) v488;
    uint32_t v490 = v489 << 24u;
    uint32_t v491 = v487 | v490;
    uint32_t v492 = v491 >> 28u;
    int v493 = (int) v492;
    int v494 = v493 * 2;
    int v495 = v494 + 1;
    const int8_t* v496 = v21 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v497[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v498[4];
    const uint8_t v499 = v477[0];
    int v500 = (int) v499;
    int v501 = v500 * 8;
    uint16_t v502 = (uint16_t) v501;
    v497[0] = v502;
    uint32_t v503 = v491 >> 0u;
    uint32_t v504 = v503 & 127u;
    int v505 = (int) v504;
    int v506 = v505 * 8;
    uint16_t v507 = (uint16_t) v506;
    v498[0] = v507;
    const uint8_t v508 = v477[1];
    int v509 = (int) v508;
    int v510 = v509 * 8;
    uint16_t v511 = (uint16_t) v510;
    v497[1] = v511;
    uint32_t v512 = v491 >> 7u;
    uint32_t v513 = v512 & 127u;
    int v514 = (int) v513;
    int v515 = v514 * 8;
    uint16_t v516 = (uint16_t) v515;
    v498[1] = v516;
    const uint8_t v517 = v477[2];
    int v518 = (int) v517;
    int v519 = v518 * 8;
    uint16_t v520 = (uint16_t) v519;
    v497[2] = v520;
    uint32_t v521 = v491 >> 14u;
    uint32_t v522 = v521 & 127u;
    int v523 = (int) v522;
    int v524 = v523 * 8;
    uint16_t v525 = (uint16_t) v524;
    v498[2] = v525;
    const uint8_t v526 = v477[3];
    int v527 = (int) v526;
    int v528 = v527 * 8;
    uint16_t v529 = (uint16_t) v528;
    v497[3] = v529;
    uint32_t v530 = v491 >> 21u;
    uint32_t v531 = v530 & 127u;
    int v532 = (int) v531;
    int v533 = v532 * 8;
    uint16_t v534 = (uint16_t) v533;
    v498[3] = v534;
    uint16_t* v535 = &v497[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v536 = __riscv_vle16_v_u16mf4(v535, 4);
    uint16_t* v537 = &v498[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v538 = __riscv_vle16_v_u16mf4(v537, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v539 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v536, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v540 = __riscv_vreinterpret_v_i64m1_i8m1(v539);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v541 = __riscv_vluxei16_v_i64m1(v8, v538, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v542 = __riscv_vreinterpret_v_i64m1_i8m1(v541);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v543 = __riscv_vle8_v_i8m1(v496, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v544 = __riscv_vmul_vv_i8m1(v540, v542, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v545 = __riscv_vwmul_vv_i16m2(v544, v543, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v546 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v547 = __riscv_vwredsum_vs_i16m2_i32m1(v545, v546, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v548 = __riscv_vmv_x_s_i32m1_i32(v547);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v549 = v22;
    int32_t v550 = (int32_t) v495;
    int32_t v551 = v548 * v550;
    int32_t v552 = v549 + v551;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v552;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_aux_scale
    const uint8_t* v553 = v19 + 56;
    const uint8_t v554 = v553[4];
    uint32_t v555 = (uint32_t) v554;
    const uint8_t v556 = v553[5];
    uint32_t v557 = (uint32_t) v556;
    uint32_t v558 = v557 << 8u;
    uint32_t v559 = v555 | v558;
    const uint8_t v560 = v553[6];
    uint32_t v561 = (uint32_t) v560;
    uint32_t v562 = v561 << 16u;
    uint32_t v563 = v559 | v562;
    const uint8_t v564 = v553[7];
    uint32_t v565 = (uint32_t) v564;
    uint32_t v566 = v565 << 24u;
    uint32_t v567 = v563 | v566;
    uint32_t v568 = v567 >> 28u;
    int v569 = (int) v568;
    int v570 = v569 * 2;
    int v571 = v570 + 1;
    const int8_t* v572 = v21 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_subblock
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v573[4];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v574[4];
    const uint8_t v575 = v553[0];
    int v576 = (int) v575;
    int v577 = v576 * 8;
    uint16_t v578 = (uint16_t) v577;
    v573[0] = v578;
    uint32_t v579 = v567 >> 0u;
    uint32_t v580 = v579 & 127u;
    int v581 = (int) v580;
    int v582 = v581 * 8;
    uint16_t v583 = (uint16_t) v582;
    v574[0] = v583;
    const uint8_t v584 = v553[1];
    int v585 = (int) v584;
    int v586 = v585 * 8;
    uint16_t v587 = (uint16_t) v586;
    v573[1] = v587;
    uint32_t v588 = v567 >> 7u;
    uint32_t v589 = v588 & 127u;
    int v590 = (int) v589;
    int v591 = v590 * 8;
    uint16_t v592 = (uint16_t) v591;
    v574[1] = v592;
    const uint8_t v593 = v553[2];
    int v594 = (int) v593;
    int v595 = v594 * 8;
    uint16_t v596 = (uint16_t) v595;
    v573[2] = v596;
    uint32_t v597 = v567 >> 14u;
    uint32_t v598 = v597 & 127u;
    int v599 = (int) v598;
    int v600 = v599 * 8;
    uint16_t v601 = (uint16_t) v600;
    v574[2] = v601;
    const uint8_t v602 = v553[3];
    int v603 = (int) v602;
    int v604 = v603 * 8;
    uint16_t v605 = (uint16_t) v604;
    v573[3] = v605;
    uint32_t v606 = v567 >> 21u;
    uint32_t v607 = v606 & 127u;
    int v608 = (int) v607;
    int v609 = v608 * 8;
    uint16_t v610 = (uint16_t) v609;
    v574[3] = v610;
    uint16_t* v611 = &v573[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v612 = __riscv_vle16_v_u16mf4(v611, 4);
    uint16_t* v613 = &v574[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v614 = __riscv_vle16_v_u16mf4(v613, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v615 = __riscv_vluxei16_v_i64m1(tcrv_iq2xxs_grid, v612, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v616 = __riscv_vreinterpret_v_i64m1_i8m1(v615);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v617 = __riscv_vluxei16_v_i64m1(v8, v614, 4);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v618 = __riscv_vreinterpret_v_i64m1_i8m1(v617);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v619 = __riscv_vle8_v_i8m1(v572, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v620 = __riscv_vmul_vv_i8m1(v616, v618, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v621 = __riscv_vwmul_vv_i16m2(v620, v619, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v622 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v623 = __riscv_vwredsum_vs_i16m2_i32m1(v621, v622, 32);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v624 = __riscv_vmv_x_s_i32m1_i32(v623);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v625 = v22;
    int32_t v626 = (int32_t) v571;
    int32_t v627 = v624 * v626;
    int32_t v628 = v625 + v627;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v22 = v628;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v629 = v22;
    float v630 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v630 + v17 * (float) v629;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v631 = v6;
  float v632 = 0.125f * v631;
  v2[0] = v632;
  return;
}


