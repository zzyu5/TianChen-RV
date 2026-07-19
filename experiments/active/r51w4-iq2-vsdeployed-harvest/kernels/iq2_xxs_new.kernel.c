#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
  static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_i64_view
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signs_table_i64_view
  const int64_t* v8 = (const int64_t*) weft_iq2xxs_signs64;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 66;
    const uint8_t* v11 = v3 + v10;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v4 + v12;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v11);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fold_activation_d
    const float* v15 = (const float*) v13;
    const float v16 = v15[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fold_scale_d
    float v17 = v14 * v16;
    const uint8_t* v18 = v11 + 2;
    const uint8_t* v19 = (const uint8_t*) v18;
    const uint8_t* v20 = v13 + 4;
    const int8_t* v21 = (const int8_t*) v20;
    // weft_emitc.local_variable=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v22;
    v22 = 0;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v23[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v24[8];
    const uint8_t v25 = v19[4];
    uint32_t v26 = (uint32_t) v25;
    const uint8_t v27 = v19[5];
    uint32_t v28 = (uint32_t) v27;
    uint32_t v29 = v28 << 8u;
    uint32_t v30 = v26 | v29;
    const uint8_t v31 = v19[6];
    uint32_t v32 = (uint32_t) v31;
    uint32_t v33 = v32 << 16u;
    uint32_t v34 = v30 | v33;
    const uint8_t v35 = v19[7];
    uint32_t v36 = (uint32_t) v35;
    uint32_t v37 = v36 << 24u;
    uint32_t v38 = v34 | v37;
    uint32_t v39 = v38 >> 28u;
    int v40 = (int) v39;
    int v41 = v40 * 2;
    int v42 = v41 + 1;
    const uint8_t v43 = v19[0];
    int v44 = (int) v43;
    int v45 = v44 * 8;
    uint16_t v46 = (uint16_t) v45;
    v23[0] = v46;
    uint32_t v47 = v38 >> 0u;
    uint32_t v48 = v47 & 127u;
    int v49 = (int) v48;
    int v50 = v49 * 8;
    uint16_t v51 = (uint16_t) v50;
    v24[0] = v51;
    const uint8_t v52 = v19[1];
    int v53 = (int) v52;
    int v54 = v53 * 8;
    uint16_t v55 = (uint16_t) v54;
    v23[1] = v55;
    uint32_t v56 = v38 >> 7u;
    uint32_t v57 = v56 & 127u;
    int v58 = (int) v57;
    int v59 = v58 * 8;
    uint16_t v60 = (uint16_t) v59;
    v24[1] = v60;
    const uint8_t v61 = v19[2];
    int v62 = (int) v61;
    int v63 = v62 * 8;
    uint16_t v64 = (uint16_t) v63;
    v23[2] = v64;
    uint32_t v65 = v38 >> 14u;
    uint32_t v66 = v65 & 127u;
    int v67 = (int) v66;
    int v68 = v67 * 8;
    uint16_t v69 = (uint16_t) v68;
    v24[2] = v69;
    const uint8_t v70 = v19[3];
    int v71 = (int) v70;
    int v72 = v71 * 8;
    uint16_t v73 = (uint16_t) v72;
    v23[3] = v73;
    uint32_t v74 = v38 >> 21u;
    uint32_t v75 = v74 & 127u;
    int v76 = (int) v75;
    int v77 = v76 * 8;
    uint16_t v78 = (uint16_t) v77;
    v24[3] = v78;
    const uint8_t* v79 = v19 + 8;
    const uint8_t v80 = v79[4];
    uint32_t v81 = (uint32_t) v80;
    const uint8_t v82 = v79[5];
    uint32_t v83 = (uint32_t) v82;
    uint32_t v84 = v83 << 8u;
    uint32_t v85 = v81 | v84;
    const uint8_t v86 = v79[6];
    uint32_t v87 = (uint32_t) v86;
    uint32_t v88 = v87 << 16u;
    uint32_t v89 = v85 | v88;
    const uint8_t v90 = v79[7];
    uint32_t v91 = (uint32_t) v90;
    uint32_t v92 = v91 << 24u;
    uint32_t v93 = v89 | v92;
    uint32_t v94 = v93 >> 28u;
    int v95 = (int) v94;
    int v96 = v95 * 2;
    int v97 = v96 + 1;
    const uint8_t v98 = v79[0];
    int v99 = (int) v98;
    int v100 = v99 * 8;
    uint16_t v101 = (uint16_t) v100;
    v23[4] = v101;
    uint32_t v102 = v93 >> 0u;
    uint32_t v103 = v102 & 127u;
    int v104 = (int) v103;
    int v105 = v104 * 8;
    uint16_t v106 = (uint16_t) v105;
    v24[4] = v106;
    const uint8_t v107 = v79[1];
    int v108 = (int) v107;
    int v109 = v108 * 8;
    uint16_t v110 = (uint16_t) v109;
    v23[5] = v110;
    uint32_t v111 = v93 >> 7u;
    uint32_t v112 = v111 & 127u;
    int v113 = (int) v112;
    int v114 = v113 * 8;
    uint16_t v115 = (uint16_t) v114;
    v24[5] = v115;
    const uint8_t v116 = v79[2];
    int v117 = (int) v116;
    int v118 = v117 * 8;
    uint16_t v119 = (uint16_t) v118;
    v23[6] = v119;
    uint32_t v120 = v93 >> 14u;
    uint32_t v121 = v120 & 127u;
    int v122 = (int) v121;
    int v123 = v122 * 8;
    uint16_t v124 = (uint16_t) v123;
    v24[6] = v124;
    const uint8_t v125 = v79[3];
    int v126 = (int) v125;
    int v127 = v126 * 8;
    uint16_t v128 = (uint16_t) v127;
    v23[7] = v128;
    uint32_t v129 = v93 >> 21u;
    uint32_t v130 = v129 & 127u;
    int v131 = (int) v130;
    int v132 = v131 * 8;
    uint16_t v133 = (uint16_t) v132;
    v24[7] = v133;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v134 = &v23[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v135 = __riscv_vle16_v_u16m1(v134, 8);
    uint16_t* v136 = &v24[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v137 = __riscv_vle16_v_u16m1(v136, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v138 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v135, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v139 = __riscv_vreinterpret_v_i64m4_i8m4(v138);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v140 = __riscv_vluxei16_v_i64m4(v8, v137, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v141 = __riscv_vreinterpret_v_i64m4_i8m4(v140);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v142 = __riscv_vle8_v_i8m4(v21, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v143 = __riscv_vmul_vv_i8m4(v139, v141, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v144 = __riscv_vget_v_i8m4_i8m2(v143, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v145 = __riscv_vget_v_i8m4_i8m2(v142, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v146 = __riscv_vwmul_vv_i16m4(v144, v145, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v147 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v148 = __riscv_vwredsum_vs_i16m4_i32m1(v146, v147, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v149 = __riscv_vmv_x_s_i32m1_i32(v148);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v150 = v22;
    int32_t v151 = (int32_t) v42;
    int32_t v152 = v149 * v151;
    int32_t v153 = v150 + v152;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v153;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v154 = __riscv_vslidedown_vx_i8m4(v143, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v155 = __riscv_vslidedown_vx_i8m4(v142, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v156 = __riscv_vget_v_i8m4_i8m2(v154, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v157 = __riscv_vget_v_i8m4_i8m2(v155, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v158 = __riscv_vwmul_vv_i16m4(v156, v157, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v159 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v160 = __riscv_vwredsum_vs_i16m4_i32m1(v158, v159, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v161 = __riscv_vmv_x_s_i32m1_i32(v160);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v162 = v22;
    int32_t v163 = (int32_t) v97;
    int32_t v164 = v161 * v163;
    int32_t v165 = v162 + v164;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v165;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v166[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v167[8];
    const uint8_t* v168 = v19 + 16;
    const uint8_t v169 = v168[4];
    uint32_t v170 = (uint32_t) v169;
    const uint8_t v171 = v168[5];
    uint32_t v172 = (uint32_t) v171;
    uint32_t v173 = v172 << 8u;
    uint32_t v174 = v170 | v173;
    const uint8_t v175 = v168[6];
    uint32_t v176 = (uint32_t) v175;
    uint32_t v177 = v176 << 16u;
    uint32_t v178 = v174 | v177;
    const uint8_t v179 = v168[7];
    uint32_t v180 = (uint32_t) v179;
    uint32_t v181 = v180 << 24u;
    uint32_t v182 = v178 | v181;
    uint32_t v183 = v182 >> 28u;
    int v184 = (int) v183;
    int v185 = v184 * 2;
    int v186 = v185 + 1;
    const uint8_t v187 = v168[0];
    int v188 = (int) v187;
    int v189 = v188 * 8;
    uint16_t v190 = (uint16_t) v189;
    v166[0] = v190;
    uint32_t v191 = v182 >> 0u;
    uint32_t v192 = v191 & 127u;
    int v193 = (int) v192;
    int v194 = v193 * 8;
    uint16_t v195 = (uint16_t) v194;
    v167[0] = v195;
    const uint8_t v196 = v168[1];
    int v197 = (int) v196;
    int v198 = v197 * 8;
    uint16_t v199 = (uint16_t) v198;
    v166[1] = v199;
    uint32_t v200 = v182 >> 7u;
    uint32_t v201 = v200 & 127u;
    int v202 = (int) v201;
    int v203 = v202 * 8;
    uint16_t v204 = (uint16_t) v203;
    v167[1] = v204;
    const uint8_t v205 = v168[2];
    int v206 = (int) v205;
    int v207 = v206 * 8;
    uint16_t v208 = (uint16_t) v207;
    v166[2] = v208;
    uint32_t v209 = v182 >> 14u;
    uint32_t v210 = v209 & 127u;
    int v211 = (int) v210;
    int v212 = v211 * 8;
    uint16_t v213 = (uint16_t) v212;
    v167[2] = v213;
    const uint8_t v214 = v168[3];
    int v215 = (int) v214;
    int v216 = v215 * 8;
    uint16_t v217 = (uint16_t) v216;
    v166[3] = v217;
    uint32_t v218 = v182 >> 21u;
    uint32_t v219 = v218 & 127u;
    int v220 = (int) v219;
    int v221 = v220 * 8;
    uint16_t v222 = (uint16_t) v221;
    v167[3] = v222;
    const uint8_t* v223 = v19 + 24;
    const uint8_t v224 = v223[4];
    uint32_t v225 = (uint32_t) v224;
    const uint8_t v226 = v223[5];
    uint32_t v227 = (uint32_t) v226;
    uint32_t v228 = v227 << 8u;
    uint32_t v229 = v225 | v228;
    const uint8_t v230 = v223[6];
    uint32_t v231 = (uint32_t) v230;
    uint32_t v232 = v231 << 16u;
    uint32_t v233 = v229 | v232;
    const uint8_t v234 = v223[7];
    uint32_t v235 = (uint32_t) v234;
    uint32_t v236 = v235 << 24u;
    uint32_t v237 = v233 | v236;
    uint32_t v238 = v237 >> 28u;
    int v239 = (int) v238;
    int v240 = v239 * 2;
    int v241 = v240 + 1;
    const uint8_t v242 = v223[0];
    int v243 = (int) v242;
    int v244 = v243 * 8;
    uint16_t v245 = (uint16_t) v244;
    v166[4] = v245;
    uint32_t v246 = v237 >> 0u;
    uint32_t v247 = v246 & 127u;
    int v248 = (int) v247;
    int v249 = v248 * 8;
    uint16_t v250 = (uint16_t) v249;
    v167[4] = v250;
    const uint8_t v251 = v223[1];
    int v252 = (int) v251;
    int v253 = v252 * 8;
    uint16_t v254 = (uint16_t) v253;
    v166[5] = v254;
    uint32_t v255 = v237 >> 7u;
    uint32_t v256 = v255 & 127u;
    int v257 = (int) v256;
    int v258 = v257 * 8;
    uint16_t v259 = (uint16_t) v258;
    v167[5] = v259;
    const uint8_t v260 = v223[2];
    int v261 = (int) v260;
    int v262 = v261 * 8;
    uint16_t v263 = (uint16_t) v262;
    v166[6] = v263;
    uint32_t v264 = v237 >> 14u;
    uint32_t v265 = v264 & 127u;
    int v266 = (int) v265;
    int v267 = v266 * 8;
    uint16_t v268 = (uint16_t) v267;
    v167[6] = v268;
    const uint8_t v269 = v223[3];
    int v270 = (int) v269;
    int v271 = v270 * 8;
    uint16_t v272 = (uint16_t) v271;
    v166[7] = v272;
    uint32_t v273 = v237 >> 21u;
    uint32_t v274 = v273 & 127u;
    int v275 = (int) v274;
    int v276 = v275 * 8;
    uint16_t v277 = (uint16_t) v276;
    v167[7] = v277;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v278 = &v166[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v279 = __riscv_vle16_v_u16m1(v278, 8);
    uint16_t* v280 = &v167[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v281 = __riscv_vle16_v_u16m1(v280, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v282 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v279, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v283 = __riscv_vreinterpret_v_i64m4_i8m4(v282);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v284 = __riscv_vluxei16_v_i64m4(v8, v281, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v285 = __riscv_vreinterpret_v_i64m4_i8m4(v284);
    const int8_t* v286 = v21 + 64;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v287 = __riscv_vle8_v_i8m4(v286, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v288 = __riscv_vmul_vv_i8m4(v283, v285, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v289 = __riscv_vget_v_i8m4_i8m2(v288, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v290 = __riscv_vget_v_i8m4_i8m2(v287, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v291 = __riscv_vwmul_vv_i16m4(v289, v290, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v292 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v293 = __riscv_vwredsum_vs_i16m4_i32m1(v291, v292, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v294 = __riscv_vmv_x_s_i32m1_i32(v293);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v295 = v22;
    int32_t v296 = (int32_t) v186;
    int32_t v297 = v294 * v296;
    int32_t v298 = v295 + v297;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v298;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v299 = __riscv_vslidedown_vx_i8m4(v288, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v300 = __riscv_vslidedown_vx_i8m4(v287, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v301 = __riscv_vget_v_i8m4_i8m2(v299, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v302 = __riscv_vget_v_i8m4_i8m2(v300, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v303 = __riscv_vwmul_vv_i16m4(v301, v302, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v304 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v305 = __riscv_vwredsum_vs_i16m4_i32m1(v303, v304, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v306 = __riscv_vmv_x_s_i32m1_i32(v305);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v307 = v22;
    int32_t v308 = (int32_t) v241;
    int32_t v309 = v306 * v308;
    int32_t v310 = v307 + v309;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v310;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v311[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v312[8];
    const uint8_t* v313 = v19 + 32;
    const uint8_t v314 = v313[4];
    uint32_t v315 = (uint32_t) v314;
    const uint8_t v316 = v313[5];
    uint32_t v317 = (uint32_t) v316;
    uint32_t v318 = v317 << 8u;
    uint32_t v319 = v315 | v318;
    const uint8_t v320 = v313[6];
    uint32_t v321 = (uint32_t) v320;
    uint32_t v322 = v321 << 16u;
    uint32_t v323 = v319 | v322;
    const uint8_t v324 = v313[7];
    uint32_t v325 = (uint32_t) v324;
    uint32_t v326 = v325 << 24u;
    uint32_t v327 = v323 | v326;
    uint32_t v328 = v327 >> 28u;
    int v329 = (int) v328;
    int v330 = v329 * 2;
    int v331 = v330 + 1;
    const uint8_t v332 = v313[0];
    int v333 = (int) v332;
    int v334 = v333 * 8;
    uint16_t v335 = (uint16_t) v334;
    v311[0] = v335;
    uint32_t v336 = v327 >> 0u;
    uint32_t v337 = v336 & 127u;
    int v338 = (int) v337;
    int v339 = v338 * 8;
    uint16_t v340 = (uint16_t) v339;
    v312[0] = v340;
    const uint8_t v341 = v313[1];
    int v342 = (int) v341;
    int v343 = v342 * 8;
    uint16_t v344 = (uint16_t) v343;
    v311[1] = v344;
    uint32_t v345 = v327 >> 7u;
    uint32_t v346 = v345 & 127u;
    int v347 = (int) v346;
    int v348 = v347 * 8;
    uint16_t v349 = (uint16_t) v348;
    v312[1] = v349;
    const uint8_t v350 = v313[2];
    int v351 = (int) v350;
    int v352 = v351 * 8;
    uint16_t v353 = (uint16_t) v352;
    v311[2] = v353;
    uint32_t v354 = v327 >> 14u;
    uint32_t v355 = v354 & 127u;
    int v356 = (int) v355;
    int v357 = v356 * 8;
    uint16_t v358 = (uint16_t) v357;
    v312[2] = v358;
    const uint8_t v359 = v313[3];
    int v360 = (int) v359;
    int v361 = v360 * 8;
    uint16_t v362 = (uint16_t) v361;
    v311[3] = v362;
    uint32_t v363 = v327 >> 21u;
    uint32_t v364 = v363 & 127u;
    int v365 = (int) v364;
    int v366 = v365 * 8;
    uint16_t v367 = (uint16_t) v366;
    v312[3] = v367;
    const uint8_t* v368 = v19 + 40;
    const uint8_t v369 = v368[4];
    uint32_t v370 = (uint32_t) v369;
    const uint8_t v371 = v368[5];
    uint32_t v372 = (uint32_t) v371;
    uint32_t v373 = v372 << 8u;
    uint32_t v374 = v370 | v373;
    const uint8_t v375 = v368[6];
    uint32_t v376 = (uint32_t) v375;
    uint32_t v377 = v376 << 16u;
    uint32_t v378 = v374 | v377;
    const uint8_t v379 = v368[7];
    uint32_t v380 = (uint32_t) v379;
    uint32_t v381 = v380 << 24u;
    uint32_t v382 = v378 | v381;
    uint32_t v383 = v382 >> 28u;
    int v384 = (int) v383;
    int v385 = v384 * 2;
    int v386 = v385 + 1;
    const uint8_t v387 = v368[0];
    int v388 = (int) v387;
    int v389 = v388 * 8;
    uint16_t v390 = (uint16_t) v389;
    v311[4] = v390;
    uint32_t v391 = v382 >> 0u;
    uint32_t v392 = v391 & 127u;
    int v393 = (int) v392;
    int v394 = v393 * 8;
    uint16_t v395 = (uint16_t) v394;
    v312[4] = v395;
    const uint8_t v396 = v368[1];
    int v397 = (int) v396;
    int v398 = v397 * 8;
    uint16_t v399 = (uint16_t) v398;
    v311[5] = v399;
    uint32_t v400 = v382 >> 7u;
    uint32_t v401 = v400 & 127u;
    int v402 = (int) v401;
    int v403 = v402 * 8;
    uint16_t v404 = (uint16_t) v403;
    v312[5] = v404;
    const uint8_t v405 = v368[2];
    int v406 = (int) v405;
    int v407 = v406 * 8;
    uint16_t v408 = (uint16_t) v407;
    v311[6] = v408;
    uint32_t v409 = v382 >> 14u;
    uint32_t v410 = v409 & 127u;
    int v411 = (int) v410;
    int v412 = v411 * 8;
    uint16_t v413 = (uint16_t) v412;
    v312[6] = v413;
    const uint8_t v414 = v368[3];
    int v415 = (int) v414;
    int v416 = v415 * 8;
    uint16_t v417 = (uint16_t) v416;
    v311[7] = v417;
    uint32_t v418 = v382 >> 21u;
    uint32_t v419 = v418 & 127u;
    int v420 = (int) v419;
    int v421 = v420 * 8;
    uint16_t v422 = (uint16_t) v421;
    v312[7] = v422;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v423 = &v311[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v424 = __riscv_vle16_v_u16m1(v423, 8);
    uint16_t* v425 = &v312[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v426 = __riscv_vle16_v_u16m1(v425, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v427 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v424, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v428 = __riscv_vreinterpret_v_i64m4_i8m4(v427);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v429 = __riscv_vluxei16_v_i64m4(v8, v426, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v430 = __riscv_vreinterpret_v_i64m4_i8m4(v429);
    const int8_t* v431 = v21 + 128;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v432 = __riscv_vle8_v_i8m4(v431, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v433 = __riscv_vmul_vv_i8m4(v428, v430, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v434 = __riscv_vget_v_i8m4_i8m2(v433, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v435 = __riscv_vget_v_i8m4_i8m2(v432, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v436 = __riscv_vwmul_vv_i16m4(v434, v435, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v437 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v438 = __riscv_vwredsum_vs_i16m4_i32m1(v436, v437, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v439 = __riscv_vmv_x_s_i32m1_i32(v438);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v440 = v22;
    int32_t v441 = (int32_t) v331;
    int32_t v442 = v439 * v441;
    int32_t v443 = v440 + v442;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v443;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v444 = __riscv_vslidedown_vx_i8m4(v433, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v445 = __riscv_vslidedown_vx_i8m4(v432, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v446 = __riscv_vget_v_i8m4_i8m2(v444, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v447 = __riscv_vget_v_i8m4_i8m2(v445, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v448 = __riscv_vwmul_vv_i16m4(v446, v447, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v449 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v450 = __riscv_vwredsum_vs_i16m4_i32m1(v448, v449, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v451 = __riscv_vmv_x_s_i32m1_i32(v450);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v452 = v22;
    int32_t v453 = (int32_t) v386;
    int32_t v454 = v451 * v453;
    int32_t v455 = v452 + v454;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v455;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v456[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v457[8];
    const uint8_t* v458 = v19 + 48;
    const uint8_t v459 = v458[4];
    uint32_t v460 = (uint32_t) v459;
    const uint8_t v461 = v458[5];
    uint32_t v462 = (uint32_t) v461;
    uint32_t v463 = v462 << 8u;
    uint32_t v464 = v460 | v463;
    const uint8_t v465 = v458[6];
    uint32_t v466 = (uint32_t) v465;
    uint32_t v467 = v466 << 16u;
    uint32_t v468 = v464 | v467;
    const uint8_t v469 = v458[7];
    uint32_t v470 = (uint32_t) v469;
    uint32_t v471 = v470 << 24u;
    uint32_t v472 = v468 | v471;
    uint32_t v473 = v472 >> 28u;
    int v474 = (int) v473;
    int v475 = v474 * 2;
    int v476 = v475 + 1;
    const uint8_t v477 = v458[0];
    int v478 = (int) v477;
    int v479 = v478 * 8;
    uint16_t v480 = (uint16_t) v479;
    v456[0] = v480;
    uint32_t v481 = v472 >> 0u;
    uint32_t v482 = v481 & 127u;
    int v483 = (int) v482;
    int v484 = v483 * 8;
    uint16_t v485 = (uint16_t) v484;
    v457[0] = v485;
    const uint8_t v486 = v458[1];
    int v487 = (int) v486;
    int v488 = v487 * 8;
    uint16_t v489 = (uint16_t) v488;
    v456[1] = v489;
    uint32_t v490 = v472 >> 7u;
    uint32_t v491 = v490 & 127u;
    int v492 = (int) v491;
    int v493 = v492 * 8;
    uint16_t v494 = (uint16_t) v493;
    v457[1] = v494;
    const uint8_t v495 = v458[2];
    int v496 = (int) v495;
    int v497 = v496 * 8;
    uint16_t v498 = (uint16_t) v497;
    v456[2] = v498;
    uint32_t v499 = v472 >> 14u;
    uint32_t v500 = v499 & 127u;
    int v501 = (int) v500;
    int v502 = v501 * 8;
    uint16_t v503 = (uint16_t) v502;
    v457[2] = v503;
    const uint8_t v504 = v458[3];
    int v505 = (int) v504;
    int v506 = v505 * 8;
    uint16_t v507 = (uint16_t) v506;
    v456[3] = v507;
    uint32_t v508 = v472 >> 21u;
    uint32_t v509 = v508 & 127u;
    int v510 = (int) v509;
    int v511 = v510 * 8;
    uint16_t v512 = (uint16_t) v511;
    v457[3] = v512;
    const uint8_t* v513 = v19 + 56;
    const uint8_t v514 = v513[4];
    uint32_t v515 = (uint32_t) v514;
    const uint8_t v516 = v513[5];
    uint32_t v517 = (uint32_t) v516;
    uint32_t v518 = v517 << 8u;
    uint32_t v519 = v515 | v518;
    const uint8_t v520 = v513[6];
    uint32_t v521 = (uint32_t) v520;
    uint32_t v522 = v521 << 16u;
    uint32_t v523 = v519 | v522;
    const uint8_t v524 = v513[7];
    uint32_t v525 = (uint32_t) v524;
    uint32_t v526 = v525 << 24u;
    uint32_t v527 = v523 | v526;
    uint32_t v528 = v527 >> 28u;
    int v529 = (int) v528;
    int v530 = v529 * 2;
    int v531 = v530 + 1;
    const uint8_t v532 = v513[0];
    int v533 = (int) v532;
    int v534 = v533 * 8;
    uint16_t v535 = (uint16_t) v534;
    v456[4] = v535;
    uint32_t v536 = v527 >> 0u;
    uint32_t v537 = v536 & 127u;
    int v538 = (int) v537;
    int v539 = v538 * 8;
    uint16_t v540 = (uint16_t) v539;
    v457[4] = v540;
    const uint8_t v541 = v513[1];
    int v542 = (int) v541;
    int v543 = v542 * 8;
    uint16_t v544 = (uint16_t) v543;
    v456[5] = v544;
    uint32_t v545 = v527 >> 7u;
    uint32_t v546 = v545 & 127u;
    int v547 = (int) v546;
    int v548 = v547 * 8;
    uint16_t v549 = (uint16_t) v548;
    v457[5] = v549;
    const uint8_t v550 = v513[2];
    int v551 = (int) v550;
    int v552 = v551 * 8;
    uint16_t v553 = (uint16_t) v552;
    v456[6] = v553;
    uint32_t v554 = v527 >> 14u;
    uint32_t v555 = v554 & 127u;
    int v556 = (int) v555;
    int v557 = v556 * 8;
    uint16_t v558 = (uint16_t) v557;
    v457[6] = v558;
    const uint8_t v559 = v513[3];
    int v560 = (int) v559;
    int v561 = v560 * 8;
    uint16_t v562 = (uint16_t) v561;
    v456[7] = v562;
    uint32_t v563 = v527 >> 21u;
    uint32_t v564 = v563 & 127u;
    int v565 = (int) v564;
    int v566 = v565 * 8;
    uint16_t v567 = (uint16_t) v566;
    v457[7] = v567;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v568 = &v456[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v569 = __riscv_vle16_v_u16m1(v568, 8);
    uint16_t* v570 = &v457[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v571 = __riscv_vle16_v_u16m1(v570, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v572 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v569, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v573 = __riscv_vreinterpret_v_i64m4_i8m4(v572);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v574 = __riscv_vluxei16_v_i64m4(v8, v571, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v575 = __riscv_vreinterpret_v_i64m4_i8m4(v574);
    const int8_t* v576 = v21 + 192;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v577 = __riscv_vle8_v_i8m4(v576, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v578 = __riscv_vmul_vv_i8m4(v573, v575, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v579 = __riscv_vget_v_i8m4_i8m2(v578, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v580 = __riscv_vget_v_i8m4_i8m2(v577, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v581 = __riscv_vwmul_vv_i16m4(v579, v580, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v582 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v583 = __riscv_vwredsum_vs_i16m4_i32m1(v581, v582, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v584 = __riscv_vmv_x_s_i32m1_i32(v583);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v585 = v22;
    int32_t v586 = (int32_t) v476;
    int32_t v587 = v584 * v586;
    int32_t v588 = v585 + v587;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v588;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v589 = __riscv_vslidedown_vx_i8m4(v578, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v590 = __riscv_vslidedown_vx_i8m4(v577, 32, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v591 = __riscv_vget_v_i8m4_i8m2(v589, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v592 = __riscv_vget_v_i8m4_i8m2(v590, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v593 = __riscv_vwmul_vv_i16m4(v591, v592, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v594 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v595 = __riscv_vwredsum_vs_i16m4_i32m1(v593, v594, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v596 = __riscv_vmv_x_s_i32m1_i32(v595);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v597 = v22;
    int32_t v598 = (int32_t) v531;
    int32_t v599 = v596 * v598;
    int32_t v600 = v597 + v599;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v600;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v601 = v22;
    float v602 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v602 + v17 * (float) v601;
  }
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v603 = v6;
  float v604 = 0.125f * v603;
  v2[0] = v604;
  return;
}


