#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void iq2xxs_m2(const uint8_t* v1, const uint8_t* v2, float* v3, size_t v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  // weft_emitc.route_source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
  static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v4 / 256;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_i64_view
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signs_table_i64_view
  const int64_t* v8 = (const int64_t*) weft_iq2xxs_signs64;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 66;
    const uint8_t* v11 = v1 + v10;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v2 + v12;
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
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v154 = __riscv_vget_v_i8m4_i8m2(v143, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v155 = __riscv_vget_v_i8m4_i8m2(v142, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v156 = __riscv_vwmul_vv_i16m4(v154, v155, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v157 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v158 = __riscv_vwredsum_vs_i16m4_i32m1(v156, v157, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v159 = __riscv_vmv_x_s_i32m1_i32(v158);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v160 = v22;
    int32_t v161 = (int32_t) v97;
    int32_t v162 = v159 * v161;
    int32_t v163 = v160 + v162;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v163;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v164[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v165[8];
    const uint8_t* v166 = v19 + 16;
    const uint8_t v167 = v166[4];
    uint32_t v168 = (uint32_t) v167;
    const uint8_t v169 = v166[5];
    uint32_t v170 = (uint32_t) v169;
    uint32_t v171 = v170 << 8u;
    uint32_t v172 = v168 | v171;
    const uint8_t v173 = v166[6];
    uint32_t v174 = (uint32_t) v173;
    uint32_t v175 = v174 << 16u;
    uint32_t v176 = v172 | v175;
    const uint8_t v177 = v166[7];
    uint32_t v178 = (uint32_t) v177;
    uint32_t v179 = v178 << 24u;
    uint32_t v180 = v176 | v179;
    uint32_t v181 = v180 >> 28u;
    int v182 = (int) v181;
    int v183 = v182 * 2;
    int v184 = v183 + 1;
    const uint8_t v185 = v166[0];
    int v186 = (int) v185;
    int v187 = v186 * 8;
    uint16_t v188 = (uint16_t) v187;
    v164[0] = v188;
    uint32_t v189 = v180 >> 0u;
    uint32_t v190 = v189 & 127u;
    int v191 = (int) v190;
    int v192 = v191 * 8;
    uint16_t v193 = (uint16_t) v192;
    v165[0] = v193;
    const uint8_t v194 = v166[1];
    int v195 = (int) v194;
    int v196 = v195 * 8;
    uint16_t v197 = (uint16_t) v196;
    v164[1] = v197;
    uint32_t v198 = v180 >> 7u;
    uint32_t v199 = v198 & 127u;
    int v200 = (int) v199;
    int v201 = v200 * 8;
    uint16_t v202 = (uint16_t) v201;
    v165[1] = v202;
    const uint8_t v203 = v166[2];
    int v204 = (int) v203;
    int v205 = v204 * 8;
    uint16_t v206 = (uint16_t) v205;
    v164[2] = v206;
    uint32_t v207 = v180 >> 14u;
    uint32_t v208 = v207 & 127u;
    int v209 = (int) v208;
    int v210 = v209 * 8;
    uint16_t v211 = (uint16_t) v210;
    v165[2] = v211;
    const uint8_t v212 = v166[3];
    int v213 = (int) v212;
    int v214 = v213 * 8;
    uint16_t v215 = (uint16_t) v214;
    v164[3] = v215;
    uint32_t v216 = v180 >> 21u;
    uint32_t v217 = v216 & 127u;
    int v218 = (int) v217;
    int v219 = v218 * 8;
    uint16_t v220 = (uint16_t) v219;
    v165[3] = v220;
    const uint8_t* v221 = v19 + 24;
    const uint8_t v222 = v221[4];
    uint32_t v223 = (uint32_t) v222;
    const uint8_t v224 = v221[5];
    uint32_t v225 = (uint32_t) v224;
    uint32_t v226 = v225 << 8u;
    uint32_t v227 = v223 | v226;
    const uint8_t v228 = v221[6];
    uint32_t v229 = (uint32_t) v228;
    uint32_t v230 = v229 << 16u;
    uint32_t v231 = v227 | v230;
    const uint8_t v232 = v221[7];
    uint32_t v233 = (uint32_t) v232;
    uint32_t v234 = v233 << 24u;
    uint32_t v235 = v231 | v234;
    uint32_t v236 = v235 >> 28u;
    int v237 = (int) v236;
    int v238 = v237 * 2;
    int v239 = v238 + 1;
    const uint8_t v240 = v221[0];
    int v241 = (int) v240;
    int v242 = v241 * 8;
    uint16_t v243 = (uint16_t) v242;
    v164[4] = v243;
    uint32_t v244 = v235 >> 0u;
    uint32_t v245 = v244 & 127u;
    int v246 = (int) v245;
    int v247 = v246 * 8;
    uint16_t v248 = (uint16_t) v247;
    v165[4] = v248;
    const uint8_t v249 = v221[1];
    int v250 = (int) v249;
    int v251 = v250 * 8;
    uint16_t v252 = (uint16_t) v251;
    v164[5] = v252;
    uint32_t v253 = v235 >> 7u;
    uint32_t v254 = v253 & 127u;
    int v255 = (int) v254;
    int v256 = v255 * 8;
    uint16_t v257 = (uint16_t) v256;
    v165[5] = v257;
    const uint8_t v258 = v221[2];
    int v259 = (int) v258;
    int v260 = v259 * 8;
    uint16_t v261 = (uint16_t) v260;
    v164[6] = v261;
    uint32_t v262 = v235 >> 14u;
    uint32_t v263 = v262 & 127u;
    int v264 = (int) v263;
    int v265 = v264 * 8;
    uint16_t v266 = (uint16_t) v265;
    v165[6] = v266;
    const uint8_t v267 = v221[3];
    int v268 = (int) v267;
    int v269 = v268 * 8;
    uint16_t v270 = (uint16_t) v269;
    v164[7] = v270;
    uint32_t v271 = v235 >> 21u;
    uint32_t v272 = v271 & 127u;
    int v273 = (int) v272;
    int v274 = v273 * 8;
    uint16_t v275 = (uint16_t) v274;
    v165[7] = v275;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v276 = &v164[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v277 = __riscv_vle16_v_u16m1(v276, 8);
    uint16_t* v278 = &v165[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v279 = __riscv_vle16_v_u16m1(v278, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v280 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v277, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v281 = __riscv_vreinterpret_v_i64m4_i8m4(v280);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v282 = __riscv_vluxei16_v_i64m4(v8, v279, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v283 = __riscv_vreinterpret_v_i64m4_i8m4(v282);
    const int8_t* v284 = v21 + 64;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v285 = __riscv_vle8_v_i8m4(v284, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v286 = __riscv_vmul_vv_i8m4(v281, v283, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v287 = __riscv_vget_v_i8m4_i8m2(v286, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v288 = __riscv_vget_v_i8m4_i8m2(v285, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v289 = __riscv_vwmul_vv_i16m4(v287, v288, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v290 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v291 = __riscv_vwredsum_vs_i16m4_i32m1(v289, v290, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v292 = __riscv_vmv_x_s_i32m1_i32(v291);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v293 = v22;
    int32_t v294 = (int32_t) v184;
    int32_t v295 = v292 * v294;
    int32_t v296 = v293 + v295;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v296;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v297 = __riscv_vget_v_i8m4_i8m2(v286, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v298 = __riscv_vget_v_i8m4_i8m2(v285, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v299 = __riscv_vwmul_vv_i16m4(v297, v298, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v300 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v301 = __riscv_vwredsum_vs_i16m4_i32m1(v299, v300, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v302 = __riscv_vmv_x_s_i32m1_i32(v301);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v303 = v22;
    int32_t v304 = (int32_t) v239;
    int32_t v305 = v302 * v304;
    int32_t v306 = v303 + v305;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v306;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v307[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v308[8];
    const uint8_t* v309 = v19 + 32;
    const uint8_t v310 = v309[4];
    uint32_t v311 = (uint32_t) v310;
    const uint8_t v312 = v309[5];
    uint32_t v313 = (uint32_t) v312;
    uint32_t v314 = v313 << 8u;
    uint32_t v315 = v311 | v314;
    const uint8_t v316 = v309[6];
    uint32_t v317 = (uint32_t) v316;
    uint32_t v318 = v317 << 16u;
    uint32_t v319 = v315 | v318;
    const uint8_t v320 = v309[7];
    uint32_t v321 = (uint32_t) v320;
    uint32_t v322 = v321 << 24u;
    uint32_t v323 = v319 | v322;
    uint32_t v324 = v323 >> 28u;
    int v325 = (int) v324;
    int v326 = v325 * 2;
    int v327 = v326 + 1;
    const uint8_t v328 = v309[0];
    int v329 = (int) v328;
    int v330 = v329 * 8;
    uint16_t v331 = (uint16_t) v330;
    v307[0] = v331;
    uint32_t v332 = v323 >> 0u;
    uint32_t v333 = v332 & 127u;
    int v334 = (int) v333;
    int v335 = v334 * 8;
    uint16_t v336 = (uint16_t) v335;
    v308[0] = v336;
    const uint8_t v337 = v309[1];
    int v338 = (int) v337;
    int v339 = v338 * 8;
    uint16_t v340 = (uint16_t) v339;
    v307[1] = v340;
    uint32_t v341 = v323 >> 7u;
    uint32_t v342 = v341 & 127u;
    int v343 = (int) v342;
    int v344 = v343 * 8;
    uint16_t v345 = (uint16_t) v344;
    v308[1] = v345;
    const uint8_t v346 = v309[2];
    int v347 = (int) v346;
    int v348 = v347 * 8;
    uint16_t v349 = (uint16_t) v348;
    v307[2] = v349;
    uint32_t v350 = v323 >> 14u;
    uint32_t v351 = v350 & 127u;
    int v352 = (int) v351;
    int v353 = v352 * 8;
    uint16_t v354 = (uint16_t) v353;
    v308[2] = v354;
    const uint8_t v355 = v309[3];
    int v356 = (int) v355;
    int v357 = v356 * 8;
    uint16_t v358 = (uint16_t) v357;
    v307[3] = v358;
    uint32_t v359 = v323 >> 21u;
    uint32_t v360 = v359 & 127u;
    int v361 = (int) v360;
    int v362 = v361 * 8;
    uint16_t v363 = (uint16_t) v362;
    v308[3] = v363;
    const uint8_t* v364 = v19 + 40;
    const uint8_t v365 = v364[4];
    uint32_t v366 = (uint32_t) v365;
    const uint8_t v367 = v364[5];
    uint32_t v368 = (uint32_t) v367;
    uint32_t v369 = v368 << 8u;
    uint32_t v370 = v366 | v369;
    const uint8_t v371 = v364[6];
    uint32_t v372 = (uint32_t) v371;
    uint32_t v373 = v372 << 16u;
    uint32_t v374 = v370 | v373;
    const uint8_t v375 = v364[7];
    uint32_t v376 = (uint32_t) v375;
    uint32_t v377 = v376 << 24u;
    uint32_t v378 = v374 | v377;
    uint32_t v379 = v378 >> 28u;
    int v380 = (int) v379;
    int v381 = v380 * 2;
    int v382 = v381 + 1;
    const uint8_t v383 = v364[0];
    int v384 = (int) v383;
    int v385 = v384 * 8;
    uint16_t v386 = (uint16_t) v385;
    v307[4] = v386;
    uint32_t v387 = v378 >> 0u;
    uint32_t v388 = v387 & 127u;
    int v389 = (int) v388;
    int v390 = v389 * 8;
    uint16_t v391 = (uint16_t) v390;
    v308[4] = v391;
    const uint8_t v392 = v364[1];
    int v393 = (int) v392;
    int v394 = v393 * 8;
    uint16_t v395 = (uint16_t) v394;
    v307[5] = v395;
    uint32_t v396 = v378 >> 7u;
    uint32_t v397 = v396 & 127u;
    int v398 = (int) v397;
    int v399 = v398 * 8;
    uint16_t v400 = (uint16_t) v399;
    v308[5] = v400;
    const uint8_t v401 = v364[2];
    int v402 = (int) v401;
    int v403 = v402 * 8;
    uint16_t v404 = (uint16_t) v403;
    v307[6] = v404;
    uint32_t v405 = v378 >> 14u;
    uint32_t v406 = v405 & 127u;
    int v407 = (int) v406;
    int v408 = v407 * 8;
    uint16_t v409 = (uint16_t) v408;
    v308[6] = v409;
    const uint8_t v410 = v364[3];
    int v411 = (int) v410;
    int v412 = v411 * 8;
    uint16_t v413 = (uint16_t) v412;
    v307[7] = v413;
    uint32_t v414 = v378 >> 21u;
    uint32_t v415 = v414 & 127u;
    int v416 = (int) v415;
    int v417 = v416 * 8;
    uint16_t v418 = (uint16_t) v417;
    v308[7] = v418;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v419 = &v307[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v420 = __riscv_vle16_v_u16m1(v419, 8);
    uint16_t* v421 = &v308[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v422 = __riscv_vle16_v_u16m1(v421, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v423 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v420, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v424 = __riscv_vreinterpret_v_i64m4_i8m4(v423);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v425 = __riscv_vluxei16_v_i64m4(v8, v422, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v426 = __riscv_vreinterpret_v_i64m4_i8m4(v425);
    const int8_t* v427 = v21 + 128;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v428 = __riscv_vle8_v_i8m4(v427, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v429 = __riscv_vmul_vv_i8m4(v424, v426, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v430 = __riscv_vget_v_i8m4_i8m2(v429, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v431 = __riscv_vget_v_i8m4_i8m2(v428, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v432 = __riscv_vwmul_vv_i16m4(v430, v431, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v433 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v434 = __riscv_vwredsum_vs_i16m4_i32m1(v432, v433, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v435 = __riscv_vmv_x_s_i32m1_i32(v434);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v436 = v22;
    int32_t v437 = (int32_t) v327;
    int32_t v438 = v435 * v437;
    int32_t v439 = v436 + v438;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v439;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v440 = __riscv_vget_v_i8m4_i8m2(v429, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v441 = __riscv_vget_v_i8m4_i8m2(v428, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v442 = __riscv_vwmul_vv_i16m4(v440, v441, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v443 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v444 = __riscv_vwredsum_vs_i16m4_i32m1(v442, v443, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v445 = __riscv_vmv_x_s_i32m1_i32(v444);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v446 = v22;
    int32_t v447 = (int32_t) v382;
    int32_t v448 = v445 * v447;
    int32_t v449 = v446 + v448;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v449;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_aux_scale
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v450[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v451[8];
    const uint8_t* v452 = v19 + 48;
    const uint8_t v453 = v452[4];
    uint32_t v454 = (uint32_t) v453;
    const uint8_t v455 = v452[5];
    uint32_t v456 = (uint32_t) v455;
    uint32_t v457 = v456 << 8u;
    uint32_t v458 = v454 | v457;
    const uint8_t v459 = v452[6];
    uint32_t v460 = (uint32_t) v459;
    uint32_t v461 = v460 << 16u;
    uint32_t v462 = v458 | v461;
    const uint8_t v463 = v452[7];
    uint32_t v464 = (uint32_t) v463;
    uint32_t v465 = v464 << 24u;
    uint32_t v466 = v462 | v465;
    uint32_t v467 = v466 >> 28u;
    int v468 = (int) v467;
    int v469 = v468 * 2;
    int v470 = v469 + 1;
    const uint8_t v471 = v452[0];
    int v472 = (int) v471;
    int v473 = v472 * 8;
    uint16_t v474 = (uint16_t) v473;
    v450[0] = v474;
    uint32_t v475 = v466 >> 0u;
    uint32_t v476 = v475 & 127u;
    int v477 = (int) v476;
    int v478 = v477 * 8;
    uint16_t v479 = (uint16_t) v478;
    v451[0] = v479;
    const uint8_t v480 = v452[1];
    int v481 = (int) v480;
    int v482 = v481 * 8;
    uint16_t v483 = (uint16_t) v482;
    v450[1] = v483;
    uint32_t v484 = v466 >> 7u;
    uint32_t v485 = v484 & 127u;
    int v486 = (int) v485;
    int v487 = v486 * 8;
    uint16_t v488 = (uint16_t) v487;
    v451[1] = v488;
    const uint8_t v489 = v452[2];
    int v490 = (int) v489;
    int v491 = v490 * 8;
    uint16_t v492 = (uint16_t) v491;
    v450[2] = v492;
    uint32_t v493 = v466 >> 14u;
    uint32_t v494 = v493 & 127u;
    int v495 = (int) v494;
    int v496 = v495 * 8;
    uint16_t v497 = (uint16_t) v496;
    v451[2] = v497;
    const uint8_t v498 = v452[3];
    int v499 = (int) v498;
    int v500 = v499 * 8;
    uint16_t v501 = (uint16_t) v500;
    v450[3] = v501;
    uint32_t v502 = v466 >> 21u;
    uint32_t v503 = v502 & 127u;
    int v504 = (int) v503;
    int v505 = v504 * 8;
    uint16_t v506 = (uint16_t) v505;
    v451[3] = v506;
    const uint8_t* v507 = v19 + 56;
    const uint8_t v508 = v507[4];
    uint32_t v509 = (uint32_t) v508;
    const uint8_t v510 = v507[5];
    uint32_t v511 = (uint32_t) v510;
    uint32_t v512 = v511 << 8u;
    uint32_t v513 = v509 | v512;
    const uint8_t v514 = v507[6];
    uint32_t v515 = (uint32_t) v514;
    uint32_t v516 = v515 << 16u;
    uint32_t v517 = v513 | v516;
    const uint8_t v518 = v507[7];
    uint32_t v519 = (uint32_t) v518;
    uint32_t v520 = v519 << 24u;
    uint32_t v521 = v517 | v520;
    uint32_t v522 = v521 >> 28u;
    int v523 = (int) v522;
    int v524 = v523 * 2;
    int v525 = v524 + 1;
    const uint8_t v526 = v507[0];
    int v527 = (int) v526;
    int v528 = v527 * 8;
    uint16_t v529 = (uint16_t) v528;
    v450[4] = v529;
    uint32_t v530 = v521 >> 0u;
    uint32_t v531 = v530 & 127u;
    int v532 = (int) v531;
    int v533 = v532 * 8;
    uint16_t v534 = (uint16_t) v533;
    v451[4] = v534;
    const uint8_t v535 = v507[1];
    int v536 = (int) v535;
    int v537 = v536 * 8;
    uint16_t v538 = (uint16_t) v537;
    v450[5] = v538;
    uint32_t v539 = v521 >> 7u;
    uint32_t v540 = v539 & 127u;
    int v541 = (int) v540;
    int v542 = v541 * 8;
    uint16_t v543 = (uint16_t) v542;
    v451[5] = v543;
    const uint8_t v544 = v507[2];
    int v545 = (int) v544;
    int v546 = v545 * 8;
    uint16_t v547 = (uint16_t) v546;
    v450[6] = v547;
    uint32_t v548 = v521 >> 14u;
    uint32_t v549 = v548 & 127u;
    int v550 = (int) v549;
    int v551 = v550 * 8;
    uint16_t v552 = (uint16_t) v551;
    v451[6] = v552;
    const uint8_t v553 = v507[3];
    int v554 = (int) v553;
    int v555 = v554 * 8;
    uint16_t v556 = (uint16_t) v555;
    v450[7] = v556;
    uint32_t v557 = v521 >> 21u;
    uint32_t v558 = v557 & 127u;
    int v559 = (int) v558;
    int v560 = v559 * 8;
    uint16_t v561 = (uint16_t) v560;
    v451[7] = v561;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    uint16_t* v562 = &v450[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v563 = __riscv_vle16_v_u16m1(v562, 8);
    uint16_t* v564 = &v451[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v565 = __riscv_vle16_v_u16m1(v564, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v566 = __riscv_vluxei16_v_i64m4(weft_iq2xxs_grid, v563, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v567 = __riscv_vreinterpret_v_i64m4_i8m4(v566);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v568 = __riscv_vluxei16_v_i64m4(v8, v565, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v569 = __riscv_vreinterpret_v_i64m4_i8m4(v568);
    const int8_t* v570 = v21 + 192;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v571 = __riscv_vle8_v_i8m4(v570, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v572 = __riscv_vmul_vv_i8m4(v567, v569, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v573 = __riscv_vget_v_i8m4_i8m2(v572, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v574 = __riscv_vget_v_i8m4_i8m2(v571, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v575 = __riscv_vwmul_vv_i16m4(v573, v574, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v576 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v577 = __riscv_vwredsum_vs_i16m4_i32m1(v575, v576, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v578 = __riscv_vmv_x_s_i32m1_i32(v577);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v579 = v22;
    int32_t v580 = (int32_t) v470;
    int32_t v581 = v578 * v580;
    int32_t v582 = v579 + v581;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v582;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v583 = __riscv_vget_v_i8m4_i8m2(v572, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m2
    vint8m2_t v584 = __riscv_vget_v_i8m4_i8m2(v571, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v585 = __riscv_vwmul_vv_i16m4(v583, v584, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v586 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v587 = __riscv_vwredsum_vs_i16m4_i32m1(v585, v586, 32);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v588 = __riscv_vmv_x_s_i32m1_i32(v587);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v589 = v22;
    int32_t v590 = (int32_t) v525;
    int32_t v591 = v588 * v590;
    int32_t v592 = v589 + v591;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v22 = v592;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v593 = v22;
    float v594 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v594 + v17 * (float) v593;
  }
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v595 = v6;
  float v596 = 0.125f * v595;
  v3[0] = v596;
  return;
}


