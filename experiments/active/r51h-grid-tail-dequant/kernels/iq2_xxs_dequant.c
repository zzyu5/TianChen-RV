#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq2_xxs_kernel_dequant_iq2_xxs(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
  static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_table_view
  const uint8_t* v6 = (const uint8_t*) weft_iq2xxs_grid;
  for (size_t v7 = 0; v7 < v5; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v8 = v7 * 66;
    const uint8_t* v9 = v2 + v8;
    size_t v10 = v7 * 256;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v9);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v14 = v9[9];
    uint32_t v15 = (uint32_t) v14;
    uint32_t v16 = v15 << 24u;
    const uint8_t v17 = v9[8];
    uint32_t v18 = (uint32_t) v17;
    uint32_t v19 = v18 << 16u;
    const uint8_t v20 = v9[7];
    uint32_t v21 = (uint32_t) v20;
    uint32_t v22 = v21 << 8u;
    const uint8_t v23 = v9[6];
    uint32_t v24 = (uint32_t) v23;
    uint32_t v25 = v24 | v22;
    uint32_t v26 = v25 | v19;
    uint32_t v27 = v26 | v16;
    uint32_t v28 = v27 >> 28u;
    float v29 = (float) v28;
    float v30 = 0.5f + v29;
    float v31 = v13 * v30;
    float v32 = v31 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v33 = v9[2];
    uint32_t v34 = (uint32_t) v33;
    size_t v35 = (size_t) v34;
    size_t v36 = v35 * 8;
    const uint8_t* v37 = v6 + v36;
    const int8_t* v38 = (const int8_t*) v37;
    uint32_t v39 = v27 & 127u;
    size_t v40 = (size_t) v39;
    size_t v41 = v40 * 8;
    const int8_t* v42 = weft_iq2xxs_signs64 + v41;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v43 = __riscv_vle8_v_i8mf2(v38, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v44 = __riscv_vle8_v_i8mf2(v42, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v45 = __riscv_vmul_vv_i8mf2(v43, v44, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v46 = __riscv_vsext_vf4_i32m2(v45, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v47 = __riscv_vfcvt_f_x_v_f32m2(v46, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v48 = __riscv_vfmul_vf_f32m2(v47, v32, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v12, v48, 8);
    const uint8_t v49 = v9[3];
    uint32_t v50 = (uint32_t) v49;
    size_t v51 = (size_t) v50;
    size_t v52 = v51 * 8;
    const uint8_t* v53 = v6 + v52;
    const int8_t* v54 = (const int8_t*) v53;
    uint32_t v55 = v27 >> 7u;
    uint32_t v56 = v55 & 127u;
    size_t v57 = (size_t) v56;
    size_t v58 = v57 * 8;
    const int8_t* v59 = weft_iq2xxs_signs64 + v58;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v60 = __riscv_vle8_v_i8mf2(v54, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v61 = __riscv_vle8_v_i8mf2(v59, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v62 = __riscv_vmul_vv_i8mf2(v60, v61, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v63 = __riscv_vsext_vf4_i32m2(v62, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v64 = __riscv_vfcvt_f_x_v_f32m2(v63, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v65 = __riscv_vfmul_vf_f32m2(v64, v32, 8);
    float* v66 = v12 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v66, v65, 8);
    const uint8_t v67 = v9[4];
    uint32_t v68 = (uint32_t) v67;
    size_t v69 = (size_t) v68;
    size_t v70 = v69 * 8;
    const uint8_t* v71 = v6 + v70;
    const int8_t* v72 = (const int8_t*) v71;
    uint32_t v73 = v27 >> 14u;
    uint32_t v74 = v73 & 127u;
    size_t v75 = (size_t) v74;
    size_t v76 = v75 * 8;
    const int8_t* v77 = weft_iq2xxs_signs64 + v76;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v78 = __riscv_vle8_v_i8mf2(v72, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v79 = __riscv_vle8_v_i8mf2(v77, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v80 = __riscv_vmul_vv_i8mf2(v78, v79, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v81 = __riscv_vsext_vf4_i32m2(v80, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v82 = __riscv_vfcvt_f_x_v_f32m2(v81, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v83 = __riscv_vfmul_vf_f32m2(v82, v32, 8);
    float* v84 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v84, v83, 8);
    const uint8_t v85 = v9[5];
    uint32_t v86 = (uint32_t) v85;
    size_t v87 = (size_t) v86;
    size_t v88 = v87 * 8;
    const uint8_t* v89 = v6 + v88;
    const int8_t* v90 = (const int8_t*) v89;
    uint32_t v91 = v27 >> 21u;
    uint32_t v92 = v91 & 127u;
    size_t v93 = (size_t) v92;
    size_t v94 = v93 * 8;
    const int8_t* v95 = weft_iq2xxs_signs64 + v94;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v96 = __riscv_vle8_v_i8mf2(v90, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v97 = __riscv_vle8_v_i8mf2(v95, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v98 = __riscv_vmul_vv_i8mf2(v96, v97, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v99 = __riscv_vsext_vf4_i32m2(v98, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v100 = __riscv_vfcvt_f_x_v_f32m2(v99, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v101 = __riscv_vfmul_vf_f32m2(v100, v32, 8);
    float* v102 = v12 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v102, v101, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v103 = v9[17];
    uint32_t v104 = (uint32_t) v103;
    uint32_t v105 = v104 << 24u;
    const uint8_t v106 = v9[16];
    uint32_t v107 = (uint32_t) v106;
    uint32_t v108 = v107 << 16u;
    const uint8_t v109 = v9[15];
    uint32_t v110 = (uint32_t) v109;
    uint32_t v111 = v110 << 8u;
    const uint8_t v112 = v9[14];
    uint32_t v113 = (uint32_t) v112;
    uint32_t v114 = v113 | v111;
    uint32_t v115 = v114 | v108;
    uint32_t v116 = v115 | v105;
    uint32_t v117 = v116 >> 28u;
    float v118 = (float) v117;
    float v119 = 0.5f + v118;
    float v120 = v13 * v119;
    float v121 = v120 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v122 = v9[10];
    uint32_t v123 = (uint32_t) v122;
    size_t v124 = (size_t) v123;
    size_t v125 = v124 * 8;
    const uint8_t* v126 = v6 + v125;
    const int8_t* v127 = (const int8_t*) v126;
    uint32_t v128 = v116 & 127u;
    size_t v129 = (size_t) v128;
    size_t v130 = v129 * 8;
    const int8_t* v131 = weft_iq2xxs_signs64 + v130;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v132 = __riscv_vle8_v_i8mf2(v127, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v133 = __riscv_vle8_v_i8mf2(v131, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v134 = __riscv_vmul_vv_i8mf2(v132, v133, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v135 = __riscv_vsext_vf4_i32m2(v134, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v136 = __riscv_vfcvt_f_x_v_f32m2(v135, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v137 = __riscv_vfmul_vf_f32m2(v136, v121, 8);
    float* v138 = v12 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v138, v137, 8);
    const uint8_t v139 = v9[11];
    uint32_t v140 = (uint32_t) v139;
    size_t v141 = (size_t) v140;
    size_t v142 = v141 * 8;
    const uint8_t* v143 = v6 + v142;
    const int8_t* v144 = (const int8_t*) v143;
    uint32_t v145 = v116 >> 7u;
    uint32_t v146 = v145 & 127u;
    size_t v147 = (size_t) v146;
    size_t v148 = v147 * 8;
    const int8_t* v149 = weft_iq2xxs_signs64 + v148;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v150 = __riscv_vle8_v_i8mf2(v144, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v151 = __riscv_vle8_v_i8mf2(v149, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v152 = __riscv_vmul_vv_i8mf2(v150, v151, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v153 = __riscv_vsext_vf4_i32m2(v152, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v154 = __riscv_vfcvt_f_x_v_f32m2(v153, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v155 = __riscv_vfmul_vf_f32m2(v154, v121, 8);
    float* v156 = v12 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v156, v155, 8);
    const uint8_t v157 = v9[12];
    uint32_t v158 = (uint32_t) v157;
    size_t v159 = (size_t) v158;
    size_t v160 = v159 * 8;
    const uint8_t* v161 = v6 + v160;
    const int8_t* v162 = (const int8_t*) v161;
    uint32_t v163 = v116 >> 14u;
    uint32_t v164 = v163 & 127u;
    size_t v165 = (size_t) v164;
    size_t v166 = v165 * 8;
    const int8_t* v167 = weft_iq2xxs_signs64 + v166;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v168 = __riscv_vle8_v_i8mf2(v162, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v169 = __riscv_vle8_v_i8mf2(v167, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v170 = __riscv_vmul_vv_i8mf2(v168, v169, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v171 = __riscv_vsext_vf4_i32m2(v170, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v172 = __riscv_vfcvt_f_x_v_f32m2(v171, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v173 = __riscv_vfmul_vf_f32m2(v172, v121, 8);
    float* v174 = v12 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v174, v173, 8);
    const uint8_t v175 = v9[13];
    uint32_t v176 = (uint32_t) v175;
    size_t v177 = (size_t) v176;
    size_t v178 = v177 * 8;
    const uint8_t* v179 = v6 + v178;
    const int8_t* v180 = (const int8_t*) v179;
    uint32_t v181 = v116 >> 21u;
    uint32_t v182 = v181 & 127u;
    size_t v183 = (size_t) v182;
    size_t v184 = v183 * 8;
    const int8_t* v185 = weft_iq2xxs_signs64 + v184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v186 = __riscv_vle8_v_i8mf2(v180, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v187 = __riscv_vle8_v_i8mf2(v185, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v188 = __riscv_vmul_vv_i8mf2(v186, v187, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v189 = __riscv_vsext_vf4_i32m2(v188, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v190 = __riscv_vfcvt_f_x_v_f32m2(v189, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v191 = __riscv_vfmul_vf_f32m2(v190, v121, 8);
    float* v192 = v12 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v192, v191, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v193 = v9[25];
    uint32_t v194 = (uint32_t) v193;
    uint32_t v195 = v194 << 24u;
    const uint8_t v196 = v9[24];
    uint32_t v197 = (uint32_t) v196;
    uint32_t v198 = v197 << 16u;
    const uint8_t v199 = v9[23];
    uint32_t v200 = (uint32_t) v199;
    uint32_t v201 = v200 << 8u;
    const uint8_t v202 = v9[22];
    uint32_t v203 = (uint32_t) v202;
    uint32_t v204 = v203 | v201;
    uint32_t v205 = v204 | v198;
    uint32_t v206 = v205 | v195;
    uint32_t v207 = v206 >> 28u;
    float v208 = (float) v207;
    float v209 = 0.5f + v208;
    float v210 = v13 * v209;
    float v211 = v210 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v212 = v9[18];
    uint32_t v213 = (uint32_t) v212;
    size_t v214 = (size_t) v213;
    size_t v215 = v214 * 8;
    const uint8_t* v216 = v6 + v215;
    const int8_t* v217 = (const int8_t*) v216;
    uint32_t v218 = v206 & 127u;
    size_t v219 = (size_t) v218;
    size_t v220 = v219 * 8;
    const int8_t* v221 = weft_iq2xxs_signs64 + v220;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v222 = __riscv_vle8_v_i8mf2(v217, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v223 = __riscv_vle8_v_i8mf2(v221, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v224 = __riscv_vmul_vv_i8mf2(v222, v223, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v225 = __riscv_vsext_vf4_i32m2(v224, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v226 = __riscv_vfcvt_f_x_v_f32m2(v225, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v227 = __riscv_vfmul_vf_f32m2(v226, v211, 8);
    float* v228 = v12 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v228, v227, 8);
    const uint8_t v229 = v9[19];
    uint32_t v230 = (uint32_t) v229;
    size_t v231 = (size_t) v230;
    size_t v232 = v231 * 8;
    const uint8_t* v233 = v6 + v232;
    const int8_t* v234 = (const int8_t*) v233;
    uint32_t v235 = v206 >> 7u;
    uint32_t v236 = v235 & 127u;
    size_t v237 = (size_t) v236;
    size_t v238 = v237 * 8;
    const int8_t* v239 = weft_iq2xxs_signs64 + v238;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v240 = __riscv_vle8_v_i8mf2(v234, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v241 = __riscv_vle8_v_i8mf2(v239, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v242 = __riscv_vmul_vv_i8mf2(v240, v241, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v243 = __riscv_vsext_vf4_i32m2(v242, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v244 = __riscv_vfcvt_f_x_v_f32m2(v243, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v245 = __riscv_vfmul_vf_f32m2(v244, v211, 8);
    float* v246 = v12 + 72;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v246, v245, 8);
    const uint8_t v247 = v9[20];
    uint32_t v248 = (uint32_t) v247;
    size_t v249 = (size_t) v248;
    size_t v250 = v249 * 8;
    const uint8_t* v251 = v6 + v250;
    const int8_t* v252 = (const int8_t*) v251;
    uint32_t v253 = v206 >> 14u;
    uint32_t v254 = v253 & 127u;
    size_t v255 = (size_t) v254;
    size_t v256 = v255 * 8;
    const int8_t* v257 = weft_iq2xxs_signs64 + v256;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v258 = __riscv_vle8_v_i8mf2(v252, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v259 = __riscv_vle8_v_i8mf2(v257, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v260 = __riscv_vmul_vv_i8mf2(v258, v259, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v261 = __riscv_vsext_vf4_i32m2(v260, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v262 = __riscv_vfcvt_f_x_v_f32m2(v261, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v263 = __riscv_vfmul_vf_f32m2(v262, v211, 8);
    float* v264 = v12 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v264, v263, 8);
    const uint8_t v265 = v9[21];
    uint32_t v266 = (uint32_t) v265;
    size_t v267 = (size_t) v266;
    size_t v268 = v267 * 8;
    const uint8_t* v269 = v6 + v268;
    const int8_t* v270 = (const int8_t*) v269;
    uint32_t v271 = v206 >> 21u;
    uint32_t v272 = v271 & 127u;
    size_t v273 = (size_t) v272;
    size_t v274 = v273 * 8;
    const int8_t* v275 = weft_iq2xxs_signs64 + v274;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v276 = __riscv_vle8_v_i8mf2(v270, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v277 = __riscv_vle8_v_i8mf2(v275, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v278 = __riscv_vmul_vv_i8mf2(v276, v277, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v279 = __riscv_vsext_vf4_i32m2(v278, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v280 = __riscv_vfcvt_f_x_v_f32m2(v279, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v281 = __riscv_vfmul_vf_f32m2(v280, v211, 8);
    float* v282 = v12 + 88;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v282, v281, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v283 = v9[33];
    uint32_t v284 = (uint32_t) v283;
    uint32_t v285 = v284 << 24u;
    const uint8_t v286 = v9[32];
    uint32_t v287 = (uint32_t) v286;
    uint32_t v288 = v287 << 16u;
    const uint8_t v289 = v9[31];
    uint32_t v290 = (uint32_t) v289;
    uint32_t v291 = v290 << 8u;
    const uint8_t v292 = v9[30];
    uint32_t v293 = (uint32_t) v292;
    uint32_t v294 = v293 | v291;
    uint32_t v295 = v294 | v288;
    uint32_t v296 = v295 | v285;
    uint32_t v297 = v296 >> 28u;
    float v298 = (float) v297;
    float v299 = 0.5f + v298;
    float v300 = v13 * v299;
    float v301 = v300 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v302 = v9[26];
    uint32_t v303 = (uint32_t) v302;
    size_t v304 = (size_t) v303;
    size_t v305 = v304 * 8;
    const uint8_t* v306 = v6 + v305;
    const int8_t* v307 = (const int8_t*) v306;
    uint32_t v308 = v296 & 127u;
    size_t v309 = (size_t) v308;
    size_t v310 = v309 * 8;
    const int8_t* v311 = weft_iq2xxs_signs64 + v310;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v312 = __riscv_vle8_v_i8mf2(v307, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v313 = __riscv_vle8_v_i8mf2(v311, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v314 = __riscv_vmul_vv_i8mf2(v312, v313, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v315 = __riscv_vsext_vf4_i32m2(v314, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v316 = __riscv_vfcvt_f_x_v_f32m2(v315, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v317 = __riscv_vfmul_vf_f32m2(v316, v301, 8);
    float* v318 = v12 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v318, v317, 8);
    const uint8_t v319 = v9[27];
    uint32_t v320 = (uint32_t) v319;
    size_t v321 = (size_t) v320;
    size_t v322 = v321 * 8;
    const uint8_t* v323 = v6 + v322;
    const int8_t* v324 = (const int8_t*) v323;
    uint32_t v325 = v296 >> 7u;
    uint32_t v326 = v325 & 127u;
    size_t v327 = (size_t) v326;
    size_t v328 = v327 * 8;
    const int8_t* v329 = weft_iq2xxs_signs64 + v328;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v330 = __riscv_vle8_v_i8mf2(v324, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v331 = __riscv_vle8_v_i8mf2(v329, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v332 = __riscv_vmul_vv_i8mf2(v330, v331, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v333 = __riscv_vsext_vf4_i32m2(v332, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v334 = __riscv_vfcvt_f_x_v_f32m2(v333, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v335 = __riscv_vfmul_vf_f32m2(v334, v301, 8);
    float* v336 = v12 + 104;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v336, v335, 8);
    const uint8_t v337 = v9[28];
    uint32_t v338 = (uint32_t) v337;
    size_t v339 = (size_t) v338;
    size_t v340 = v339 * 8;
    const uint8_t* v341 = v6 + v340;
    const int8_t* v342 = (const int8_t*) v341;
    uint32_t v343 = v296 >> 14u;
    uint32_t v344 = v343 & 127u;
    size_t v345 = (size_t) v344;
    size_t v346 = v345 * 8;
    const int8_t* v347 = weft_iq2xxs_signs64 + v346;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v348 = __riscv_vle8_v_i8mf2(v342, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v349 = __riscv_vle8_v_i8mf2(v347, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v350 = __riscv_vmul_vv_i8mf2(v348, v349, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v351 = __riscv_vsext_vf4_i32m2(v350, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v352 = __riscv_vfcvt_f_x_v_f32m2(v351, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v353 = __riscv_vfmul_vf_f32m2(v352, v301, 8);
    float* v354 = v12 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v354, v353, 8);
    const uint8_t v355 = v9[29];
    uint32_t v356 = (uint32_t) v355;
    size_t v357 = (size_t) v356;
    size_t v358 = v357 * 8;
    const uint8_t* v359 = v6 + v358;
    const int8_t* v360 = (const int8_t*) v359;
    uint32_t v361 = v296 >> 21u;
    uint32_t v362 = v361 & 127u;
    size_t v363 = (size_t) v362;
    size_t v364 = v363 * 8;
    const int8_t* v365 = weft_iq2xxs_signs64 + v364;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v366 = __riscv_vle8_v_i8mf2(v360, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v367 = __riscv_vle8_v_i8mf2(v365, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v368 = __riscv_vmul_vv_i8mf2(v366, v367, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v369 = __riscv_vsext_vf4_i32m2(v368, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v370 = __riscv_vfcvt_f_x_v_f32m2(v369, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v371 = __riscv_vfmul_vf_f32m2(v370, v301, 8);
    float* v372 = v12 + 120;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v372, v371, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v373 = v9[41];
    uint32_t v374 = (uint32_t) v373;
    uint32_t v375 = v374 << 24u;
    const uint8_t v376 = v9[40];
    uint32_t v377 = (uint32_t) v376;
    uint32_t v378 = v377 << 16u;
    const uint8_t v379 = v9[39];
    uint32_t v380 = (uint32_t) v379;
    uint32_t v381 = v380 << 8u;
    const uint8_t v382 = v9[38];
    uint32_t v383 = (uint32_t) v382;
    uint32_t v384 = v383 | v381;
    uint32_t v385 = v384 | v378;
    uint32_t v386 = v385 | v375;
    uint32_t v387 = v386 >> 28u;
    float v388 = (float) v387;
    float v389 = 0.5f + v388;
    float v390 = v13 * v389;
    float v391 = v390 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v392 = v9[34];
    uint32_t v393 = (uint32_t) v392;
    size_t v394 = (size_t) v393;
    size_t v395 = v394 * 8;
    const uint8_t* v396 = v6 + v395;
    const int8_t* v397 = (const int8_t*) v396;
    uint32_t v398 = v386 & 127u;
    size_t v399 = (size_t) v398;
    size_t v400 = v399 * 8;
    const int8_t* v401 = weft_iq2xxs_signs64 + v400;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v402 = __riscv_vle8_v_i8mf2(v397, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v403 = __riscv_vle8_v_i8mf2(v401, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v404 = __riscv_vmul_vv_i8mf2(v402, v403, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v405 = __riscv_vsext_vf4_i32m2(v404, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v406 = __riscv_vfcvt_f_x_v_f32m2(v405, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v407 = __riscv_vfmul_vf_f32m2(v406, v391, 8);
    float* v408 = v12 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v408, v407, 8);
    const uint8_t v409 = v9[35];
    uint32_t v410 = (uint32_t) v409;
    size_t v411 = (size_t) v410;
    size_t v412 = v411 * 8;
    const uint8_t* v413 = v6 + v412;
    const int8_t* v414 = (const int8_t*) v413;
    uint32_t v415 = v386 >> 7u;
    uint32_t v416 = v415 & 127u;
    size_t v417 = (size_t) v416;
    size_t v418 = v417 * 8;
    const int8_t* v419 = weft_iq2xxs_signs64 + v418;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v420 = __riscv_vle8_v_i8mf2(v414, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v421 = __riscv_vle8_v_i8mf2(v419, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v422 = __riscv_vmul_vv_i8mf2(v420, v421, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v423 = __riscv_vsext_vf4_i32m2(v422, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v424 = __riscv_vfcvt_f_x_v_f32m2(v423, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v425 = __riscv_vfmul_vf_f32m2(v424, v391, 8);
    float* v426 = v12 + 136;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v426, v425, 8);
    const uint8_t v427 = v9[36];
    uint32_t v428 = (uint32_t) v427;
    size_t v429 = (size_t) v428;
    size_t v430 = v429 * 8;
    const uint8_t* v431 = v6 + v430;
    const int8_t* v432 = (const int8_t*) v431;
    uint32_t v433 = v386 >> 14u;
    uint32_t v434 = v433 & 127u;
    size_t v435 = (size_t) v434;
    size_t v436 = v435 * 8;
    const int8_t* v437 = weft_iq2xxs_signs64 + v436;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v438 = __riscv_vle8_v_i8mf2(v432, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v439 = __riscv_vle8_v_i8mf2(v437, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v440 = __riscv_vmul_vv_i8mf2(v438, v439, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v441 = __riscv_vsext_vf4_i32m2(v440, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v442 = __riscv_vfcvt_f_x_v_f32m2(v441, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v443 = __riscv_vfmul_vf_f32m2(v442, v391, 8);
    float* v444 = v12 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v444, v443, 8);
    const uint8_t v445 = v9[37];
    uint32_t v446 = (uint32_t) v445;
    size_t v447 = (size_t) v446;
    size_t v448 = v447 * 8;
    const uint8_t* v449 = v6 + v448;
    const int8_t* v450 = (const int8_t*) v449;
    uint32_t v451 = v386 >> 21u;
    uint32_t v452 = v451 & 127u;
    size_t v453 = (size_t) v452;
    size_t v454 = v453 * 8;
    const int8_t* v455 = weft_iq2xxs_signs64 + v454;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v456 = __riscv_vle8_v_i8mf2(v450, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v457 = __riscv_vle8_v_i8mf2(v455, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v458 = __riscv_vmul_vv_i8mf2(v456, v457, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v459 = __riscv_vsext_vf4_i32m2(v458, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v460 = __riscv_vfcvt_f_x_v_f32m2(v459, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v461 = __riscv_vfmul_vf_f32m2(v460, v391, 8);
    float* v462 = v12 + 152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v462, v461, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v463 = v9[49];
    uint32_t v464 = (uint32_t) v463;
    uint32_t v465 = v464 << 24u;
    const uint8_t v466 = v9[48];
    uint32_t v467 = (uint32_t) v466;
    uint32_t v468 = v467 << 16u;
    const uint8_t v469 = v9[47];
    uint32_t v470 = (uint32_t) v469;
    uint32_t v471 = v470 << 8u;
    const uint8_t v472 = v9[46];
    uint32_t v473 = (uint32_t) v472;
    uint32_t v474 = v473 | v471;
    uint32_t v475 = v474 | v468;
    uint32_t v476 = v475 | v465;
    uint32_t v477 = v476 >> 28u;
    float v478 = (float) v477;
    float v479 = 0.5f + v478;
    float v480 = v13 * v479;
    float v481 = v480 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v482 = v9[42];
    uint32_t v483 = (uint32_t) v482;
    size_t v484 = (size_t) v483;
    size_t v485 = v484 * 8;
    const uint8_t* v486 = v6 + v485;
    const int8_t* v487 = (const int8_t*) v486;
    uint32_t v488 = v476 & 127u;
    size_t v489 = (size_t) v488;
    size_t v490 = v489 * 8;
    const int8_t* v491 = weft_iq2xxs_signs64 + v490;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v492 = __riscv_vle8_v_i8mf2(v487, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v493 = __riscv_vle8_v_i8mf2(v491, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v494 = __riscv_vmul_vv_i8mf2(v492, v493, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v495 = __riscv_vsext_vf4_i32m2(v494, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v496 = __riscv_vfcvt_f_x_v_f32m2(v495, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v497 = __riscv_vfmul_vf_f32m2(v496, v481, 8);
    float* v498 = v12 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v498, v497, 8);
    const uint8_t v499 = v9[43];
    uint32_t v500 = (uint32_t) v499;
    size_t v501 = (size_t) v500;
    size_t v502 = v501 * 8;
    const uint8_t* v503 = v6 + v502;
    const int8_t* v504 = (const int8_t*) v503;
    uint32_t v505 = v476 >> 7u;
    uint32_t v506 = v505 & 127u;
    size_t v507 = (size_t) v506;
    size_t v508 = v507 * 8;
    const int8_t* v509 = weft_iq2xxs_signs64 + v508;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v510 = __riscv_vle8_v_i8mf2(v504, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v511 = __riscv_vle8_v_i8mf2(v509, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v512 = __riscv_vmul_vv_i8mf2(v510, v511, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v513 = __riscv_vsext_vf4_i32m2(v512, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v514 = __riscv_vfcvt_f_x_v_f32m2(v513, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v515 = __riscv_vfmul_vf_f32m2(v514, v481, 8);
    float* v516 = v12 + 168;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v516, v515, 8);
    const uint8_t v517 = v9[44];
    uint32_t v518 = (uint32_t) v517;
    size_t v519 = (size_t) v518;
    size_t v520 = v519 * 8;
    const uint8_t* v521 = v6 + v520;
    const int8_t* v522 = (const int8_t*) v521;
    uint32_t v523 = v476 >> 14u;
    uint32_t v524 = v523 & 127u;
    size_t v525 = (size_t) v524;
    size_t v526 = v525 * 8;
    const int8_t* v527 = weft_iq2xxs_signs64 + v526;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v528 = __riscv_vle8_v_i8mf2(v522, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v529 = __riscv_vle8_v_i8mf2(v527, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v530 = __riscv_vmul_vv_i8mf2(v528, v529, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v531 = __riscv_vsext_vf4_i32m2(v530, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v532 = __riscv_vfcvt_f_x_v_f32m2(v531, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v533 = __riscv_vfmul_vf_f32m2(v532, v481, 8);
    float* v534 = v12 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v534, v533, 8);
    const uint8_t v535 = v9[45];
    uint32_t v536 = (uint32_t) v535;
    size_t v537 = (size_t) v536;
    size_t v538 = v537 * 8;
    const uint8_t* v539 = v6 + v538;
    const int8_t* v540 = (const int8_t*) v539;
    uint32_t v541 = v476 >> 21u;
    uint32_t v542 = v541 & 127u;
    size_t v543 = (size_t) v542;
    size_t v544 = v543 * 8;
    const int8_t* v545 = weft_iq2xxs_signs64 + v544;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v546 = __riscv_vle8_v_i8mf2(v540, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v547 = __riscv_vle8_v_i8mf2(v545, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v548 = __riscv_vmul_vv_i8mf2(v546, v547, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v549 = __riscv_vsext_vf4_i32m2(v548, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v550 = __riscv_vfcvt_f_x_v_f32m2(v549, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v551 = __riscv_vfmul_vf_f32m2(v550, v481, 8);
    float* v552 = v12 + 184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v552, v551, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v553 = v9[57];
    uint32_t v554 = (uint32_t) v553;
    uint32_t v555 = v554 << 24u;
    const uint8_t v556 = v9[56];
    uint32_t v557 = (uint32_t) v556;
    uint32_t v558 = v557 << 16u;
    const uint8_t v559 = v9[55];
    uint32_t v560 = (uint32_t) v559;
    uint32_t v561 = v560 << 8u;
    const uint8_t v562 = v9[54];
    uint32_t v563 = (uint32_t) v562;
    uint32_t v564 = v563 | v561;
    uint32_t v565 = v564 | v558;
    uint32_t v566 = v565 | v555;
    uint32_t v567 = v566 >> 28u;
    float v568 = (float) v567;
    float v569 = 0.5f + v568;
    float v570 = v13 * v569;
    float v571 = v570 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v572 = v9[50];
    uint32_t v573 = (uint32_t) v572;
    size_t v574 = (size_t) v573;
    size_t v575 = v574 * 8;
    const uint8_t* v576 = v6 + v575;
    const int8_t* v577 = (const int8_t*) v576;
    uint32_t v578 = v566 & 127u;
    size_t v579 = (size_t) v578;
    size_t v580 = v579 * 8;
    const int8_t* v581 = weft_iq2xxs_signs64 + v580;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v582 = __riscv_vle8_v_i8mf2(v577, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v583 = __riscv_vle8_v_i8mf2(v581, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v584 = __riscv_vmul_vv_i8mf2(v582, v583, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v585 = __riscv_vsext_vf4_i32m2(v584, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v586 = __riscv_vfcvt_f_x_v_f32m2(v585, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v587 = __riscv_vfmul_vf_f32m2(v586, v571, 8);
    float* v588 = v12 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v588, v587, 8);
    const uint8_t v589 = v9[51];
    uint32_t v590 = (uint32_t) v589;
    size_t v591 = (size_t) v590;
    size_t v592 = v591 * 8;
    const uint8_t* v593 = v6 + v592;
    const int8_t* v594 = (const int8_t*) v593;
    uint32_t v595 = v566 >> 7u;
    uint32_t v596 = v595 & 127u;
    size_t v597 = (size_t) v596;
    size_t v598 = v597 * 8;
    const int8_t* v599 = weft_iq2xxs_signs64 + v598;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v600 = __riscv_vle8_v_i8mf2(v594, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v601 = __riscv_vle8_v_i8mf2(v599, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v602 = __riscv_vmul_vv_i8mf2(v600, v601, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v603 = __riscv_vsext_vf4_i32m2(v602, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v604 = __riscv_vfcvt_f_x_v_f32m2(v603, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v605 = __riscv_vfmul_vf_f32m2(v604, v571, 8);
    float* v606 = v12 + 200;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v606, v605, 8);
    const uint8_t v607 = v9[52];
    uint32_t v608 = (uint32_t) v607;
    size_t v609 = (size_t) v608;
    size_t v610 = v609 * 8;
    const uint8_t* v611 = v6 + v610;
    const int8_t* v612 = (const int8_t*) v611;
    uint32_t v613 = v566 >> 14u;
    uint32_t v614 = v613 & 127u;
    size_t v615 = (size_t) v614;
    size_t v616 = v615 * 8;
    const int8_t* v617 = weft_iq2xxs_signs64 + v616;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v618 = __riscv_vle8_v_i8mf2(v612, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v619 = __riscv_vle8_v_i8mf2(v617, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v620 = __riscv_vmul_vv_i8mf2(v618, v619, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v621 = __riscv_vsext_vf4_i32m2(v620, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v622 = __riscv_vfcvt_f_x_v_f32m2(v621, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v623 = __riscv_vfmul_vf_f32m2(v622, v571, 8);
    float* v624 = v12 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v624, v623, 8);
    const uint8_t v625 = v9[53];
    uint32_t v626 = (uint32_t) v625;
    size_t v627 = (size_t) v626;
    size_t v628 = v627 * 8;
    const uint8_t* v629 = v6 + v628;
    const int8_t* v630 = (const int8_t*) v629;
    uint32_t v631 = v566 >> 21u;
    uint32_t v632 = v631 & 127u;
    size_t v633 = (size_t) v632;
    size_t v634 = v633 * 8;
    const int8_t* v635 = weft_iq2xxs_signs64 + v634;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v636 = __riscv_vle8_v_i8mf2(v630, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v637 = __riscv_vle8_v_i8mf2(v635, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v638 = __riscv_vmul_vv_i8mf2(v636, v637, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v639 = __riscv_vsext_vf4_i32m2(v638, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v640 = __riscv_vfcvt_f_x_v_f32m2(v639, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v641 = __riscv_vfmul_vf_f32m2(v640, v571, 8);
    float* v642 = v12 + 216;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v642, v641, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v643 = v9[65];
    uint32_t v644 = (uint32_t) v643;
    uint32_t v645 = v644 << 24u;
    const uint8_t v646 = v9[64];
    uint32_t v647 = (uint32_t) v646;
    uint32_t v648 = v647 << 16u;
    const uint8_t v649 = v9[63];
    uint32_t v650 = (uint32_t) v649;
    uint32_t v651 = v650 << 8u;
    const uint8_t v652 = v9[62];
    uint32_t v653 = (uint32_t) v652;
    uint32_t v654 = v653 | v651;
    uint32_t v655 = v654 | v648;
    uint32_t v656 = v655 | v645;
    uint32_t v657 = v656 >> 28u;
    float v658 = (float) v657;
    float v659 = 0.5f + v658;
    float v660 = v13 * v659;
    float v661 = v660 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v662 = v9[58];
    uint32_t v663 = (uint32_t) v662;
    size_t v664 = (size_t) v663;
    size_t v665 = v664 * 8;
    const uint8_t* v666 = v6 + v665;
    const int8_t* v667 = (const int8_t*) v666;
    uint32_t v668 = v656 & 127u;
    size_t v669 = (size_t) v668;
    size_t v670 = v669 * 8;
    const int8_t* v671 = weft_iq2xxs_signs64 + v670;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v672 = __riscv_vle8_v_i8mf2(v667, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v673 = __riscv_vle8_v_i8mf2(v671, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v674 = __riscv_vmul_vv_i8mf2(v672, v673, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v675 = __riscv_vsext_vf4_i32m2(v674, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v676 = __riscv_vfcvt_f_x_v_f32m2(v675, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v677 = __riscv_vfmul_vf_f32m2(v676, v661, 8);
    float* v678 = v12 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v678, v677, 8);
    const uint8_t v679 = v9[59];
    uint32_t v680 = (uint32_t) v679;
    size_t v681 = (size_t) v680;
    size_t v682 = v681 * 8;
    const uint8_t* v683 = v6 + v682;
    const int8_t* v684 = (const int8_t*) v683;
    uint32_t v685 = v656 >> 7u;
    uint32_t v686 = v685 & 127u;
    size_t v687 = (size_t) v686;
    size_t v688 = v687 * 8;
    const int8_t* v689 = weft_iq2xxs_signs64 + v688;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v690 = __riscv_vle8_v_i8mf2(v684, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v691 = __riscv_vle8_v_i8mf2(v689, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v692 = __riscv_vmul_vv_i8mf2(v690, v691, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v693 = __riscv_vsext_vf4_i32m2(v692, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v694 = __riscv_vfcvt_f_x_v_f32m2(v693, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v695 = __riscv_vfmul_vf_f32m2(v694, v661, 8);
    float* v696 = v12 + 232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v696, v695, 8);
    const uint8_t v697 = v9[60];
    uint32_t v698 = (uint32_t) v697;
    size_t v699 = (size_t) v698;
    size_t v700 = v699 * 8;
    const uint8_t* v701 = v6 + v700;
    const int8_t* v702 = (const int8_t*) v701;
    uint32_t v703 = v656 >> 14u;
    uint32_t v704 = v703 & 127u;
    size_t v705 = (size_t) v704;
    size_t v706 = v705 * 8;
    const int8_t* v707 = weft_iq2xxs_signs64 + v706;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v708 = __riscv_vle8_v_i8mf2(v702, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v709 = __riscv_vle8_v_i8mf2(v707, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v710 = __riscv_vmul_vv_i8mf2(v708, v709, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v711 = __riscv_vsext_vf4_i32m2(v710, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v712 = __riscv_vfcvt_f_x_v_f32m2(v711, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v713 = __riscv_vfmul_vf_f32m2(v712, v661, 8);
    float* v714 = v12 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v714, v713, 8);
    const uint8_t v715 = v9[61];
    uint32_t v716 = (uint32_t) v715;
    size_t v717 = (size_t) v716;
    size_t v718 = v717 * 8;
    const uint8_t* v719 = v6 + v718;
    const int8_t* v720 = (const int8_t*) v719;
    uint32_t v721 = v656 >> 21u;
    uint32_t v722 = v721 & 127u;
    size_t v723 = (size_t) v722;
    size_t v724 = v723 * 8;
    const int8_t* v725 = weft_iq2xxs_signs64 + v724;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v726 = __riscv_vle8_v_i8mf2(v720, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v727 = __riscv_vle8_v_i8mf2(v725, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v728 = __riscv_vmul_vv_i8mf2(v726, v727, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v729 = __riscv_vsext_vf4_i32m2(v728, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v730 = __riscv_vfcvt_f_x_v_f32m2(v729, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v731 = __riscv_vfmul_vf_f32m2(v730, v661, 8);
    float* v732 = v12 + 248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v732, v731, 8);
  }
  return;
}


