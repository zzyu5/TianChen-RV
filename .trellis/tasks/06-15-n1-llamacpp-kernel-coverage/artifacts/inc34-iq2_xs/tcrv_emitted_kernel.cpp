#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_ggml_vec_dot_iq2_xs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int64_t tcrv_iq2xs_grid[512] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL, 0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL, 0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL, 0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL, 0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL, 0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL};
  static const uint8_t tcrv_iq2xs_ksigns[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
  static const uint8_t tcrv_iq2xs_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_iq2xs_kmask, 8);
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_byte_view
  const int8_t* v9 = (const int8_t*) tcrv_iq2xs_grid;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v7; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 74;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v16 = (const float*) v14;
    const float v17 = v16[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v18 = v15 * v17;
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    const uint8_t* v21 = v12 + 66;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t* v23 = v14 + 4;
    const int8_t* v24 = (const int8_t*) v23;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t v26 = v22[0];
    int v27 = (int) v26;
    int v28 = v27 & 15;
    int v29 = v27 >> 4;
    int v30 = v28 * 2;
    int v31 = v30 + 1;
    int v32 = v29 * 2;
    int v33 = v32 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v34 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v35 = v20[0];
    uint32_t v36 = (uint32_t) v35;
    const uint8_t v37 = v20[1];
    uint32_t v38 = (uint32_t) v37;
    uint32_t v39 = v38 << 8u;
    uint32_t v40 = v36 | v39;
    uint32_t v41 = v40 & 511u;
    int v42 = (int) v41;
    uint32_t v43 = v40 >> 9u;
    int v44 = (int) v43;
    const uint8_t v45 = tcrv_iq2xs_ksigns[v44];
    int v46 = (int) v45;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v47 = __riscv_vsetvl_e8m1(8);
    size_t v48 = v42 * 8;
    const int8_t* v49 = v9 + v48;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v50 = __riscv_vle8_v_i8m1(v49, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v51 = __riscv_vle8_v_i8m1(v24, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v52 = __riscv_vmv_v_x_u8m1(v46, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v53 = __riscv_vand_vv_u8m1(v52, v8, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v54 = __riscv_vmsne_vx_u8m1_b8(v53, 0, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v55 = __riscv_vneg_v_i8m1(v50, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v56 = __riscv_vmerge_vvm_i8m1(v50, v55, v54, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v57 = __riscv_vwmul_vv_i16m2(v56, v51, v47);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v58 = __riscv_vwredsum_vs_i16m2_i32m1(v57, v34, v47);
    const int8_t* v59 = v24 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v60 = v20[2];
    uint32_t v61 = (uint32_t) v60;
    const uint8_t v62 = v20[3];
    uint32_t v63 = (uint32_t) v62;
    uint32_t v64 = v63 << 8u;
    uint32_t v65 = v61 | v64;
    uint32_t v66 = v65 & 511u;
    int v67 = (int) v66;
    uint32_t v68 = v65 >> 9u;
    int v69 = (int) v68;
    const uint8_t v70 = tcrv_iq2xs_ksigns[v69];
    int v71 = (int) v70;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v72 = __riscv_vsetvl_e8m1(8);
    size_t v73 = v67 * 8;
    const int8_t* v74 = v9 + v73;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v75 = __riscv_vle8_v_i8m1(v74, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v76 = __riscv_vle8_v_i8m1(v59, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v77 = __riscv_vmv_v_x_u8m1(v71, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v78 = __riscv_vand_vv_u8m1(v77, v8, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v79 = __riscv_vmsne_vx_u8m1_b8(v78, 0, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v80 = __riscv_vneg_v_i8m1(v75, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v81 = __riscv_vmerge_vvm_i8m1(v75, v80, v79, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v82 = __riscv_vwmul_vv_i16m2(v81, v76, v72);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v83 = __riscv_vwredsum_vs_i16m2_i32m1(v82, v58, v72);
    const int8_t* v84 = v59 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v85 = __riscv_vmv_x_s_i32m1_i32(v83);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v86 = v25;
    int32_t v87 = (int32_t) v31;
    int32_t v88 = v85 * v87;
    int32_t v89 = v86 + v88;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v89;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v90 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v91 = v20[4];
    uint32_t v92 = (uint32_t) v91;
    const uint8_t v93 = v20[5];
    uint32_t v94 = (uint32_t) v93;
    uint32_t v95 = v94 << 8u;
    uint32_t v96 = v92 | v95;
    uint32_t v97 = v96 & 511u;
    int v98 = (int) v97;
    uint32_t v99 = v96 >> 9u;
    int v100 = (int) v99;
    const uint8_t v101 = tcrv_iq2xs_ksigns[v100];
    int v102 = (int) v101;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v103 = __riscv_vsetvl_e8m1(8);
    size_t v104 = v98 * 8;
    const int8_t* v105 = v9 + v104;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v106 = __riscv_vle8_v_i8m1(v105, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v107 = __riscv_vle8_v_i8m1(v84, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v108 = __riscv_vmv_v_x_u8m1(v102, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v109 = __riscv_vand_vv_u8m1(v108, v8, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v110 = __riscv_vmsne_vx_u8m1_b8(v109, 0, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v111 = __riscv_vneg_v_i8m1(v106, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v112 = __riscv_vmerge_vvm_i8m1(v106, v111, v110, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v113 = __riscv_vwmul_vv_i16m2(v112, v107, v103);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v114 = __riscv_vwredsum_vs_i16m2_i32m1(v113, v90, v103);
    const int8_t* v115 = v84 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v116 = v20[6];
    uint32_t v117 = (uint32_t) v116;
    const uint8_t v118 = v20[7];
    uint32_t v119 = (uint32_t) v118;
    uint32_t v120 = v119 << 8u;
    uint32_t v121 = v117 | v120;
    uint32_t v122 = v121 & 511u;
    int v123 = (int) v122;
    uint32_t v124 = v121 >> 9u;
    int v125 = (int) v124;
    const uint8_t v126 = tcrv_iq2xs_ksigns[v125];
    int v127 = (int) v126;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v128 = __riscv_vsetvl_e8m1(8);
    size_t v129 = v123 * 8;
    const int8_t* v130 = v9 + v129;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v131 = __riscv_vle8_v_i8m1(v130, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v132 = __riscv_vle8_v_i8m1(v115, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v133 = __riscv_vmv_v_x_u8m1(v127, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v134 = __riscv_vand_vv_u8m1(v133, v8, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v135 = __riscv_vmsne_vx_u8m1_b8(v134, 0, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v136 = __riscv_vneg_v_i8m1(v131, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v137 = __riscv_vmerge_vvm_i8m1(v131, v136, v135, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v138 = __riscv_vwmul_vv_i16m2(v137, v132, v128);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v139 = __riscv_vwredsum_vs_i16m2_i32m1(v138, v114, v128);
    const int8_t* v140 = v115 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v141 = __riscv_vmv_x_s_i32m1_i32(v139);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v142 = v25;
    int32_t v143 = (int32_t) v33;
    int32_t v144 = v141 * v143;
    int32_t v145 = v142 + v144;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v145;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v146 = v20 + 8;
    const uint8_t v147 = v22[1];
    int v148 = (int) v147;
    int v149 = v148 & 15;
    int v150 = v148 >> 4;
    int v151 = v149 * 2;
    int v152 = v151 + 1;
    int v153 = v150 * 2;
    int v154 = v153 + 1;
    const int8_t* v155 = v24 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v156 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v157 = v146[0];
    uint32_t v158 = (uint32_t) v157;
    const uint8_t v159 = v146[1];
    uint32_t v160 = (uint32_t) v159;
    uint32_t v161 = v160 << 8u;
    uint32_t v162 = v158 | v161;
    uint32_t v163 = v162 & 511u;
    int v164 = (int) v163;
    uint32_t v165 = v162 >> 9u;
    int v166 = (int) v165;
    const uint8_t v167 = tcrv_iq2xs_ksigns[v166];
    int v168 = (int) v167;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v169 = __riscv_vsetvl_e8m1(8);
    size_t v170 = v164 * 8;
    const int8_t* v171 = v9 + v170;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v172 = __riscv_vle8_v_i8m1(v171, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v173 = __riscv_vle8_v_i8m1(v155, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v174 = __riscv_vmv_v_x_u8m1(v168, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v175 = __riscv_vand_vv_u8m1(v174, v8, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v176 = __riscv_vmsne_vx_u8m1_b8(v175, 0, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v177 = __riscv_vneg_v_i8m1(v172, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v178 = __riscv_vmerge_vvm_i8m1(v172, v177, v176, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v179 = __riscv_vwmul_vv_i16m2(v178, v173, v169);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v180 = __riscv_vwredsum_vs_i16m2_i32m1(v179, v156, v169);
    const int8_t* v181 = v155 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v182 = v146[2];
    uint32_t v183 = (uint32_t) v182;
    const uint8_t v184 = v146[3];
    uint32_t v185 = (uint32_t) v184;
    uint32_t v186 = v185 << 8u;
    uint32_t v187 = v183 | v186;
    uint32_t v188 = v187 & 511u;
    int v189 = (int) v188;
    uint32_t v190 = v187 >> 9u;
    int v191 = (int) v190;
    const uint8_t v192 = tcrv_iq2xs_ksigns[v191];
    int v193 = (int) v192;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v194 = __riscv_vsetvl_e8m1(8);
    size_t v195 = v189 * 8;
    const int8_t* v196 = v9 + v195;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v197 = __riscv_vle8_v_i8m1(v196, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v198 = __riscv_vle8_v_i8m1(v181, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v199 = __riscv_vmv_v_x_u8m1(v193, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v200 = __riscv_vand_vv_u8m1(v199, v8, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v201 = __riscv_vmsne_vx_u8m1_b8(v200, 0, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v202 = __riscv_vneg_v_i8m1(v197, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v203 = __riscv_vmerge_vvm_i8m1(v197, v202, v201, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v204 = __riscv_vwmul_vv_i16m2(v203, v198, v194);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v205 = __riscv_vwredsum_vs_i16m2_i32m1(v204, v180, v194);
    const int8_t* v206 = v181 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v207 = __riscv_vmv_x_s_i32m1_i32(v205);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v208 = v25;
    int32_t v209 = (int32_t) v152;
    int32_t v210 = v207 * v209;
    int32_t v211 = v208 + v210;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v211;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v212 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v213 = v146[4];
    uint32_t v214 = (uint32_t) v213;
    const uint8_t v215 = v146[5];
    uint32_t v216 = (uint32_t) v215;
    uint32_t v217 = v216 << 8u;
    uint32_t v218 = v214 | v217;
    uint32_t v219 = v218 & 511u;
    int v220 = (int) v219;
    uint32_t v221 = v218 >> 9u;
    int v222 = (int) v221;
    const uint8_t v223 = tcrv_iq2xs_ksigns[v222];
    int v224 = (int) v223;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v225 = __riscv_vsetvl_e8m1(8);
    size_t v226 = v220 * 8;
    const int8_t* v227 = v9 + v226;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v228 = __riscv_vle8_v_i8m1(v227, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v229 = __riscv_vle8_v_i8m1(v206, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v230 = __riscv_vmv_v_x_u8m1(v224, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v231 = __riscv_vand_vv_u8m1(v230, v8, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v232 = __riscv_vmsne_vx_u8m1_b8(v231, 0, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v233 = __riscv_vneg_v_i8m1(v228, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v234 = __riscv_vmerge_vvm_i8m1(v228, v233, v232, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v235 = __riscv_vwmul_vv_i16m2(v234, v229, v225);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v236 = __riscv_vwredsum_vs_i16m2_i32m1(v235, v212, v225);
    const int8_t* v237 = v206 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v238 = v146[6];
    uint32_t v239 = (uint32_t) v238;
    const uint8_t v240 = v146[7];
    uint32_t v241 = (uint32_t) v240;
    uint32_t v242 = v241 << 8u;
    uint32_t v243 = v239 | v242;
    uint32_t v244 = v243 & 511u;
    int v245 = (int) v244;
    uint32_t v246 = v243 >> 9u;
    int v247 = (int) v246;
    const uint8_t v248 = tcrv_iq2xs_ksigns[v247];
    int v249 = (int) v248;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v250 = __riscv_vsetvl_e8m1(8);
    size_t v251 = v245 * 8;
    const int8_t* v252 = v9 + v251;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v253 = __riscv_vle8_v_i8m1(v252, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v254 = __riscv_vle8_v_i8m1(v237, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v255 = __riscv_vmv_v_x_u8m1(v249, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v256 = __riscv_vand_vv_u8m1(v255, v8, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v257 = __riscv_vmsne_vx_u8m1_b8(v256, 0, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v258 = __riscv_vneg_v_i8m1(v253, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v259 = __riscv_vmerge_vvm_i8m1(v253, v258, v257, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v260 = __riscv_vwmul_vv_i16m2(v259, v254, v250);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v261 = __riscv_vwredsum_vs_i16m2_i32m1(v260, v236, v250);
    const int8_t* v262 = v237 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v263 = __riscv_vmv_x_s_i32m1_i32(v261);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v264 = v25;
    int32_t v265 = (int32_t) v154;
    int32_t v266 = v263 * v265;
    int32_t v267 = v264 + v266;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v267;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v268 = v20 + 16;
    const uint8_t v269 = v22[2];
    int v270 = (int) v269;
    int v271 = v270 & 15;
    int v272 = v270 >> 4;
    int v273 = v271 * 2;
    int v274 = v273 + 1;
    int v275 = v272 * 2;
    int v276 = v275 + 1;
    const int8_t* v277 = v24 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v278 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v279 = v268[0];
    uint32_t v280 = (uint32_t) v279;
    const uint8_t v281 = v268[1];
    uint32_t v282 = (uint32_t) v281;
    uint32_t v283 = v282 << 8u;
    uint32_t v284 = v280 | v283;
    uint32_t v285 = v284 & 511u;
    int v286 = (int) v285;
    uint32_t v287 = v284 >> 9u;
    int v288 = (int) v287;
    const uint8_t v289 = tcrv_iq2xs_ksigns[v288];
    int v290 = (int) v289;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v291 = __riscv_vsetvl_e8m1(8);
    size_t v292 = v286 * 8;
    const int8_t* v293 = v9 + v292;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v294 = __riscv_vle8_v_i8m1(v293, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v295 = __riscv_vle8_v_i8m1(v277, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v296 = __riscv_vmv_v_x_u8m1(v290, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v297 = __riscv_vand_vv_u8m1(v296, v8, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v298 = __riscv_vmsne_vx_u8m1_b8(v297, 0, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v299 = __riscv_vneg_v_i8m1(v294, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v300 = __riscv_vmerge_vvm_i8m1(v294, v299, v298, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v301 = __riscv_vwmul_vv_i16m2(v300, v295, v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v302 = __riscv_vwredsum_vs_i16m2_i32m1(v301, v278, v291);
    const int8_t* v303 = v277 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v304 = v268[2];
    uint32_t v305 = (uint32_t) v304;
    const uint8_t v306 = v268[3];
    uint32_t v307 = (uint32_t) v306;
    uint32_t v308 = v307 << 8u;
    uint32_t v309 = v305 | v308;
    uint32_t v310 = v309 & 511u;
    int v311 = (int) v310;
    uint32_t v312 = v309 >> 9u;
    int v313 = (int) v312;
    const uint8_t v314 = tcrv_iq2xs_ksigns[v313];
    int v315 = (int) v314;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v316 = __riscv_vsetvl_e8m1(8);
    size_t v317 = v311 * 8;
    const int8_t* v318 = v9 + v317;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v319 = __riscv_vle8_v_i8m1(v318, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v320 = __riscv_vle8_v_i8m1(v303, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v321 = __riscv_vmv_v_x_u8m1(v315, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v322 = __riscv_vand_vv_u8m1(v321, v8, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v323 = __riscv_vmsne_vx_u8m1_b8(v322, 0, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v324 = __riscv_vneg_v_i8m1(v319, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v325 = __riscv_vmerge_vvm_i8m1(v319, v324, v323, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v326 = __riscv_vwmul_vv_i16m2(v325, v320, v316);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v327 = __riscv_vwredsum_vs_i16m2_i32m1(v326, v302, v316);
    const int8_t* v328 = v303 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v329 = __riscv_vmv_x_s_i32m1_i32(v327);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v330 = v25;
    int32_t v331 = (int32_t) v274;
    int32_t v332 = v329 * v331;
    int32_t v333 = v330 + v332;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v333;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v334 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v335 = v268[4];
    uint32_t v336 = (uint32_t) v335;
    const uint8_t v337 = v268[5];
    uint32_t v338 = (uint32_t) v337;
    uint32_t v339 = v338 << 8u;
    uint32_t v340 = v336 | v339;
    uint32_t v341 = v340 & 511u;
    int v342 = (int) v341;
    uint32_t v343 = v340 >> 9u;
    int v344 = (int) v343;
    const uint8_t v345 = tcrv_iq2xs_ksigns[v344];
    int v346 = (int) v345;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v347 = __riscv_vsetvl_e8m1(8);
    size_t v348 = v342 * 8;
    const int8_t* v349 = v9 + v348;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v350 = __riscv_vle8_v_i8m1(v349, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v351 = __riscv_vle8_v_i8m1(v328, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v352 = __riscv_vmv_v_x_u8m1(v346, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v353 = __riscv_vand_vv_u8m1(v352, v8, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v354 = __riscv_vmsne_vx_u8m1_b8(v353, 0, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v355 = __riscv_vneg_v_i8m1(v350, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v356 = __riscv_vmerge_vvm_i8m1(v350, v355, v354, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v357 = __riscv_vwmul_vv_i16m2(v356, v351, v347);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v358 = __riscv_vwredsum_vs_i16m2_i32m1(v357, v334, v347);
    const int8_t* v359 = v328 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v360 = v268[6];
    uint32_t v361 = (uint32_t) v360;
    const uint8_t v362 = v268[7];
    uint32_t v363 = (uint32_t) v362;
    uint32_t v364 = v363 << 8u;
    uint32_t v365 = v361 | v364;
    uint32_t v366 = v365 & 511u;
    int v367 = (int) v366;
    uint32_t v368 = v365 >> 9u;
    int v369 = (int) v368;
    const uint8_t v370 = tcrv_iq2xs_ksigns[v369];
    int v371 = (int) v370;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v372 = __riscv_vsetvl_e8m1(8);
    size_t v373 = v367 * 8;
    const int8_t* v374 = v9 + v373;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v375 = __riscv_vle8_v_i8m1(v374, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v376 = __riscv_vle8_v_i8m1(v359, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v377 = __riscv_vmv_v_x_u8m1(v371, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v378 = __riscv_vand_vv_u8m1(v377, v8, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v379 = __riscv_vmsne_vx_u8m1_b8(v378, 0, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v380 = __riscv_vneg_v_i8m1(v375, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v381 = __riscv_vmerge_vvm_i8m1(v375, v380, v379, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v382 = __riscv_vwmul_vv_i16m2(v381, v376, v372);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v383 = __riscv_vwredsum_vs_i16m2_i32m1(v382, v358, v372);
    const int8_t* v384 = v359 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v385 = __riscv_vmv_x_s_i32m1_i32(v383);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v386 = v25;
    int32_t v387 = (int32_t) v276;
    int32_t v388 = v385 * v387;
    int32_t v389 = v386 + v388;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v389;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v390 = v20 + 24;
    const uint8_t v391 = v22[3];
    int v392 = (int) v391;
    int v393 = v392 & 15;
    int v394 = v392 >> 4;
    int v395 = v393 * 2;
    int v396 = v395 + 1;
    int v397 = v394 * 2;
    int v398 = v397 + 1;
    const int8_t* v399 = v24 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v400 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v401 = v390[0];
    uint32_t v402 = (uint32_t) v401;
    const uint8_t v403 = v390[1];
    uint32_t v404 = (uint32_t) v403;
    uint32_t v405 = v404 << 8u;
    uint32_t v406 = v402 | v405;
    uint32_t v407 = v406 & 511u;
    int v408 = (int) v407;
    uint32_t v409 = v406 >> 9u;
    int v410 = (int) v409;
    const uint8_t v411 = tcrv_iq2xs_ksigns[v410];
    int v412 = (int) v411;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v413 = __riscv_vsetvl_e8m1(8);
    size_t v414 = v408 * 8;
    const int8_t* v415 = v9 + v414;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v416 = __riscv_vle8_v_i8m1(v415, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v417 = __riscv_vle8_v_i8m1(v399, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v418 = __riscv_vmv_v_x_u8m1(v412, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v419 = __riscv_vand_vv_u8m1(v418, v8, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v420 = __riscv_vmsne_vx_u8m1_b8(v419, 0, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v421 = __riscv_vneg_v_i8m1(v416, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v422 = __riscv_vmerge_vvm_i8m1(v416, v421, v420, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v423 = __riscv_vwmul_vv_i16m2(v422, v417, v413);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v424 = __riscv_vwredsum_vs_i16m2_i32m1(v423, v400, v413);
    const int8_t* v425 = v399 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v426 = v390[2];
    uint32_t v427 = (uint32_t) v426;
    const uint8_t v428 = v390[3];
    uint32_t v429 = (uint32_t) v428;
    uint32_t v430 = v429 << 8u;
    uint32_t v431 = v427 | v430;
    uint32_t v432 = v431 & 511u;
    int v433 = (int) v432;
    uint32_t v434 = v431 >> 9u;
    int v435 = (int) v434;
    const uint8_t v436 = tcrv_iq2xs_ksigns[v435];
    int v437 = (int) v436;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v438 = __riscv_vsetvl_e8m1(8);
    size_t v439 = v433 * 8;
    const int8_t* v440 = v9 + v439;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v441 = __riscv_vle8_v_i8m1(v440, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v442 = __riscv_vle8_v_i8m1(v425, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v443 = __riscv_vmv_v_x_u8m1(v437, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v444 = __riscv_vand_vv_u8m1(v443, v8, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v445 = __riscv_vmsne_vx_u8m1_b8(v444, 0, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v446 = __riscv_vneg_v_i8m1(v441, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v447 = __riscv_vmerge_vvm_i8m1(v441, v446, v445, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v448 = __riscv_vwmul_vv_i16m2(v447, v442, v438);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v449 = __riscv_vwredsum_vs_i16m2_i32m1(v448, v424, v438);
    const int8_t* v450 = v425 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v451 = __riscv_vmv_x_s_i32m1_i32(v449);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v452 = v25;
    int32_t v453 = (int32_t) v396;
    int32_t v454 = v451 * v453;
    int32_t v455 = v452 + v454;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v455;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v456 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v457 = v390[4];
    uint32_t v458 = (uint32_t) v457;
    const uint8_t v459 = v390[5];
    uint32_t v460 = (uint32_t) v459;
    uint32_t v461 = v460 << 8u;
    uint32_t v462 = v458 | v461;
    uint32_t v463 = v462 & 511u;
    int v464 = (int) v463;
    uint32_t v465 = v462 >> 9u;
    int v466 = (int) v465;
    const uint8_t v467 = tcrv_iq2xs_ksigns[v466];
    int v468 = (int) v467;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v469 = __riscv_vsetvl_e8m1(8);
    size_t v470 = v464 * 8;
    const int8_t* v471 = v9 + v470;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v472 = __riscv_vle8_v_i8m1(v471, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v473 = __riscv_vle8_v_i8m1(v450, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v474 = __riscv_vmv_v_x_u8m1(v468, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v475 = __riscv_vand_vv_u8m1(v474, v8, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v476 = __riscv_vmsne_vx_u8m1_b8(v475, 0, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v477 = __riscv_vneg_v_i8m1(v472, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v478 = __riscv_vmerge_vvm_i8m1(v472, v477, v476, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v479 = __riscv_vwmul_vv_i16m2(v478, v473, v469);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v480 = __riscv_vwredsum_vs_i16m2_i32m1(v479, v456, v469);
    const int8_t* v481 = v450 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v482 = v390[6];
    uint32_t v483 = (uint32_t) v482;
    const uint8_t v484 = v390[7];
    uint32_t v485 = (uint32_t) v484;
    uint32_t v486 = v485 << 8u;
    uint32_t v487 = v483 | v486;
    uint32_t v488 = v487 & 511u;
    int v489 = (int) v488;
    uint32_t v490 = v487 >> 9u;
    int v491 = (int) v490;
    const uint8_t v492 = tcrv_iq2xs_ksigns[v491];
    int v493 = (int) v492;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v494 = __riscv_vsetvl_e8m1(8);
    size_t v495 = v489 * 8;
    const int8_t* v496 = v9 + v495;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v497 = __riscv_vle8_v_i8m1(v496, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v498 = __riscv_vle8_v_i8m1(v481, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v499 = __riscv_vmv_v_x_u8m1(v493, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v500 = __riscv_vand_vv_u8m1(v499, v8, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v501 = __riscv_vmsne_vx_u8m1_b8(v500, 0, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v502 = __riscv_vneg_v_i8m1(v497, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v503 = __riscv_vmerge_vvm_i8m1(v497, v502, v501, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v504 = __riscv_vwmul_vv_i16m2(v503, v498, v494);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v505 = __riscv_vwredsum_vs_i16m2_i32m1(v504, v480, v494);
    const int8_t* v506 = v481 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v507 = __riscv_vmv_x_s_i32m1_i32(v505);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v508 = v25;
    int32_t v509 = (int32_t) v398;
    int32_t v510 = v507 * v509;
    int32_t v511 = v508 + v510;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v511;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v512 = v20 + 32;
    const uint8_t v513 = v22[4];
    int v514 = (int) v513;
    int v515 = v514 & 15;
    int v516 = v514 >> 4;
    int v517 = v515 * 2;
    int v518 = v517 + 1;
    int v519 = v516 * 2;
    int v520 = v519 + 1;
    const int8_t* v521 = v24 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v522 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v523 = v512[0];
    uint32_t v524 = (uint32_t) v523;
    const uint8_t v525 = v512[1];
    uint32_t v526 = (uint32_t) v525;
    uint32_t v527 = v526 << 8u;
    uint32_t v528 = v524 | v527;
    uint32_t v529 = v528 & 511u;
    int v530 = (int) v529;
    uint32_t v531 = v528 >> 9u;
    int v532 = (int) v531;
    const uint8_t v533 = tcrv_iq2xs_ksigns[v532];
    int v534 = (int) v533;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v535 = __riscv_vsetvl_e8m1(8);
    size_t v536 = v530 * 8;
    const int8_t* v537 = v9 + v536;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v538 = __riscv_vle8_v_i8m1(v537, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v539 = __riscv_vle8_v_i8m1(v521, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v540 = __riscv_vmv_v_x_u8m1(v534, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v541 = __riscv_vand_vv_u8m1(v540, v8, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v542 = __riscv_vmsne_vx_u8m1_b8(v541, 0, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v543 = __riscv_vneg_v_i8m1(v538, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v544 = __riscv_vmerge_vvm_i8m1(v538, v543, v542, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v545 = __riscv_vwmul_vv_i16m2(v544, v539, v535);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v546 = __riscv_vwredsum_vs_i16m2_i32m1(v545, v522, v535);
    const int8_t* v547 = v521 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v548 = v512[2];
    uint32_t v549 = (uint32_t) v548;
    const uint8_t v550 = v512[3];
    uint32_t v551 = (uint32_t) v550;
    uint32_t v552 = v551 << 8u;
    uint32_t v553 = v549 | v552;
    uint32_t v554 = v553 & 511u;
    int v555 = (int) v554;
    uint32_t v556 = v553 >> 9u;
    int v557 = (int) v556;
    const uint8_t v558 = tcrv_iq2xs_ksigns[v557];
    int v559 = (int) v558;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v560 = __riscv_vsetvl_e8m1(8);
    size_t v561 = v555 * 8;
    const int8_t* v562 = v9 + v561;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v563 = __riscv_vle8_v_i8m1(v562, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v564 = __riscv_vle8_v_i8m1(v547, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v565 = __riscv_vmv_v_x_u8m1(v559, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v566 = __riscv_vand_vv_u8m1(v565, v8, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v567 = __riscv_vmsne_vx_u8m1_b8(v566, 0, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v568 = __riscv_vneg_v_i8m1(v563, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v569 = __riscv_vmerge_vvm_i8m1(v563, v568, v567, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v570 = __riscv_vwmul_vv_i16m2(v569, v564, v560);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v571 = __riscv_vwredsum_vs_i16m2_i32m1(v570, v546, v560);
    const int8_t* v572 = v547 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v573 = __riscv_vmv_x_s_i32m1_i32(v571);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v574 = v25;
    int32_t v575 = (int32_t) v518;
    int32_t v576 = v573 * v575;
    int32_t v577 = v574 + v576;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v577;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v578 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v579 = v512[4];
    uint32_t v580 = (uint32_t) v579;
    const uint8_t v581 = v512[5];
    uint32_t v582 = (uint32_t) v581;
    uint32_t v583 = v582 << 8u;
    uint32_t v584 = v580 | v583;
    uint32_t v585 = v584 & 511u;
    int v586 = (int) v585;
    uint32_t v587 = v584 >> 9u;
    int v588 = (int) v587;
    const uint8_t v589 = tcrv_iq2xs_ksigns[v588];
    int v590 = (int) v589;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v591 = __riscv_vsetvl_e8m1(8);
    size_t v592 = v586 * 8;
    const int8_t* v593 = v9 + v592;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v594 = __riscv_vle8_v_i8m1(v593, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v595 = __riscv_vle8_v_i8m1(v572, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v596 = __riscv_vmv_v_x_u8m1(v590, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v597 = __riscv_vand_vv_u8m1(v596, v8, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v598 = __riscv_vmsne_vx_u8m1_b8(v597, 0, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v599 = __riscv_vneg_v_i8m1(v594, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v600 = __riscv_vmerge_vvm_i8m1(v594, v599, v598, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v601 = __riscv_vwmul_vv_i16m2(v600, v595, v591);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v602 = __riscv_vwredsum_vs_i16m2_i32m1(v601, v578, v591);
    const int8_t* v603 = v572 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v604 = v512[6];
    uint32_t v605 = (uint32_t) v604;
    const uint8_t v606 = v512[7];
    uint32_t v607 = (uint32_t) v606;
    uint32_t v608 = v607 << 8u;
    uint32_t v609 = v605 | v608;
    uint32_t v610 = v609 & 511u;
    int v611 = (int) v610;
    uint32_t v612 = v609 >> 9u;
    int v613 = (int) v612;
    const uint8_t v614 = tcrv_iq2xs_ksigns[v613];
    int v615 = (int) v614;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v616 = __riscv_vsetvl_e8m1(8);
    size_t v617 = v611 * 8;
    const int8_t* v618 = v9 + v617;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v619 = __riscv_vle8_v_i8m1(v618, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v620 = __riscv_vle8_v_i8m1(v603, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v621 = __riscv_vmv_v_x_u8m1(v615, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v622 = __riscv_vand_vv_u8m1(v621, v8, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v623 = __riscv_vmsne_vx_u8m1_b8(v622, 0, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v624 = __riscv_vneg_v_i8m1(v619, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v625 = __riscv_vmerge_vvm_i8m1(v619, v624, v623, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v626 = __riscv_vwmul_vv_i16m2(v625, v620, v616);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v627 = __riscv_vwredsum_vs_i16m2_i32m1(v626, v602, v616);
    const int8_t* v628 = v603 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v629 = __riscv_vmv_x_s_i32m1_i32(v627);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v630 = v25;
    int32_t v631 = (int32_t) v520;
    int32_t v632 = v629 * v631;
    int32_t v633 = v630 + v632;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v633;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v634 = v20 + 40;
    const uint8_t v635 = v22[5];
    int v636 = (int) v635;
    int v637 = v636 & 15;
    int v638 = v636 >> 4;
    int v639 = v637 * 2;
    int v640 = v639 + 1;
    int v641 = v638 * 2;
    int v642 = v641 + 1;
    const int8_t* v643 = v24 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v644 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v645 = v634[0];
    uint32_t v646 = (uint32_t) v645;
    const uint8_t v647 = v634[1];
    uint32_t v648 = (uint32_t) v647;
    uint32_t v649 = v648 << 8u;
    uint32_t v650 = v646 | v649;
    uint32_t v651 = v650 & 511u;
    int v652 = (int) v651;
    uint32_t v653 = v650 >> 9u;
    int v654 = (int) v653;
    const uint8_t v655 = tcrv_iq2xs_ksigns[v654];
    int v656 = (int) v655;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v657 = __riscv_vsetvl_e8m1(8);
    size_t v658 = v652 * 8;
    const int8_t* v659 = v9 + v658;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v660 = __riscv_vle8_v_i8m1(v659, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v661 = __riscv_vle8_v_i8m1(v643, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v662 = __riscv_vmv_v_x_u8m1(v656, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v663 = __riscv_vand_vv_u8m1(v662, v8, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v664 = __riscv_vmsne_vx_u8m1_b8(v663, 0, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v665 = __riscv_vneg_v_i8m1(v660, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v666 = __riscv_vmerge_vvm_i8m1(v660, v665, v664, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v667 = __riscv_vwmul_vv_i16m2(v666, v661, v657);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v668 = __riscv_vwredsum_vs_i16m2_i32m1(v667, v644, v657);
    const int8_t* v669 = v643 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v670 = v634[2];
    uint32_t v671 = (uint32_t) v670;
    const uint8_t v672 = v634[3];
    uint32_t v673 = (uint32_t) v672;
    uint32_t v674 = v673 << 8u;
    uint32_t v675 = v671 | v674;
    uint32_t v676 = v675 & 511u;
    int v677 = (int) v676;
    uint32_t v678 = v675 >> 9u;
    int v679 = (int) v678;
    const uint8_t v680 = tcrv_iq2xs_ksigns[v679];
    int v681 = (int) v680;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v682 = __riscv_vsetvl_e8m1(8);
    size_t v683 = v677 * 8;
    const int8_t* v684 = v9 + v683;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v685 = __riscv_vle8_v_i8m1(v684, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v686 = __riscv_vle8_v_i8m1(v669, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v687 = __riscv_vmv_v_x_u8m1(v681, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v688 = __riscv_vand_vv_u8m1(v687, v8, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v689 = __riscv_vmsne_vx_u8m1_b8(v688, 0, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v690 = __riscv_vneg_v_i8m1(v685, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v691 = __riscv_vmerge_vvm_i8m1(v685, v690, v689, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v692 = __riscv_vwmul_vv_i16m2(v691, v686, v682);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v693 = __riscv_vwredsum_vs_i16m2_i32m1(v692, v668, v682);
    const int8_t* v694 = v669 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v695 = __riscv_vmv_x_s_i32m1_i32(v693);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v696 = v25;
    int32_t v697 = (int32_t) v640;
    int32_t v698 = v695 * v697;
    int32_t v699 = v696 + v698;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v699;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v700 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v701 = v634[4];
    uint32_t v702 = (uint32_t) v701;
    const uint8_t v703 = v634[5];
    uint32_t v704 = (uint32_t) v703;
    uint32_t v705 = v704 << 8u;
    uint32_t v706 = v702 | v705;
    uint32_t v707 = v706 & 511u;
    int v708 = (int) v707;
    uint32_t v709 = v706 >> 9u;
    int v710 = (int) v709;
    const uint8_t v711 = tcrv_iq2xs_ksigns[v710];
    int v712 = (int) v711;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v713 = __riscv_vsetvl_e8m1(8);
    size_t v714 = v708 * 8;
    const int8_t* v715 = v9 + v714;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v716 = __riscv_vle8_v_i8m1(v715, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v717 = __riscv_vle8_v_i8m1(v694, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v718 = __riscv_vmv_v_x_u8m1(v712, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v719 = __riscv_vand_vv_u8m1(v718, v8, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v720 = __riscv_vmsne_vx_u8m1_b8(v719, 0, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v721 = __riscv_vneg_v_i8m1(v716, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v722 = __riscv_vmerge_vvm_i8m1(v716, v721, v720, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v723 = __riscv_vwmul_vv_i16m2(v722, v717, v713);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v724 = __riscv_vwredsum_vs_i16m2_i32m1(v723, v700, v713);
    const int8_t* v725 = v694 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v726 = v634[6];
    uint32_t v727 = (uint32_t) v726;
    const uint8_t v728 = v634[7];
    uint32_t v729 = (uint32_t) v728;
    uint32_t v730 = v729 << 8u;
    uint32_t v731 = v727 | v730;
    uint32_t v732 = v731 & 511u;
    int v733 = (int) v732;
    uint32_t v734 = v731 >> 9u;
    int v735 = (int) v734;
    const uint8_t v736 = tcrv_iq2xs_ksigns[v735];
    int v737 = (int) v736;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v738 = __riscv_vsetvl_e8m1(8);
    size_t v739 = v733 * 8;
    const int8_t* v740 = v9 + v739;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v741 = __riscv_vle8_v_i8m1(v740, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v742 = __riscv_vle8_v_i8m1(v725, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v743 = __riscv_vmv_v_x_u8m1(v737, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v744 = __riscv_vand_vv_u8m1(v743, v8, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v745 = __riscv_vmsne_vx_u8m1_b8(v744, 0, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v746 = __riscv_vneg_v_i8m1(v741, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v747 = __riscv_vmerge_vvm_i8m1(v741, v746, v745, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v748 = __riscv_vwmul_vv_i16m2(v747, v742, v738);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v749 = __riscv_vwredsum_vs_i16m2_i32m1(v748, v724, v738);
    const int8_t* v750 = v725 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v751 = __riscv_vmv_x_s_i32m1_i32(v749);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v752 = v25;
    int32_t v753 = (int32_t) v642;
    int32_t v754 = v751 * v753;
    int32_t v755 = v752 + v754;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v755;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v756 = v20 + 48;
    const uint8_t v757 = v22[6];
    int v758 = (int) v757;
    int v759 = v758 & 15;
    int v760 = v758 >> 4;
    int v761 = v759 * 2;
    int v762 = v761 + 1;
    int v763 = v760 * 2;
    int v764 = v763 + 1;
    const int8_t* v765 = v24 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v766 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v767 = v756[0];
    uint32_t v768 = (uint32_t) v767;
    const uint8_t v769 = v756[1];
    uint32_t v770 = (uint32_t) v769;
    uint32_t v771 = v770 << 8u;
    uint32_t v772 = v768 | v771;
    uint32_t v773 = v772 & 511u;
    int v774 = (int) v773;
    uint32_t v775 = v772 >> 9u;
    int v776 = (int) v775;
    const uint8_t v777 = tcrv_iq2xs_ksigns[v776];
    int v778 = (int) v777;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v779 = __riscv_vsetvl_e8m1(8);
    size_t v780 = v774 * 8;
    const int8_t* v781 = v9 + v780;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v782 = __riscv_vle8_v_i8m1(v781, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v783 = __riscv_vle8_v_i8m1(v765, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v784 = __riscv_vmv_v_x_u8m1(v778, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v785 = __riscv_vand_vv_u8m1(v784, v8, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v786 = __riscv_vmsne_vx_u8m1_b8(v785, 0, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v787 = __riscv_vneg_v_i8m1(v782, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v788 = __riscv_vmerge_vvm_i8m1(v782, v787, v786, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v789 = __riscv_vwmul_vv_i16m2(v788, v783, v779);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v790 = __riscv_vwredsum_vs_i16m2_i32m1(v789, v766, v779);
    const int8_t* v791 = v765 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v792 = v756[2];
    uint32_t v793 = (uint32_t) v792;
    const uint8_t v794 = v756[3];
    uint32_t v795 = (uint32_t) v794;
    uint32_t v796 = v795 << 8u;
    uint32_t v797 = v793 | v796;
    uint32_t v798 = v797 & 511u;
    int v799 = (int) v798;
    uint32_t v800 = v797 >> 9u;
    int v801 = (int) v800;
    const uint8_t v802 = tcrv_iq2xs_ksigns[v801];
    int v803 = (int) v802;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v804 = __riscv_vsetvl_e8m1(8);
    size_t v805 = v799 * 8;
    const int8_t* v806 = v9 + v805;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v807 = __riscv_vle8_v_i8m1(v806, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v808 = __riscv_vle8_v_i8m1(v791, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v809 = __riscv_vmv_v_x_u8m1(v803, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v810 = __riscv_vand_vv_u8m1(v809, v8, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v811 = __riscv_vmsne_vx_u8m1_b8(v810, 0, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v812 = __riscv_vneg_v_i8m1(v807, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v813 = __riscv_vmerge_vvm_i8m1(v807, v812, v811, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v814 = __riscv_vwmul_vv_i16m2(v813, v808, v804);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v815 = __riscv_vwredsum_vs_i16m2_i32m1(v814, v790, v804);
    const int8_t* v816 = v791 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v817 = __riscv_vmv_x_s_i32m1_i32(v815);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v818 = v25;
    int32_t v819 = (int32_t) v762;
    int32_t v820 = v817 * v819;
    int32_t v821 = v818 + v820;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v821;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v822 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v823 = v756[4];
    uint32_t v824 = (uint32_t) v823;
    const uint8_t v825 = v756[5];
    uint32_t v826 = (uint32_t) v825;
    uint32_t v827 = v826 << 8u;
    uint32_t v828 = v824 | v827;
    uint32_t v829 = v828 & 511u;
    int v830 = (int) v829;
    uint32_t v831 = v828 >> 9u;
    int v832 = (int) v831;
    const uint8_t v833 = tcrv_iq2xs_ksigns[v832];
    int v834 = (int) v833;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v835 = __riscv_vsetvl_e8m1(8);
    size_t v836 = v830 * 8;
    const int8_t* v837 = v9 + v836;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v838 = __riscv_vle8_v_i8m1(v837, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v839 = __riscv_vle8_v_i8m1(v816, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v840 = __riscv_vmv_v_x_u8m1(v834, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v841 = __riscv_vand_vv_u8m1(v840, v8, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v842 = __riscv_vmsne_vx_u8m1_b8(v841, 0, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v843 = __riscv_vneg_v_i8m1(v838, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v844 = __riscv_vmerge_vvm_i8m1(v838, v843, v842, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v845 = __riscv_vwmul_vv_i16m2(v844, v839, v835);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v846 = __riscv_vwredsum_vs_i16m2_i32m1(v845, v822, v835);
    const int8_t* v847 = v816 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v848 = v756[6];
    uint32_t v849 = (uint32_t) v848;
    const uint8_t v850 = v756[7];
    uint32_t v851 = (uint32_t) v850;
    uint32_t v852 = v851 << 8u;
    uint32_t v853 = v849 | v852;
    uint32_t v854 = v853 & 511u;
    int v855 = (int) v854;
    uint32_t v856 = v853 >> 9u;
    int v857 = (int) v856;
    const uint8_t v858 = tcrv_iq2xs_ksigns[v857];
    int v859 = (int) v858;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v860 = __riscv_vsetvl_e8m1(8);
    size_t v861 = v855 * 8;
    const int8_t* v862 = v9 + v861;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v863 = __riscv_vle8_v_i8m1(v862, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v864 = __riscv_vle8_v_i8m1(v847, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v865 = __riscv_vmv_v_x_u8m1(v859, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v866 = __riscv_vand_vv_u8m1(v865, v8, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v867 = __riscv_vmsne_vx_u8m1_b8(v866, 0, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v868 = __riscv_vneg_v_i8m1(v863, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v869 = __riscv_vmerge_vvm_i8m1(v863, v868, v867, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v870 = __riscv_vwmul_vv_i16m2(v869, v864, v860);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v871 = __riscv_vwredsum_vs_i16m2_i32m1(v870, v846, v860);
    const int8_t* v872 = v847 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v873 = __riscv_vmv_x_s_i32m1_i32(v871);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v874 = v25;
    int32_t v875 = (int32_t) v764;
    int32_t v876 = v873 * v875;
    int32_t v877 = v874 + v876;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v877;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v878 = v20 + 56;
    const uint8_t v879 = v22[7];
    int v880 = (int) v879;
    int v881 = v880 & 15;
    int v882 = v880 >> 4;
    int v883 = v881 * 2;
    int v884 = v883 + 1;
    int v885 = v882 * 2;
    int v886 = v885 + 1;
    const int8_t* v887 = v24 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v888 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v889 = v878[0];
    uint32_t v890 = (uint32_t) v889;
    const uint8_t v891 = v878[1];
    uint32_t v892 = (uint32_t) v891;
    uint32_t v893 = v892 << 8u;
    uint32_t v894 = v890 | v893;
    uint32_t v895 = v894 & 511u;
    int v896 = (int) v895;
    uint32_t v897 = v894 >> 9u;
    int v898 = (int) v897;
    const uint8_t v899 = tcrv_iq2xs_ksigns[v898];
    int v900 = (int) v899;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v901 = __riscv_vsetvl_e8m1(8);
    size_t v902 = v896 * 8;
    const int8_t* v903 = v9 + v902;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v904 = __riscv_vle8_v_i8m1(v903, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v905 = __riscv_vle8_v_i8m1(v887, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v906 = __riscv_vmv_v_x_u8m1(v900, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v907 = __riscv_vand_vv_u8m1(v906, v8, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v908 = __riscv_vmsne_vx_u8m1_b8(v907, 0, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v909 = __riscv_vneg_v_i8m1(v904, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v910 = __riscv_vmerge_vvm_i8m1(v904, v909, v908, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v911 = __riscv_vwmul_vv_i16m2(v910, v905, v901);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v912 = __riscv_vwredsum_vs_i16m2_i32m1(v911, v888, v901);
    const int8_t* v913 = v887 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v914 = v878[2];
    uint32_t v915 = (uint32_t) v914;
    const uint8_t v916 = v878[3];
    uint32_t v917 = (uint32_t) v916;
    uint32_t v918 = v917 << 8u;
    uint32_t v919 = v915 | v918;
    uint32_t v920 = v919 & 511u;
    int v921 = (int) v920;
    uint32_t v922 = v919 >> 9u;
    int v923 = (int) v922;
    const uint8_t v924 = tcrv_iq2xs_ksigns[v923];
    int v925 = (int) v924;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v926 = __riscv_vsetvl_e8m1(8);
    size_t v927 = v921 * 8;
    const int8_t* v928 = v9 + v927;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v929 = __riscv_vle8_v_i8m1(v928, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v930 = __riscv_vle8_v_i8m1(v913, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v931 = __riscv_vmv_v_x_u8m1(v925, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v932 = __riscv_vand_vv_u8m1(v931, v8, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v933 = __riscv_vmsne_vx_u8m1_b8(v932, 0, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v934 = __riscv_vneg_v_i8m1(v929, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v935 = __riscv_vmerge_vvm_i8m1(v929, v934, v933, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v936 = __riscv_vwmul_vv_i16m2(v935, v930, v926);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v937 = __riscv_vwredsum_vs_i16m2_i32m1(v936, v912, v926);
    const int8_t* v938 = v913 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v939 = __riscv_vmv_x_s_i32m1_i32(v937);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v940 = v25;
    int32_t v941 = (int32_t) v884;
    int32_t v942 = v939 * v941;
    int32_t v943 = v940 + v942;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v943;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v944 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v945 = v878[4];
    uint32_t v946 = (uint32_t) v945;
    const uint8_t v947 = v878[5];
    uint32_t v948 = (uint32_t) v947;
    uint32_t v949 = v948 << 8u;
    uint32_t v950 = v946 | v949;
    uint32_t v951 = v950 & 511u;
    int v952 = (int) v951;
    uint32_t v953 = v950 >> 9u;
    int v954 = (int) v953;
    const uint8_t v955 = tcrv_iq2xs_ksigns[v954];
    int v956 = (int) v955;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v957 = __riscv_vsetvl_e8m1(8);
    size_t v958 = v952 * 8;
    const int8_t* v959 = v9 + v958;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v960 = __riscv_vle8_v_i8m1(v959, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v961 = __riscv_vle8_v_i8m1(v938, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v962 = __riscv_vmv_v_x_u8m1(v956, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v963 = __riscv_vand_vv_u8m1(v962, v8, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v964 = __riscv_vmsne_vx_u8m1_b8(v963, 0, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v965 = __riscv_vneg_v_i8m1(v960, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v966 = __riscv_vmerge_vvm_i8m1(v960, v965, v964, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v967 = __riscv_vwmul_vv_i16m2(v966, v961, v957);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v968 = __riscv_vwredsum_vs_i16m2_i32m1(v967, v944, v957);
    const int8_t* v969 = v938 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_group
    const uint8_t v970 = v878[6];
    uint32_t v971 = (uint32_t) v970;
    const uint8_t v972 = v878[7];
    uint32_t v973 = (uint32_t) v972;
    uint32_t v974 = v973 << 8u;
    uint32_t v975 = v971 | v974;
    uint32_t v976 = v975 & 511u;
    int v977 = (int) v976;
    uint32_t v978 = v975 >> 9u;
    int v979 = (int) v978;
    const uint8_t v980 = tcrv_iq2xs_ksigns[v979];
    int v981 = (int) v980;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v982 = __riscv_vsetvl_e8m1(8);
    size_t v983 = v977 * 8;
    const int8_t* v984 = v9 + v983;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v985 = __riscv_vle8_v_i8m1(v984, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v986 = __riscv_vle8_v_i8m1(v969, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v987 = __riscv_vmv_v_x_u8m1(v981, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v988 = __riscv_vand_vv_u8m1(v987, v8, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v989 = __riscv_vmsne_vx_u8m1_b8(v988, 0, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v990 = __riscv_vneg_v_i8m1(v985, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v991 = __riscv_vmerge_vvm_i8m1(v985, v990, v989, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v992 = __riscv_vwmul_vv_i16m2(v991, v986, v982);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v993 = __riscv_vwredsum_vs_i16m2_i32m1(v992, v968, v982);
    const int8_t* v994 = v969 + 8;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v995 = __riscv_vmv_x_s_i32m1_i32(v993);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v996 = v25;
    int32_t v997 = (int32_t) v886;
    int32_t v998 = v995 * v997;
    int32_t v999 = v996 + v998;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v999;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v1000 = v25;
    float v1001 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v1001 + v18 * (float) v1000;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v1002 = v6;
  float v1003 = 0.125f * v1002;
  v2[0] = v1003;
  return;
}


