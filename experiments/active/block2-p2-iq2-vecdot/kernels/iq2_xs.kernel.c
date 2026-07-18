#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int64_t weft_iq2xs_grid[512] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL, 0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL, 0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL, 0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL, 0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL, 0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL};
  static const int8_t weft_iq2xs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_table_i64_view
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=signs_table_i64_view
  const int64_t* v8 = (const int64_t*) weft_iq2xs_signs64;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 74;
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
    const uint8_t* v20 = v11 + 66;
    const uint8_t* v21 = (const uint8_t*) v20;
    const uint8_t* v22 = v13 + 4;
    const int8_t* v23 = (const int8_t*) v22;
    // weft_emitc.local_variable=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v24;
    v24 = 0;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v25[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v26[8];
    const uint8_t v27 = v21[0];
    int v28 = (int) v27;
    int v29 = v28 & 15;
    int v30 = v28 >> 4;
    int v31 = v29 * 2;
    int v32 = v31 + 1;
    int v33 = v30 * 2;
    int v34 = v33 + 1;
    const uint8_t v35 = v19[0];
    uint32_t v36 = (uint32_t) v35;
    const uint8_t v37 = v19[1];
    uint32_t v38 = (uint32_t) v37;
    uint32_t v39 = v38 << 8u;
    uint32_t v40 = v36 | v39;
    uint32_t v41 = v40 & 511u;
    int v42 = (int) v41;
    int v43 = v42 * 8;
    uint16_t v44 = (uint16_t) v43;
    v25[0] = v44;
    uint32_t v45 = v40 >> 9u;
    int v46 = (int) v45;
    int v47 = v46 * 8;
    uint16_t v48 = (uint16_t) v47;
    v26[0] = v48;
    const uint8_t v49 = v19[2];
    uint32_t v50 = (uint32_t) v49;
    const uint8_t v51 = v19[3];
    uint32_t v52 = (uint32_t) v51;
    uint32_t v53 = v52 << 8u;
    uint32_t v54 = v50 | v53;
    uint32_t v55 = v54 & 511u;
    int v56 = (int) v55;
    int v57 = v56 * 8;
    uint16_t v58 = (uint16_t) v57;
    v25[1] = v58;
    uint32_t v59 = v54 >> 9u;
    int v60 = (int) v59;
    int v61 = v60 * 8;
    uint16_t v62 = (uint16_t) v61;
    v26[1] = v62;
    const uint8_t v63 = v19[4];
    uint32_t v64 = (uint32_t) v63;
    const uint8_t v65 = v19[5];
    uint32_t v66 = (uint32_t) v65;
    uint32_t v67 = v66 << 8u;
    uint32_t v68 = v64 | v67;
    uint32_t v69 = v68 & 511u;
    int v70 = (int) v69;
    int v71 = v70 * 8;
    uint16_t v72 = (uint16_t) v71;
    v25[2] = v72;
    uint32_t v73 = v68 >> 9u;
    int v74 = (int) v73;
    int v75 = v74 * 8;
    uint16_t v76 = (uint16_t) v75;
    v26[2] = v76;
    const uint8_t v77 = v19[6];
    uint32_t v78 = (uint32_t) v77;
    const uint8_t v79 = v19[7];
    uint32_t v80 = (uint32_t) v79;
    uint32_t v81 = v80 << 8u;
    uint32_t v82 = v78 | v81;
    uint32_t v83 = v82 & 511u;
    int v84 = (int) v83;
    int v85 = v84 * 8;
    uint16_t v86 = (uint16_t) v85;
    v25[3] = v86;
    uint32_t v87 = v82 >> 9u;
    int v88 = (int) v87;
    int v89 = v88 * 8;
    uint16_t v90 = (uint16_t) v89;
    v26[3] = v90;
    const uint8_t* v91 = v19 + 8;
    const uint8_t v92 = v21[1];
    int v93 = (int) v92;
    int v94 = v93 & 15;
    int v95 = v93 >> 4;
    int v96 = v94 * 2;
    int v97 = v96 + 1;
    int v98 = v95 * 2;
    int v99 = v98 + 1;
    const uint8_t v100 = v91[0];
    uint32_t v101 = (uint32_t) v100;
    const uint8_t v102 = v91[1];
    uint32_t v103 = (uint32_t) v102;
    uint32_t v104 = v103 << 8u;
    uint32_t v105 = v101 | v104;
    uint32_t v106 = v105 & 511u;
    int v107 = (int) v106;
    int v108 = v107 * 8;
    uint16_t v109 = (uint16_t) v108;
    v25[4] = v109;
    uint32_t v110 = v105 >> 9u;
    int v111 = (int) v110;
    int v112 = v111 * 8;
    uint16_t v113 = (uint16_t) v112;
    v26[4] = v113;
    const uint8_t v114 = v91[2];
    uint32_t v115 = (uint32_t) v114;
    const uint8_t v116 = v91[3];
    uint32_t v117 = (uint32_t) v116;
    uint32_t v118 = v117 << 8u;
    uint32_t v119 = v115 | v118;
    uint32_t v120 = v119 & 511u;
    int v121 = (int) v120;
    int v122 = v121 * 8;
    uint16_t v123 = (uint16_t) v122;
    v25[5] = v123;
    uint32_t v124 = v119 >> 9u;
    int v125 = (int) v124;
    int v126 = v125 * 8;
    uint16_t v127 = (uint16_t) v126;
    v26[5] = v127;
    const uint8_t v128 = v91[4];
    uint32_t v129 = (uint32_t) v128;
    const uint8_t v130 = v91[5];
    uint32_t v131 = (uint32_t) v130;
    uint32_t v132 = v131 << 8u;
    uint32_t v133 = v129 | v132;
    uint32_t v134 = v133 & 511u;
    int v135 = (int) v134;
    int v136 = v135 * 8;
    uint16_t v137 = (uint16_t) v136;
    v25[6] = v137;
    uint32_t v138 = v133 >> 9u;
    int v139 = (int) v138;
    int v140 = v139 * 8;
    uint16_t v141 = (uint16_t) v140;
    v26[6] = v141;
    const uint8_t v142 = v91[6];
    uint32_t v143 = (uint32_t) v142;
    const uint8_t v144 = v91[7];
    uint32_t v145 = (uint32_t) v144;
    uint32_t v146 = v145 << 8u;
    uint32_t v147 = v143 | v146;
    uint32_t v148 = v147 & 511u;
    int v149 = (int) v148;
    int v150 = v149 * 8;
    uint16_t v151 = (uint16_t) v150;
    v25[7] = v151;
    uint32_t v152 = v147 >> 9u;
    int v153 = (int) v152;
    int v154 = v153 * 8;
    uint16_t v155 = (uint16_t) v154;
    v26[7] = v155;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v156 = &v25[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v157 = __riscv_vle16_v_u16m1(v156, 8);
    uint16_t* v158 = &v26[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v159 = __riscv_vle16_v_u16m1(v158, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v160 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v157, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v161 = __riscv_vreinterpret_v_i64m4_i8m4(v160);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v162 = __riscv_vluxei16_v_i64m4(v8, v159, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v163 = __riscv_vreinterpret_v_i64m4_i8m4(v162);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v164 = __riscv_vle8_v_i8m4(v23, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v165 = __riscv_vmul_vv_i8m4(v161, v163, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v166 = __riscv_vget_v_i8m4_i8m1(v165, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v167 = __riscv_vget_v_i8m4_i8m1(v164, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v168 = __riscv_vwmul_vv_i16m2(v166, v167, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v169 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v170 = __riscv_vwredsum_vs_i16m2_i32m1(v168, v169, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v171 = __riscv_vmv_x_s_i32m1_i32(v170);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v172 = v24;
    int32_t v173 = (int32_t) v32;
    int32_t v174 = v171 * v173;
    int32_t v175 = v172 + v174;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v175;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v176 = __riscv_vget_v_i8m4_i8m1(v165, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v177 = __riscv_vget_v_i8m4_i8m1(v164, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v178 = __riscv_vwmul_vv_i16m2(v176, v177, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v179 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v180 = __riscv_vwredsum_vs_i16m2_i32m1(v178, v179, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v181 = __riscv_vmv_x_s_i32m1_i32(v180);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v182 = v24;
    int32_t v183 = (int32_t) v34;
    int32_t v184 = v181 * v183;
    int32_t v185 = v182 + v184;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v185;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v186 = __riscv_vget_v_i8m4_i8m1(v165, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v187 = __riscv_vget_v_i8m4_i8m1(v164, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v188 = __riscv_vwmul_vv_i16m2(v186, v187, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v189 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v190 = __riscv_vwredsum_vs_i16m2_i32m1(v188, v189, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v191 = __riscv_vmv_x_s_i32m1_i32(v190);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v192 = v24;
    int32_t v193 = (int32_t) v97;
    int32_t v194 = v191 * v193;
    int32_t v195 = v192 + v194;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v195;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v196 = __riscv_vget_v_i8m4_i8m1(v165, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v197 = __riscv_vget_v_i8m4_i8m1(v164, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v198 = __riscv_vwmul_vv_i16m2(v196, v197, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v199 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v200 = __riscv_vwredsum_vs_i16m2_i32m1(v198, v199, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v201 = __riscv_vmv_x_s_i32m1_i32(v200);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v202 = v24;
    int32_t v203 = (int32_t) v99;
    int32_t v204 = v201 * v203;
    int32_t v205 = v202 + v204;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v205;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v206[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v207[8];
    const uint8_t* v208 = v19 + 16;
    const uint8_t v209 = v21[2];
    int v210 = (int) v209;
    int v211 = v210 & 15;
    int v212 = v210 >> 4;
    int v213 = v211 * 2;
    int v214 = v213 + 1;
    int v215 = v212 * 2;
    int v216 = v215 + 1;
    const uint8_t v217 = v208[0];
    uint32_t v218 = (uint32_t) v217;
    const uint8_t v219 = v208[1];
    uint32_t v220 = (uint32_t) v219;
    uint32_t v221 = v220 << 8u;
    uint32_t v222 = v218 | v221;
    uint32_t v223 = v222 & 511u;
    int v224 = (int) v223;
    int v225 = v224 * 8;
    uint16_t v226 = (uint16_t) v225;
    v206[0] = v226;
    uint32_t v227 = v222 >> 9u;
    int v228 = (int) v227;
    int v229 = v228 * 8;
    uint16_t v230 = (uint16_t) v229;
    v207[0] = v230;
    const uint8_t v231 = v208[2];
    uint32_t v232 = (uint32_t) v231;
    const uint8_t v233 = v208[3];
    uint32_t v234 = (uint32_t) v233;
    uint32_t v235 = v234 << 8u;
    uint32_t v236 = v232 | v235;
    uint32_t v237 = v236 & 511u;
    int v238 = (int) v237;
    int v239 = v238 * 8;
    uint16_t v240 = (uint16_t) v239;
    v206[1] = v240;
    uint32_t v241 = v236 >> 9u;
    int v242 = (int) v241;
    int v243 = v242 * 8;
    uint16_t v244 = (uint16_t) v243;
    v207[1] = v244;
    const uint8_t v245 = v208[4];
    uint32_t v246 = (uint32_t) v245;
    const uint8_t v247 = v208[5];
    uint32_t v248 = (uint32_t) v247;
    uint32_t v249 = v248 << 8u;
    uint32_t v250 = v246 | v249;
    uint32_t v251 = v250 & 511u;
    int v252 = (int) v251;
    int v253 = v252 * 8;
    uint16_t v254 = (uint16_t) v253;
    v206[2] = v254;
    uint32_t v255 = v250 >> 9u;
    int v256 = (int) v255;
    int v257 = v256 * 8;
    uint16_t v258 = (uint16_t) v257;
    v207[2] = v258;
    const uint8_t v259 = v208[6];
    uint32_t v260 = (uint32_t) v259;
    const uint8_t v261 = v208[7];
    uint32_t v262 = (uint32_t) v261;
    uint32_t v263 = v262 << 8u;
    uint32_t v264 = v260 | v263;
    uint32_t v265 = v264 & 511u;
    int v266 = (int) v265;
    int v267 = v266 * 8;
    uint16_t v268 = (uint16_t) v267;
    v206[3] = v268;
    uint32_t v269 = v264 >> 9u;
    int v270 = (int) v269;
    int v271 = v270 * 8;
    uint16_t v272 = (uint16_t) v271;
    v207[3] = v272;
    const uint8_t* v273 = v19 + 24;
    const uint8_t v274 = v21[3];
    int v275 = (int) v274;
    int v276 = v275 & 15;
    int v277 = v275 >> 4;
    int v278 = v276 * 2;
    int v279 = v278 + 1;
    int v280 = v277 * 2;
    int v281 = v280 + 1;
    const uint8_t v282 = v273[0];
    uint32_t v283 = (uint32_t) v282;
    const uint8_t v284 = v273[1];
    uint32_t v285 = (uint32_t) v284;
    uint32_t v286 = v285 << 8u;
    uint32_t v287 = v283 | v286;
    uint32_t v288 = v287 & 511u;
    int v289 = (int) v288;
    int v290 = v289 * 8;
    uint16_t v291 = (uint16_t) v290;
    v206[4] = v291;
    uint32_t v292 = v287 >> 9u;
    int v293 = (int) v292;
    int v294 = v293 * 8;
    uint16_t v295 = (uint16_t) v294;
    v207[4] = v295;
    const uint8_t v296 = v273[2];
    uint32_t v297 = (uint32_t) v296;
    const uint8_t v298 = v273[3];
    uint32_t v299 = (uint32_t) v298;
    uint32_t v300 = v299 << 8u;
    uint32_t v301 = v297 | v300;
    uint32_t v302 = v301 & 511u;
    int v303 = (int) v302;
    int v304 = v303 * 8;
    uint16_t v305 = (uint16_t) v304;
    v206[5] = v305;
    uint32_t v306 = v301 >> 9u;
    int v307 = (int) v306;
    int v308 = v307 * 8;
    uint16_t v309 = (uint16_t) v308;
    v207[5] = v309;
    const uint8_t v310 = v273[4];
    uint32_t v311 = (uint32_t) v310;
    const uint8_t v312 = v273[5];
    uint32_t v313 = (uint32_t) v312;
    uint32_t v314 = v313 << 8u;
    uint32_t v315 = v311 | v314;
    uint32_t v316 = v315 & 511u;
    int v317 = (int) v316;
    int v318 = v317 * 8;
    uint16_t v319 = (uint16_t) v318;
    v206[6] = v319;
    uint32_t v320 = v315 >> 9u;
    int v321 = (int) v320;
    int v322 = v321 * 8;
    uint16_t v323 = (uint16_t) v322;
    v207[6] = v323;
    const uint8_t v324 = v273[6];
    uint32_t v325 = (uint32_t) v324;
    const uint8_t v326 = v273[7];
    uint32_t v327 = (uint32_t) v326;
    uint32_t v328 = v327 << 8u;
    uint32_t v329 = v325 | v328;
    uint32_t v330 = v329 & 511u;
    int v331 = (int) v330;
    int v332 = v331 * 8;
    uint16_t v333 = (uint16_t) v332;
    v206[7] = v333;
    uint32_t v334 = v329 >> 9u;
    int v335 = (int) v334;
    int v336 = v335 * 8;
    uint16_t v337 = (uint16_t) v336;
    v207[7] = v337;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v338 = &v206[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v339 = __riscv_vle16_v_u16m1(v338, 8);
    uint16_t* v340 = &v207[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v341 = __riscv_vle16_v_u16m1(v340, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v342 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v339, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v343 = __riscv_vreinterpret_v_i64m4_i8m4(v342);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v344 = __riscv_vluxei16_v_i64m4(v8, v341, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v345 = __riscv_vreinterpret_v_i64m4_i8m4(v344);
    const int8_t* v346 = v23 + 64;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v347 = __riscv_vle8_v_i8m4(v346, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v348 = __riscv_vmul_vv_i8m4(v343, v345, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v349 = __riscv_vget_v_i8m4_i8m1(v348, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v350 = __riscv_vget_v_i8m4_i8m1(v347, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v351 = __riscv_vwmul_vv_i16m2(v349, v350, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v352 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v353 = __riscv_vwredsum_vs_i16m2_i32m1(v351, v352, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v354 = __riscv_vmv_x_s_i32m1_i32(v353);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v355 = v24;
    int32_t v356 = (int32_t) v214;
    int32_t v357 = v354 * v356;
    int32_t v358 = v355 + v357;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v358;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v359 = __riscv_vget_v_i8m4_i8m1(v348, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v360 = __riscv_vget_v_i8m4_i8m1(v347, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v361 = __riscv_vwmul_vv_i16m2(v359, v360, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v362 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v363 = __riscv_vwredsum_vs_i16m2_i32m1(v361, v362, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v364 = __riscv_vmv_x_s_i32m1_i32(v363);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v365 = v24;
    int32_t v366 = (int32_t) v216;
    int32_t v367 = v364 * v366;
    int32_t v368 = v365 + v367;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v368;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v369 = __riscv_vget_v_i8m4_i8m1(v348, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v370 = __riscv_vget_v_i8m4_i8m1(v347, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v371 = __riscv_vwmul_vv_i16m2(v369, v370, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v372 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v373 = __riscv_vwredsum_vs_i16m2_i32m1(v371, v372, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v374 = __riscv_vmv_x_s_i32m1_i32(v373);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v375 = v24;
    int32_t v376 = (int32_t) v279;
    int32_t v377 = v374 * v376;
    int32_t v378 = v375 + v377;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v378;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v379 = __riscv_vget_v_i8m4_i8m1(v348, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v380 = __riscv_vget_v_i8m4_i8m1(v347, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v381 = __riscv_vwmul_vv_i16m2(v379, v380, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v382 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v383 = __riscv_vwredsum_vs_i16m2_i32m1(v381, v382, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v384 = __riscv_vmv_x_s_i32m1_i32(v383);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v385 = v24;
    int32_t v386 = (int32_t) v281;
    int32_t v387 = v384 * v386;
    int32_t v388 = v385 + v387;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v388;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v389[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v390[8];
    const uint8_t* v391 = v19 + 32;
    const uint8_t v392 = v21[4];
    int v393 = (int) v392;
    int v394 = v393 & 15;
    int v395 = v393 >> 4;
    int v396 = v394 * 2;
    int v397 = v396 + 1;
    int v398 = v395 * 2;
    int v399 = v398 + 1;
    const uint8_t v400 = v391[0];
    uint32_t v401 = (uint32_t) v400;
    const uint8_t v402 = v391[1];
    uint32_t v403 = (uint32_t) v402;
    uint32_t v404 = v403 << 8u;
    uint32_t v405 = v401 | v404;
    uint32_t v406 = v405 & 511u;
    int v407 = (int) v406;
    int v408 = v407 * 8;
    uint16_t v409 = (uint16_t) v408;
    v389[0] = v409;
    uint32_t v410 = v405 >> 9u;
    int v411 = (int) v410;
    int v412 = v411 * 8;
    uint16_t v413 = (uint16_t) v412;
    v390[0] = v413;
    const uint8_t v414 = v391[2];
    uint32_t v415 = (uint32_t) v414;
    const uint8_t v416 = v391[3];
    uint32_t v417 = (uint32_t) v416;
    uint32_t v418 = v417 << 8u;
    uint32_t v419 = v415 | v418;
    uint32_t v420 = v419 & 511u;
    int v421 = (int) v420;
    int v422 = v421 * 8;
    uint16_t v423 = (uint16_t) v422;
    v389[1] = v423;
    uint32_t v424 = v419 >> 9u;
    int v425 = (int) v424;
    int v426 = v425 * 8;
    uint16_t v427 = (uint16_t) v426;
    v390[1] = v427;
    const uint8_t v428 = v391[4];
    uint32_t v429 = (uint32_t) v428;
    const uint8_t v430 = v391[5];
    uint32_t v431 = (uint32_t) v430;
    uint32_t v432 = v431 << 8u;
    uint32_t v433 = v429 | v432;
    uint32_t v434 = v433 & 511u;
    int v435 = (int) v434;
    int v436 = v435 * 8;
    uint16_t v437 = (uint16_t) v436;
    v389[2] = v437;
    uint32_t v438 = v433 >> 9u;
    int v439 = (int) v438;
    int v440 = v439 * 8;
    uint16_t v441 = (uint16_t) v440;
    v390[2] = v441;
    const uint8_t v442 = v391[6];
    uint32_t v443 = (uint32_t) v442;
    const uint8_t v444 = v391[7];
    uint32_t v445 = (uint32_t) v444;
    uint32_t v446 = v445 << 8u;
    uint32_t v447 = v443 | v446;
    uint32_t v448 = v447 & 511u;
    int v449 = (int) v448;
    int v450 = v449 * 8;
    uint16_t v451 = (uint16_t) v450;
    v389[3] = v451;
    uint32_t v452 = v447 >> 9u;
    int v453 = (int) v452;
    int v454 = v453 * 8;
    uint16_t v455 = (uint16_t) v454;
    v390[3] = v455;
    const uint8_t* v456 = v19 + 40;
    const uint8_t v457 = v21[5];
    int v458 = (int) v457;
    int v459 = v458 & 15;
    int v460 = v458 >> 4;
    int v461 = v459 * 2;
    int v462 = v461 + 1;
    int v463 = v460 * 2;
    int v464 = v463 + 1;
    const uint8_t v465 = v456[0];
    uint32_t v466 = (uint32_t) v465;
    const uint8_t v467 = v456[1];
    uint32_t v468 = (uint32_t) v467;
    uint32_t v469 = v468 << 8u;
    uint32_t v470 = v466 | v469;
    uint32_t v471 = v470 & 511u;
    int v472 = (int) v471;
    int v473 = v472 * 8;
    uint16_t v474 = (uint16_t) v473;
    v389[4] = v474;
    uint32_t v475 = v470 >> 9u;
    int v476 = (int) v475;
    int v477 = v476 * 8;
    uint16_t v478 = (uint16_t) v477;
    v390[4] = v478;
    const uint8_t v479 = v456[2];
    uint32_t v480 = (uint32_t) v479;
    const uint8_t v481 = v456[3];
    uint32_t v482 = (uint32_t) v481;
    uint32_t v483 = v482 << 8u;
    uint32_t v484 = v480 | v483;
    uint32_t v485 = v484 & 511u;
    int v486 = (int) v485;
    int v487 = v486 * 8;
    uint16_t v488 = (uint16_t) v487;
    v389[5] = v488;
    uint32_t v489 = v484 >> 9u;
    int v490 = (int) v489;
    int v491 = v490 * 8;
    uint16_t v492 = (uint16_t) v491;
    v390[5] = v492;
    const uint8_t v493 = v456[4];
    uint32_t v494 = (uint32_t) v493;
    const uint8_t v495 = v456[5];
    uint32_t v496 = (uint32_t) v495;
    uint32_t v497 = v496 << 8u;
    uint32_t v498 = v494 | v497;
    uint32_t v499 = v498 & 511u;
    int v500 = (int) v499;
    int v501 = v500 * 8;
    uint16_t v502 = (uint16_t) v501;
    v389[6] = v502;
    uint32_t v503 = v498 >> 9u;
    int v504 = (int) v503;
    int v505 = v504 * 8;
    uint16_t v506 = (uint16_t) v505;
    v390[6] = v506;
    const uint8_t v507 = v456[6];
    uint32_t v508 = (uint32_t) v507;
    const uint8_t v509 = v456[7];
    uint32_t v510 = (uint32_t) v509;
    uint32_t v511 = v510 << 8u;
    uint32_t v512 = v508 | v511;
    uint32_t v513 = v512 & 511u;
    int v514 = (int) v513;
    int v515 = v514 * 8;
    uint16_t v516 = (uint16_t) v515;
    v389[7] = v516;
    uint32_t v517 = v512 >> 9u;
    int v518 = (int) v517;
    int v519 = v518 * 8;
    uint16_t v520 = (uint16_t) v519;
    v390[7] = v520;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v521 = &v389[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v522 = __riscv_vle16_v_u16m1(v521, 8);
    uint16_t* v523 = &v390[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v524 = __riscv_vle16_v_u16m1(v523, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v525 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v522, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v526 = __riscv_vreinterpret_v_i64m4_i8m4(v525);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v527 = __riscv_vluxei16_v_i64m4(v8, v524, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v528 = __riscv_vreinterpret_v_i64m4_i8m4(v527);
    const int8_t* v529 = v23 + 128;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v530 = __riscv_vle8_v_i8m4(v529, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v531 = __riscv_vmul_vv_i8m4(v526, v528, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v532 = __riscv_vget_v_i8m4_i8m1(v531, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v533 = __riscv_vget_v_i8m4_i8m1(v530, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v534 = __riscv_vwmul_vv_i16m2(v532, v533, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v535 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v536 = __riscv_vwredsum_vs_i16m2_i32m1(v534, v535, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v537 = __riscv_vmv_x_s_i32m1_i32(v536);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v538 = v24;
    int32_t v539 = (int32_t) v397;
    int32_t v540 = v537 * v539;
    int32_t v541 = v538 + v540;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v541;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v542 = __riscv_vget_v_i8m4_i8m1(v531, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v543 = __riscv_vget_v_i8m4_i8m1(v530, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v544 = __riscv_vwmul_vv_i16m2(v542, v543, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v545 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v546 = __riscv_vwredsum_vs_i16m2_i32m1(v544, v545, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v547 = __riscv_vmv_x_s_i32m1_i32(v546);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v548 = v24;
    int32_t v549 = (int32_t) v399;
    int32_t v550 = v547 * v549;
    int32_t v551 = v548 + v550;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v551;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v552 = __riscv_vget_v_i8m4_i8m1(v531, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v553 = __riscv_vget_v_i8m4_i8m1(v530, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v554 = __riscv_vwmul_vv_i16m2(v552, v553, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v555 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v556 = __riscv_vwredsum_vs_i16m2_i32m1(v554, v555, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v557 = __riscv_vmv_x_s_i32m1_i32(v556);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v558 = v24;
    int32_t v559 = (int32_t) v462;
    int32_t v560 = v557 * v559;
    int32_t v561 = v558 + v560;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v561;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v562 = __riscv_vget_v_i8m4_i8m1(v531, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v563 = __riscv_vget_v_i8m4_i8m1(v530, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v564 = __riscv_vwmul_vv_i16m2(v562, v563, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v565 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v566 = __riscv_vwredsum_vs_i16m2_i32m1(v564, v565, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v567 = __riscv_vmv_x_s_i32m1_i32(v566);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v568 = v24;
    int32_t v569 = (int32_t) v464;
    int32_t v570 = v567 * v569;
    int32_t v571 = v568 + v570;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v571;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v572[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v573[8];
    const uint8_t* v574 = v19 + 48;
    const uint8_t v575 = v21[6];
    int v576 = (int) v575;
    int v577 = v576 & 15;
    int v578 = v576 >> 4;
    int v579 = v577 * 2;
    int v580 = v579 + 1;
    int v581 = v578 * 2;
    int v582 = v581 + 1;
    const uint8_t v583 = v574[0];
    uint32_t v584 = (uint32_t) v583;
    const uint8_t v585 = v574[1];
    uint32_t v586 = (uint32_t) v585;
    uint32_t v587 = v586 << 8u;
    uint32_t v588 = v584 | v587;
    uint32_t v589 = v588 & 511u;
    int v590 = (int) v589;
    int v591 = v590 * 8;
    uint16_t v592 = (uint16_t) v591;
    v572[0] = v592;
    uint32_t v593 = v588 >> 9u;
    int v594 = (int) v593;
    int v595 = v594 * 8;
    uint16_t v596 = (uint16_t) v595;
    v573[0] = v596;
    const uint8_t v597 = v574[2];
    uint32_t v598 = (uint32_t) v597;
    const uint8_t v599 = v574[3];
    uint32_t v600 = (uint32_t) v599;
    uint32_t v601 = v600 << 8u;
    uint32_t v602 = v598 | v601;
    uint32_t v603 = v602 & 511u;
    int v604 = (int) v603;
    int v605 = v604 * 8;
    uint16_t v606 = (uint16_t) v605;
    v572[1] = v606;
    uint32_t v607 = v602 >> 9u;
    int v608 = (int) v607;
    int v609 = v608 * 8;
    uint16_t v610 = (uint16_t) v609;
    v573[1] = v610;
    const uint8_t v611 = v574[4];
    uint32_t v612 = (uint32_t) v611;
    const uint8_t v613 = v574[5];
    uint32_t v614 = (uint32_t) v613;
    uint32_t v615 = v614 << 8u;
    uint32_t v616 = v612 | v615;
    uint32_t v617 = v616 & 511u;
    int v618 = (int) v617;
    int v619 = v618 * 8;
    uint16_t v620 = (uint16_t) v619;
    v572[2] = v620;
    uint32_t v621 = v616 >> 9u;
    int v622 = (int) v621;
    int v623 = v622 * 8;
    uint16_t v624 = (uint16_t) v623;
    v573[2] = v624;
    const uint8_t v625 = v574[6];
    uint32_t v626 = (uint32_t) v625;
    const uint8_t v627 = v574[7];
    uint32_t v628 = (uint32_t) v627;
    uint32_t v629 = v628 << 8u;
    uint32_t v630 = v626 | v629;
    uint32_t v631 = v630 & 511u;
    int v632 = (int) v631;
    int v633 = v632 * 8;
    uint16_t v634 = (uint16_t) v633;
    v572[3] = v634;
    uint32_t v635 = v630 >> 9u;
    int v636 = (int) v635;
    int v637 = v636 * 8;
    uint16_t v638 = (uint16_t) v637;
    v573[3] = v638;
    const uint8_t* v639 = v19 + 56;
    const uint8_t v640 = v21[7];
    int v641 = (int) v640;
    int v642 = v641 & 15;
    int v643 = v641 >> 4;
    int v644 = v642 * 2;
    int v645 = v644 + 1;
    int v646 = v643 * 2;
    int v647 = v646 + 1;
    const uint8_t v648 = v639[0];
    uint32_t v649 = (uint32_t) v648;
    const uint8_t v650 = v639[1];
    uint32_t v651 = (uint32_t) v650;
    uint32_t v652 = v651 << 8u;
    uint32_t v653 = v649 | v652;
    uint32_t v654 = v653 & 511u;
    int v655 = (int) v654;
    int v656 = v655 * 8;
    uint16_t v657 = (uint16_t) v656;
    v572[4] = v657;
    uint32_t v658 = v653 >> 9u;
    int v659 = (int) v658;
    int v660 = v659 * 8;
    uint16_t v661 = (uint16_t) v660;
    v573[4] = v661;
    const uint8_t v662 = v639[2];
    uint32_t v663 = (uint32_t) v662;
    const uint8_t v664 = v639[3];
    uint32_t v665 = (uint32_t) v664;
    uint32_t v666 = v665 << 8u;
    uint32_t v667 = v663 | v666;
    uint32_t v668 = v667 & 511u;
    int v669 = (int) v668;
    int v670 = v669 * 8;
    uint16_t v671 = (uint16_t) v670;
    v572[5] = v671;
    uint32_t v672 = v667 >> 9u;
    int v673 = (int) v672;
    int v674 = v673 * 8;
    uint16_t v675 = (uint16_t) v674;
    v573[5] = v675;
    const uint8_t v676 = v639[4];
    uint32_t v677 = (uint32_t) v676;
    const uint8_t v678 = v639[5];
    uint32_t v679 = (uint32_t) v678;
    uint32_t v680 = v679 << 8u;
    uint32_t v681 = v677 | v680;
    uint32_t v682 = v681 & 511u;
    int v683 = (int) v682;
    int v684 = v683 * 8;
    uint16_t v685 = (uint16_t) v684;
    v572[6] = v685;
    uint32_t v686 = v681 >> 9u;
    int v687 = (int) v686;
    int v688 = v687 * 8;
    uint16_t v689 = (uint16_t) v688;
    v573[6] = v689;
    const uint8_t v690 = v639[6];
    uint32_t v691 = (uint32_t) v690;
    const uint8_t v692 = v639[7];
    uint32_t v693 = (uint32_t) v692;
    uint32_t v694 = v693 << 8u;
    uint32_t v695 = v691 | v694;
    uint32_t v696 = v695 & 511u;
    int v697 = (int) v696;
    int v698 = v697 * 8;
    uint16_t v699 = (uint16_t) v698;
    v572[7] = v699;
    uint32_t v700 = v695 >> 9u;
    int v701 = (int) v700;
    int v702 = v701 * 8;
    uint16_t v703 = (uint16_t) v702;
    v573[7] = v703;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v704 = &v572[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v705 = __riscv_vle16_v_u16m1(v704, 8);
    uint16_t* v706 = &v573[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v707 = __riscv_vle16_v_u16m1(v706, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v708 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v705, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v709 = __riscv_vreinterpret_v_i64m4_i8m4(v708);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v710 = __riscv_vluxei16_v_i64m4(v8, v707, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v711 = __riscv_vreinterpret_v_i64m4_i8m4(v710);
    const int8_t* v712 = v23 + 192;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v713 = __riscv_vle8_v_i8m4(v712, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v714 = __riscv_vmul_vv_i8m4(v709, v711, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v715 = __riscv_vget_v_i8m4_i8m1(v714, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v716 = __riscv_vget_v_i8m4_i8m1(v713, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v717 = __riscv_vwmul_vv_i16m2(v715, v716, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v718 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v719 = __riscv_vwredsum_vs_i16m2_i32m1(v717, v718, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v720 = __riscv_vmv_x_s_i32m1_i32(v719);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v721 = v24;
    int32_t v722 = (int32_t) v580;
    int32_t v723 = v720 * v722;
    int32_t v724 = v721 + v723;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v724;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v725 = __riscv_vget_v_i8m4_i8m1(v714, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v726 = __riscv_vget_v_i8m4_i8m1(v713, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v727 = __riscv_vwmul_vv_i16m2(v725, v726, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v728 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v729 = __riscv_vwredsum_vs_i16m2_i32m1(v727, v728, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v730 = __riscv_vmv_x_s_i32m1_i32(v729);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v731 = v24;
    int32_t v732 = (int32_t) v582;
    int32_t v733 = v730 * v732;
    int32_t v734 = v731 + v733;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v734;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v735 = __riscv_vget_v_i8m4_i8m1(v714, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v736 = __riscv_vget_v_i8m4_i8m1(v713, 2);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v737 = __riscv_vwmul_vv_i16m2(v735, v736, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v738 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v739 = __riscv_vwredsum_vs_i16m2_i32m1(v737, v738, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v740 = __riscv_vmv_x_s_i32m1_i32(v739);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v741 = v24;
    int32_t v742 = (int32_t) v645;
    int32_t v743 = v740 * v742;
    int32_t v744 = v741 + v743;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v744;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v745 = __riscv_vget_v_i8m4_i8m1(v714, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v746 = __riscv_vget_v_i8m4_i8m1(v713, 3);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v747 = __riscv_vwmul_vv_i16m2(v745, v746, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v748 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v749 = __riscv_vwredsum_vs_i16m2_i32m1(v747, v748, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v750 = __riscv_vmv_x_s_i32m1_i32(v749);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v751 = v24;
    int32_t v752 = (int32_t) v647;
    int32_t v753 = v750 * v752;
    int32_t v754 = v751 + v753;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v754;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v755 = v24;
    float v756 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v756 + v17 * (float) v755;
  }
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v757 = v6;
  float v758 = 0.125f * v757;
  v2[0] = v758;
  return;
}


