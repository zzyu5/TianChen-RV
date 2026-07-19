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
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v176 = __riscv_vslidedown_vx_i8m4(v165, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v177 = __riscv_vslidedown_vx_i8m4(v164, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v178 = __riscv_vget_v_i8m4_i8m1(v176, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v179 = __riscv_vget_v_i8m4_i8m1(v177, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v180 = __riscv_vwmul_vv_i16m2(v178, v179, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v181 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v182 = __riscv_vwredsum_vs_i16m2_i32m1(v180, v181, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v183 = __riscv_vmv_x_s_i32m1_i32(v182);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v184 = v24;
    int32_t v185 = (int32_t) v34;
    int32_t v186 = v183 * v185;
    int32_t v187 = v184 + v186;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v187;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v188 = __riscv_vslidedown_vx_i8m4(v165, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v189 = __riscv_vslidedown_vx_i8m4(v164, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v190 = __riscv_vget_v_i8m4_i8m1(v188, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v191 = __riscv_vget_v_i8m4_i8m1(v189, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v192 = __riscv_vwmul_vv_i16m2(v190, v191, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v193 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v194 = __riscv_vwredsum_vs_i16m2_i32m1(v192, v193, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v195 = __riscv_vmv_x_s_i32m1_i32(v194);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v196 = v24;
    int32_t v197 = (int32_t) v97;
    int32_t v198 = v195 * v197;
    int32_t v199 = v196 + v198;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v199;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v200 = __riscv_vslidedown_vx_i8m4(v165, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v201 = __riscv_vslidedown_vx_i8m4(v164, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v202 = __riscv_vget_v_i8m4_i8m1(v200, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v203 = __riscv_vget_v_i8m4_i8m1(v201, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v204 = __riscv_vwmul_vv_i16m2(v202, v203, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v205 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v206 = __riscv_vwredsum_vs_i16m2_i32m1(v204, v205, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v207 = __riscv_vmv_x_s_i32m1_i32(v206);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v208 = v24;
    int32_t v209 = (int32_t) v99;
    int32_t v210 = v207 * v209;
    int32_t v211 = v208 + v210;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v211;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v212[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v213[8];
    const uint8_t* v214 = v19 + 16;
    const uint8_t v215 = v21[2];
    int v216 = (int) v215;
    int v217 = v216 & 15;
    int v218 = v216 >> 4;
    int v219 = v217 * 2;
    int v220 = v219 + 1;
    int v221 = v218 * 2;
    int v222 = v221 + 1;
    const uint8_t v223 = v214[0];
    uint32_t v224 = (uint32_t) v223;
    const uint8_t v225 = v214[1];
    uint32_t v226 = (uint32_t) v225;
    uint32_t v227 = v226 << 8u;
    uint32_t v228 = v224 | v227;
    uint32_t v229 = v228 & 511u;
    int v230 = (int) v229;
    int v231 = v230 * 8;
    uint16_t v232 = (uint16_t) v231;
    v212[0] = v232;
    uint32_t v233 = v228 >> 9u;
    int v234 = (int) v233;
    int v235 = v234 * 8;
    uint16_t v236 = (uint16_t) v235;
    v213[0] = v236;
    const uint8_t v237 = v214[2];
    uint32_t v238 = (uint32_t) v237;
    const uint8_t v239 = v214[3];
    uint32_t v240 = (uint32_t) v239;
    uint32_t v241 = v240 << 8u;
    uint32_t v242 = v238 | v241;
    uint32_t v243 = v242 & 511u;
    int v244 = (int) v243;
    int v245 = v244 * 8;
    uint16_t v246 = (uint16_t) v245;
    v212[1] = v246;
    uint32_t v247 = v242 >> 9u;
    int v248 = (int) v247;
    int v249 = v248 * 8;
    uint16_t v250 = (uint16_t) v249;
    v213[1] = v250;
    const uint8_t v251 = v214[4];
    uint32_t v252 = (uint32_t) v251;
    const uint8_t v253 = v214[5];
    uint32_t v254 = (uint32_t) v253;
    uint32_t v255 = v254 << 8u;
    uint32_t v256 = v252 | v255;
    uint32_t v257 = v256 & 511u;
    int v258 = (int) v257;
    int v259 = v258 * 8;
    uint16_t v260 = (uint16_t) v259;
    v212[2] = v260;
    uint32_t v261 = v256 >> 9u;
    int v262 = (int) v261;
    int v263 = v262 * 8;
    uint16_t v264 = (uint16_t) v263;
    v213[2] = v264;
    const uint8_t v265 = v214[6];
    uint32_t v266 = (uint32_t) v265;
    const uint8_t v267 = v214[7];
    uint32_t v268 = (uint32_t) v267;
    uint32_t v269 = v268 << 8u;
    uint32_t v270 = v266 | v269;
    uint32_t v271 = v270 & 511u;
    int v272 = (int) v271;
    int v273 = v272 * 8;
    uint16_t v274 = (uint16_t) v273;
    v212[3] = v274;
    uint32_t v275 = v270 >> 9u;
    int v276 = (int) v275;
    int v277 = v276 * 8;
    uint16_t v278 = (uint16_t) v277;
    v213[3] = v278;
    const uint8_t* v279 = v19 + 24;
    const uint8_t v280 = v21[3];
    int v281 = (int) v280;
    int v282 = v281 & 15;
    int v283 = v281 >> 4;
    int v284 = v282 * 2;
    int v285 = v284 + 1;
    int v286 = v283 * 2;
    int v287 = v286 + 1;
    const uint8_t v288 = v279[0];
    uint32_t v289 = (uint32_t) v288;
    const uint8_t v290 = v279[1];
    uint32_t v291 = (uint32_t) v290;
    uint32_t v292 = v291 << 8u;
    uint32_t v293 = v289 | v292;
    uint32_t v294 = v293 & 511u;
    int v295 = (int) v294;
    int v296 = v295 * 8;
    uint16_t v297 = (uint16_t) v296;
    v212[4] = v297;
    uint32_t v298 = v293 >> 9u;
    int v299 = (int) v298;
    int v300 = v299 * 8;
    uint16_t v301 = (uint16_t) v300;
    v213[4] = v301;
    const uint8_t v302 = v279[2];
    uint32_t v303 = (uint32_t) v302;
    const uint8_t v304 = v279[3];
    uint32_t v305 = (uint32_t) v304;
    uint32_t v306 = v305 << 8u;
    uint32_t v307 = v303 | v306;
    uint32_t v308 = v307 & 511u;
    int v309 = (int) v308;
    int v310 = v309 * 8;
    uint16_t v311 = (uint16_t) v310;
    v212[5] = v311;
    uint32_t v312 = v307 >> 9u;
    int v313 = (int) v312;
    int v314 = v313 * 8;
    uint16_t v315 = (uint16_t) v314;
    v213[5] = v315;
    const uint8_t v316 = v279[4];
    uint32_t v317 = (uint32_t) v316;
    const uint8_t v318 = v279[5];
    uint32_t v319 = (uint32_t) v318;
    uint32_t v320 = v319 << 8u;
    uint32_t v321 = v317 | v320;
    uint32_t v322 = v321 & 511u;
    int v323 = (int) v322;
    int v324 = v323 * 8;
    uint16_t v325 = (uint16_t) v324;
    v212[6] = v325;
    uint32_t v326 = v321 >> 9u;
    int v327 = (int) v326;
    int v328 = v327 * 8;
    uint16_t v329 = (uint16_t) v328;
    v213[6] = v329;
    const uint8_t v330 = v279[6];
    uint32_t v331 = (uint32_t) v330;
    const uint8_t v332 = v279[7];
    uint32_t v333 = (uint32_t) v332;
    uint32_t v334 = v333 << 8u;
    uint32_t v335 = v331 | v334;
    uint32_t v336 = v335 & 511u;
    int v337 = (int) v336;
    int v338 = v337 * 8;
    uint16_t v339 = (uint16_t) v338;
    v212[7] = v339;
    uint32_t v340 = v335 >> 9u;
    int v341 = (int) v340;
    int v342 = v341 * 8;
    uint16_t v343 = (uint16_t) v342;
    v213[7] = v343;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v344 = &v212[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v345 = __riscv_vle16_v_u16m1(v344, 8);
    uint16_t* v346 = &v213[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v347 = __riscv_vle16_v_u16m1(v346, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v348 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v345, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v349 = __riscv_vreinterpret_v_i64m4_i8m4(v348);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v350 = __riscv_vluxei16_v_i64m4(v8, v347, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v351 = __riscv_vreinterpret_v_i64m4_i8m4(v350);
    const int8_t* v352 = v23 + 64;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v353 = __riscv_vle8_v_i8m4(v352, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v354 = __riscv_vmul_vv_i8m4(v349, v351, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v355 = __riscv_vget_v_i8m4_i8m1(v354, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v356 = __riscv_vget_v_i8m4_i8m1(v353, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v357 = __riscv_vwmul_vv_i16m2(v355, v356, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v358 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v359 = __riscv_vwredsum_vs_i16m2_i32m1(v357, v358, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v360 = __riscv_vmv_x_s_i32m1_i32(v359);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v361 = v24;
    int32_t v362 = (int32_t) v220;
    int32_t v363 = v360 * v362;
    int32_t v364 = v361 + v363;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v364;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v365 = __riscv_vslidedown_vx_i8m4(v354, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v366 = __riscv_vslidedown_vx_i8m4(v353, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v367 = __riscv_vget_v_i8m4_i8m1(v365, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v368 = __riscv_vget_v_i8m4_i8m1(v366, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v369 = __riscv_vwmul_vv_i16m2(v367, v368, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v370 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v371 = __riscv_vwredsum_vs_i16m2_i32m1(v369, v370, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v372 = __riscv_vmv_x_s_i32m1_i32(v371);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v373 = v24;
    int32_t v374 = (int32_t) v222;
    int32_t v375 = v372 * v374;
    int32_t v376 = v373 + v375;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v376;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v377 = __riscv_vslidedown_vx_i8m4(v354, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v378 = __riscv_vslidedown_vx_i8m4(v353, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v379 = __riscv_vget_v_i8m4_i8m1(v377, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v380 = __riscv_vget_v_i8m4_i8m1(v378, 0);
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
    int32_t v386 = (int32_t) v285;
    int32_t v387 = v384 * v386;
    int32_t v388 = v385 + v387;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v388;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v389 = __riscv_vslidedown_vx_i8m4(v354, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v390 = __riscv_vslidedown_vx_i8m4(v353, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v391 = __riscv_vget_v_i8m4_i8m1(v389, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v392 = __riscv_vget_v_i8m4_i8m1(v390, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v393 = __riscv_vwmul_vv_i16m2(v391, v392, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v394 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v395 = __riscv_vwredsum_vs_i16m2_i32m1(v393, v394, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v396 = __riscv_vmv_x_s_i32m1_i32(v395);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v397 = v24;
    int32_t v398 = (int32_t) v287;
    int32_t v399 = v396 * v398;
    int32_t v400 = v397 + v399;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v400;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v401[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v402[8];
    const uint8_t* v403 = v19 + 32;
    const uint8_t v404 = v21[4];
    int v405 = (int) v404;
    int v406 = v405 & 15;
    int v407 = v405 >> 4;
    int v408 = v406 * 2;
    int v409 = v408 + 1;
    int v410 = v407 * 2;
    int v411 = v410 + 1;
    const uint8_t v412 = v403[0];
    uint32_t v413 = (uint32_t) v412;
    const uint8_t v414 = v403[1];
    uint32_t v415 = (uint32_t) v414;
    uint32_t v416 = v415 << 8u;
    uint32_t v417 = v413 | v416;
    uint32_t v418 = v417 & 511u;
    int v419 = (int) v418;
    int v420 = v419 * 8;
    uint16_t v421 = (uint16_t) v420;
    v401[0] = v421;
    uint32_t v422 = v417 >> 9u;
    int v423 = (int) v422;
    int v424 = v423 * 8;
    uint16_t v425 = (uint16_t) v424;
    v402[0] = v425;
    const uint8_t v426 = v403[2];
    uint32_t v427 = (uint32_t) v426;
    const uint8_t v428 = v403[3];
    uint32_t v429 = (uint32_t) v428;
    uint32_t v430 = v429 << 8u;
    uint32_t v431 = v427 | v430;
    uint32_t v432 = v431 & 511u;
    int v433 = (int) v432;
    int v434 = v433 * 8;
    uint16_t v435 = (uint16_t) v434;
    v401[1] = v435;
    uint32_t v436 = v431 >> 9u;
    int v437 = (int) v436;
    int v438 = v437 * 8;
    uint16_t v439 = (uint16_t) v438;
    v402[1] = v439;
    const uint8_t v440 = v403[4];
    uint32_t v441 = (uint32_t) v440;
    const uint8_t v442 = v403[5];
    uint32_t v443 = (uint32_t) v442;
    uint32_t v444 = v443 << 8u;
    uint32_t v445 = v441 | v444;
    uint32_t v446 = v445 & 511u;
    int v447 = (int) v446;
    int v448 = v447 * 8;
    uint16_t v449 = (uint16_t) v448;
    v401[2] = v449;
    uint32_t v450 = v445 >> 9u;
    int v451 = (int) v450;
    int v452 = v451 * 8;
    uint16_t v453 = (uint16_t) v452;
    v402[2] = v453;
    const uint8_t v454 = v403[6];
    uint32_t v455 = (uint32_t) v454;
    const uint8_t v456 = v403[7];
    uint32_t v457 = (uint32_t) v456;
    uint32_t v458 = v457 << 8u;
    uint32_t v459 = v455 | v458;
    uint32_t v460 = v459 & 511u;
    int v461 = (int) v460;
    int v462 = v461 * 8;
    uint16_t v463 = (uint16_t) v462;
    v401[3] = v463;
    uint32_t v464 = v459 >> 9u;
    int v465 = (int) v464;
    int v466 = v465 * 8;
    uint16_t v467 = (uint16_t) v466;
    v402[3] = v467;
    const uint8_t* v468 = v19 + 40;
    const uint8_t v469 = v21[5];
    int v470 = (int) v469;
    int v471 = v470 & 15;
    int v472 = v470 >> 4;
    int v473 = v471 * 2;
    int v474 = v473 + 1;
    int v475 = v472 * 2;
    int v476 = v475 + 1;
    const uint8_t v477 = v468[0];
    uint32_t v478 = (uint32_t) v477;
    const uint8_t v479 = v468[1];
    uint32_t v480 = (uint32_t) v479;
    uint32_t v481 = v480 << 8u;
    uint32_t v482 = v478 | v481;
    uint32_t v483 = v482 & 511u;
    int v484 = (int) v483;
    int v485 = v484 * 8;
    uint16_t v486 = (uint16_t) v485;
    v401[4] = v486;
    uint32_t v487 = v482 >> 9u;
    int v488 = (int) v487;
    int v489 = v488 * 8;
    uint16_t v490 = (uint16_t) v489;
    v402[4] = v490;
    const uint8_t v491 = v468[2];
    uint32_t v492 = (uint32_t) v491;
    const uint8_t v493 = v468[3];
    uint32_t v494 = (uint32_t) v493;
    uint32_t v495 = v494 << 8u;
    uint32_t v496 = v492 | v495;
    uint32_t v497 = v496 & 511u;
    int v498 = (int) v497;
    int v499 = v498 * 8;
    uint16_t v500 = (uint16_t) v499;
    v401[5] = v500;
    uint32_t v501 = v496 >> 9u;
    int v502 = (int) v501;
    int v503 = v502 * 8;
    uint16_t v504 = (uint16_t) v503;
    v402[5] = v504;
    const uint8_t v505 = v468[4];
    uint32_t v506 = (uint32_t) v505;
    const uint8_t v507 = v468[5];
    uint32_t v508 = (uint32_t) v507;
    uint32_t v509 = v508 << 8u;
    uint32_t v510 = v506 | v509;
    uint32_t v511 = v510 & 511u;
    int v512 = (int) v511;
    int v513 = v512 * 8;
    uint16_t v514 = (uint16_t) v513;
    v401[6] = v514;
    uint32_t v515 = v510 >> 9u;
    int v516 = (int) v515;
    int v517 = v516 * 8;
    uint16_t v518 = (uint16_t) v517;
    v402[6] = v518;
    const uint8_t v519 = v468[6];
    uint32_t v520 = (uint32_t) v519;
    const uint8_t v521 = v468[7];
    uint32_t v522 = (uint32_t) v521;
    uint32_t v523 = v522 << 8u;
    uint32_t v524 = v520 | v523;
    uint32_t v525 = v524 & 511u;
    int v526 = (int) v525;
    int v527 = v526 * 8;
    uint16_t v528 = (uint16_t) v527;
    v401[7] = v528;
    uint32_t v529 = v524 >> 9u;
    int v530 = (int) v529;
    int v531 = v530 * 8;
    uint16_t v532 = (uint16_t) v531;
    v402[7] = v532;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v533 = &v401[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v534 = __riscv_vle16_v_u16m1(v533, 8);
    uint16_t* v535 = &v402[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v536 = __riscv_vle16_v_u16m1(v535, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v537 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v534, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v538 = __riscv_vreinterpret_v_i64m4_i8m4(v537);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v539 = __riscv_vluxei16_v_i64m4(v8, v536, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v540 = __riscv_vreinterpret_v_i64m4_i8m4(v539);
    const int8_t* v541 = v23 + 128;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v542 = __riscv_vle8_v_i8m4(v541, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v543 = __riscv_vmul_vv_i8m4(v538, v540, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v544 = __riscv_vget_v_i8m4_i8m1(v543, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v545 = __riscv_vget_v_i8m4_i8m1(v542, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v546 = __riscv_vwmul_vv_i16m2(v544, v545, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v547 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v548 = __riscv_vwredsum_vs_i16m2_i32m1(v546, v547, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v549 = __riscv_vmv_x_s_i32m1_i32(v548);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v550 = v24;
    int32_t v551 = (int32_t) v409;
    int32_t v552 = v549 * v551;
    int32_t v553 = v550 + v552;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v553;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v554 = __riscv_vslidedown_vx_i8m4(v543, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v555 = __riscv_vslidedown_vx_i8m4(v542, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v556 = __riscv_vget_v_i8m4_i8m1(v554, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v557 = __riscv_vget_v_i8m4_i8m1(v555, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v558 = __riscv_vwmul_vv_i16m2(v556, v557, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v559 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v560 = __riscv_vwredsum_vs_i16m2_i32m1(v558, v559, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v561 = __riscv_vmv_x_s_i32m1_i32(v560);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v562 = v24;
    int32_t v563 = (int32_t) v411;
    int32_t v564 = v561 * v563;
    int32_t v565 = v562 + v564;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v565;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v566 = __riscv_vslidedown_vx_i8m4(v543, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v567 = __riscv_vslidedown_vx_i8m4(v542, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v568 = __riscv_vget_v_i8m4_i8m1(v566, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v569 = __riscv_vget_v_i8m4_i8m1(v567, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v570 = __riscv_vwmul_vv_i16m2(v568, v569, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v571 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v572 = __riscv_vwredsum_vs_i16m2_i32m1(v570, v571, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v573 = __riscv_vmv_x_s_i32m1_i32(v572);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v574 = v24;
    int32_t v575 = (int32_t) v474;
    int32_t v576 = v573 * v575;
    int32_t v577 = v574 + v576;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v577;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v578 = __riscv_vslidedown_vx_i8m4(v543, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v579 = __riscv_vslidedown_vx_i8m4(v542, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v580 = __riscv_vget_v_i8m4_i8m1(v578, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v581 = __riscv_vget_v_i8m4_i8m1(v579, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v582 = __riscv_vwmul_vv_i16m2(v580, v581, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v583 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v584 = __riscv_vwredsum_vs_i16m2_i32m1(v582, v583, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v585 = __riscv_vmv_x_s_i32m1_i32(v584);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v586 = v24;
    int32_t v587 = (int32_t) v476;
    int32_t v588 = v585 * v587;
    int32_t v589 = v586 + v588;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v589;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_explicit_scales
    // weft_emitc.local_variable=gridoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v590[8];
    // weft_emitc.local_variable=signoff source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    uint16_t v591[8];
    const uint8_t* v592 = v19 + 48;
    const uint8_t v593 = v21[6];
    int v594 = (int) v593;
    int v595 = v594 & 15;
    int v596 = v594 >> 4;
    int v597 = v595 * 2;
    int v598 = v597 + 1;
    int v599 = v596 * 2;
    int v600 = v599 + 1;
    const uint8_t v601 = v592[0];
    uint32_t v602 = (uint32_t) v601;
    const uint8_t v603 = v592[1];
    uint32_t v604 = (uint32_t) v603;
    uint32_t v605 = v604 << 8u;
    uint32_t v606 = v602 | v605;
    uint32_t v607 = v606 & 511u;
    int v608 = (int) v607;
    int v609 = v608 * 8;
    uint16_t v610 = (uint16_t) v609;
    v590[0] = v610;
    uint32_t v611 = v606 >> 9u;
    int v612 = (int) v611;
    int v613 = v612 * 8;
    uint16_t v614 = (uint16_t) v613;
    v591[0] = v614;
    const uint8_t v615 = v592[2];
    uint32_t v616 = (uint32_t) v615;
    const uint8_t v617 = v592[3];
    uint32_t v618 = (uint32_t) v617;
    uint32_t v619 = v618 << 8u;
    uint32_t v620 = v616 | v619;
    uint32_t v621 = v620 & 511u;
    int v622 = (int) v621;
    int v623 = v622 * 8;
    uint16_t v624 = (uint16_t) v623;
    v590[1] = v624;
    uint32_t v625 = v620 >> 9u;
    int v626 = (int) v625;
    int v627 = v626 * 8;
    uint16_t v628 = (uint16_t) v627;
    v591[1] = v628;
    const uint8_t v629 = v592[4];
    uint32_t v630 = (uint32_t) v629;
    const uint8_t v631 = v592[5];
    uint32_t v632 = (uint32_t) v631;
    uint32_t v633 = v632 << 8u;
    uint32_t v634 = v630 | v633;
    uint32_t v635 = v634 & 511u;
    int v636 = (int) v635;
    int v637 = v636 * 8;
    uint16_t v638 = (uint16_t) v637;
    v590[2] = v638;
    uint32_t v639 = v634 >> 9u;
    int v640 = (int) v639;
    int v641 = v640 * 8;
    uint16_t v642 = (uint16_t) v641;
    v591[2] = v642;
    const uint8_t v643 = v592[6];
    uint32_t v644 = (uint32_t) v643;
    const uint8_t v645 = v592[7];
    uint32_t v646 = (uint32_t) v645;
    uint32_t v647 = v646 << 8u;
    uint32_t v648 = v644 | v647;
    uint32_t v649 = v648 & 511u;
    int v650 = (int) v649;
    int v651 = v650 * 8;
    uint16_t v652 = (uint16_t) v651;
    v590[3] = v652;
    uint32_t v653 = v648 >> 9u;
    int v654 = (int) v653;
    int v655 = v654 * 8;
    uint16_t v656 = (uint16_t) v655;
    v591[3] = v656;
    const uint8_t* v657 = v19 + 56;
    const uint8_t v658 = v21[7];
    int v659 = (int) v658;
    int v660 = v659 & 15;
    int v661 = v659 >> 4;
    int v662 = v660 * 2;
    int v663 = v662 + 1;
    int v664 = v661 * 2;
    int v665 = v664 + 1;
    const uint8_t v666 = v657[0];
    uint32_t v667 = (uint32_t) v666;
    const uint8_t v668 = v657[1];
    uint32_t v669 = (uint32_t) v668;
    uint32_t v670 = v669 << 8u;
    uint32_t v671 = v667 | v670;
    uint32_t v672 = v671 & 511u;
    int v673 = (int) v672;
    int v674 = v673 * 8;
    uint16_t v675 = (uint16_t) v674;
    v590[4] = v675;
    uint32_t v676 = v671 >> 9u;
    int v677 = (int) v676;
    int v678 = v677 * 8;
    uint16_t v679 = (uint16_t) v678;
    v591[4] = v679;
    const uint8_t v680 = v657[2];
    uint32_t v681 = (uint32_t) v680;
    const uint8_t v682 = v657[3];
    uint32_t v683 = (uint32_t) v682;
    uint32_t v684 = v683 << 8u;
    uint32_t v685 = v681 | v684;
    uint32_t v686 = v685 & 511u;
    int v687 = (int) v686;
    int v688 = v687 * 8;
    uint16_t v689 = (uint16_t) v688;
    v590[5] = v689;
    uint32_t v690 = v685 >> 9u;
    int v691 = (int) v690;
    int v692 = v691 * 8;
    uint16_t v693 = (uint16_t) v692;
    v591[5] = v693;
    const uint8_t v694 = v657[4];
    uint32_t v695 = (uint32_t) v694;
    const uint8_t v696 = v657[5];
    uint32_t v697 = (uint32_t) v696;
    uint32_t v698 = v697 << 8u;
    uint32_t v699 = v695 | v698;
    uint32_t v700 = v699 & 511u;
    int v701 = (int) v700;
    int v702 = v701 * 8;
    uint16_t v703 = (uint16_t) v702;
    v590[6] = v703;
    uint32_t v704 = v699 >> 9u;
    int v705 = (int) v704;
    int v706 = v705 * 8;
    uint16_t v707 = (uint16_t) v706;
    v591[6] = v707;
    const uint8_t v708 = v657[6];
    uint32_t v709 = (uint32_t) v708;
    const uint8_t v710 = v657[7];
    uint32_t v711 = (uint32_t) v710;
    uint32_t v712 = v711 << 8u;
    uint32_t v713 = v709 | v712;
    uint32_t v714 = v713 & 511u;
    int v715 = (int) v714;
    int v716 = v715 * 8;
    uint16_t v717 = (uint16_t) v716;
    v590[7] = v717;
    uint32_t v718 = v713 >> 9u;
    int v719 = (int) v718;
    int v720 = v719 * 8;
    uint16_t v721 = (uint16_t) v720;
    v591[7] = v721;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_half
    uint16_t* v722 = &v590[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v723 = __riscv_vle16_v_u16m1(v722, 8);
    uint16_t* v724 = &v591[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_u16m1
    vuint16m1_t v725 = __riscv_vle16_v_u16m1(v724, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v726 = __riscv_vluxei16_v_i64m4(weft_iq2xs_grid, v723, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v727 = __riscv_vreinterpret_v_i64m4_i8m4(v726);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m4
    vint64m4_t v728 = __riscv_vluxei16_v_i64m4(v8, v725, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m4_i8m4
    vint8m4_t v729 = __riscv_vreinterpret_v_i64m4_i8m4(v728);
    const int8_t* v730 = v23 + 192;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m4
    vint8m4_t v731 = __riscv_vle8_v_i8m4(v730, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m4
    vint8m4_t v732 = __riscv_vmul_vv_i8m4(v727, v729, 64);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v733 = __riscv_vget_v_i8m4_i8m1(v732, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v734 = __riscv_vget_v_i8m4_i8m1(v731, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v735 = __riscv_vwmul_vv_i16m2(v733, v734, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v736 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v737 = __riscv_vwredsum_vs_i16m2_i32m1(v735, v736, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v738 = __riscv_vmv_x_s_i32m1_i32(v737);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v739 = v24;
    int32_t v740 = (int32_t) v598;
    int32_t v741 = v738 * v740;
    int32_t v742 = v739 + v741;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v742;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v743 = __riscv_vslidedown_vx_i8m4(v732, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v744 = __riscv_vslidedown_vx_i8m4(v731, 16, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v745 = __riscv_vget_v_i8m4_i8m1(v743, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v746 = __riscv_vget_v_i8m4_i8m1(v744, 0);
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
    int32_t v752 = (int32_t) v600;
    int32_t v753 = v750 * v752;
    int32_t v754 = v751 + v753;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v754;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v755 = __riscv_vslidedown_vx_i8m4(v732, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v756 = __riscv_vslidedown_vx_i8m4(v731, 32, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v757 = __riscv_vget_v_i8m4_i8m1(v755, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v758 = __riscv_vget_v_i8m4_i8m1(v756, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v759 = __riscv_vwmul_vv_i16m2(v757, v758, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v760 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v761 = __riscv_vwredsum_vs_i16m2_i32m1(v759, v760, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v762 = __riscv_vmv_x_s_i32m1_i32(v761);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v763 = v24;
    int32_t v764 = (int32_t) v663;
    int32_t v765 = v762 * v764;
    int32_t v766 = v763 + v765;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v766;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v767 = __riscv_vslidedown_vx_i8m4(v732, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m4
    vint8m4_t v768 = __riscv_vslidedown_vx_i8m4(v731, 48, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v769 = __riscv_vget_v_i8m4_i8m1(v767, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m4_i8m1
    vint8m1_t v770 = __riscv_vget_v_i8m4_i8m1(v768, 0);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v771 = __riscv_vwmul_vv_i16m2(v769, v770, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v772 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v773 = __riscv_vwredsum_vs_i16m2_i32m1(v771, v772, 16);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v774 = __riscv_vmv_x_s_i32m1_i32(v773);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v775 = v24;
    int32_t v776 = (int32_t) v665;
    int32_t v777 = v774 * v776;
    int32_t v778 = v775 + v777;
    // weft_emitc.assign target=bsum source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v24 = v778;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v779 = v24;
    float v780 = v6;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v6 = v780 + v17 * (float) v779;
  }
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v781 = v6;
  float v782 = 0.125f * v781;
  v2[0] = v782;
  return;
}


