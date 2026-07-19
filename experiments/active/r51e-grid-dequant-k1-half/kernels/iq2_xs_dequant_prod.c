#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_iq2_xs_kernel_dequant_iq2_xs(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int64_t weft_iq2xs_grid[512] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL, 0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL, 0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL, 0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL, 0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL, 0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL};
  static const int8_t weft_iq2xs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_table_view
  const uint8_t* v6 = (const uint8_t*) weft_iq2xs_grid;
  for (size_t v7 = 0; v7 < v5; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v8 = v7 * 74;
    const uint8_t* v9 = v2 + v8;
    size_t v10 = v7 * 256;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v9);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v14 = v9[66];
    int v15 = (int) v14;
    int v16 = v15 & 15;
    float v17 = (float) v16;
    float v18 = 0.5f + v17;
    float v19 = v13 * v18;
    float v20 = v19 * 0.25f;
    int v21 = v15 >> 4;
    float v22 = (float) v21;
    float v23 = 0.5f + v22;
    float v24 = v13 * v23;
    float v25 = v24 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v26 = v9[3];
    uint32_t v27 = (uint32_t) v26;
    uint32_t v28 = v27 << 8u;
    const uint8_t v29 = v9[2];
    uint32_t v30 = (uint32_t) v29;
    uint32_t v31 = v30 | v28;
    uint32_t v32 = v31 & 511u;
    size_t v33 = (size_t) v32;
    size_t v34 = v33 * 8;
    const uint8_t* v35 = v6 + v34;
    const int8_t* v36 = (const int8_t*) v35;
    uint32_t v37 = v31 >> 9u;
    size_t v38 = (size_t) v37;
    size_t v39 = v38 * 8;
    const int8_t* v40 = weft_iq2xs_signs64 + v39;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v41 = __riscv_vle8_v_i8mf2(v36, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v42 = __riscv_vle8_v_i8mf2(v40, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v43 = __riscv_vmul_vv_i8mf2(v41, v42, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v44 = __riscv_vsext_vf4_i32m2(v43, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v45 = __riscv_vfcvt_f_x_v_f32m2(v44, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v46 = __riscv_vfmul_vf_f32m2(v45, v20, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v12, v46, 8);
    const uint8_t v47 = v9[5];
    uint32_t v48 = (uint32_t) v47;
    uint32_t v49 = v48 << 8u;
    const uint8_t v50 = v9[4];
    uint32_t v51 = (uint32_t) v50;
    uint32_t v52 = v51 | v49;
    uint32_t v53 = v52 & 511u;
    size_t v54 = (size_t) v53;
    size_t v55 = v54 * 8;
    const uint8_t* v56 = v6 + v55;
    const int8_t* v57 = (const int8_t*) v56;
    uint32_t v58 = v52 >> 9u;
    size_t v59 = (size_t) v58;
    size_t v60 = v59 * 8;
    const int8_t* v61 = weft_iq2xs_signs64 + v60;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v62 = __riscv_vle8_v_i8mf2(v57, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v63 = __riscv_vle8_v_i8mf2(v61, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v64 = __riscv_vmul_vv_i8mf2(v62, v63, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v65 = __riscv_vsext_vf4_i32m2(v64, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v66 = __riscv_vfcvt_f_x_v_f32m2(v65, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v67 = __riscv_vfmul_vf_f32m2(v66, v20, 8);
    float* v68 = v12 + 8;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v68, v67, 8);
    const uint8_t v69 = v9[7];
    uint32_t v70 = (uint32_t) v69;
    uint32_t v71 = v70 << 8u;
    const uint8_t v72 = v9[6];
    uint32_t v73 = (uint32_t) v72;
    uint32_t v74 = v73 | v71;
    uint32_t v75 = v74 & 511u;
    size_t v76 = (size_t) v75;
    size_t v77 = v76 * 8;
    const uint8_t* v78 = v6 + v77;
    const int8_t* v79 = (const int8_t*) v78;
    uint32_t v80 = v74 >> 9u;
    size_t v81 = (size_t) v80;
    size_t v82 = v81 * 8;
    const int8_t* v83 = weft_iq2xs_signs64 + v82;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v84 = __riscv_vle8_v_i8mf2(v79, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v85 = __riscv_vle8_v_i8mf2(v83, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v86 = __riscv_vmul_vv_i8mf2(v84, v85, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v87 = __riscv_vsext_vf4_i32m2(v86, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v88 = __riscv_vfcvt_f_x_v_f32m2(v87, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v89 = __riscv_vfmul_vf_f32m2(v88, v25, 8);
    float* v90 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v90, v89, 8);
    const uint8_t v91 = v9[9];
    uint32_t v92 = (uint32_t) v91;
    uint32_t v93 = v92 << 8u;
    const uint8_t v94 = v9[8];
    uint32_t v95 = (uint32_t) v94;
    uint32_t v96 = v95 | v93;
    uint32_t v97 = v96 & 511u;
    size_t v98 = (size_t) v97;
    size_t v99 = v98 * 8;
    const uint8_t* v100 = v6 + v99;
    const int8_t* v101 = (const int8_t*) v100;
    uint32_t v102 = v96 >> 9u;
    size_t v103 = (size_t) v102;
    size_t v104 = v103 * 8;
    const int8_t* v105 = weft_iq2xs_signs64 + v104;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v106 = __riscv_vle8_v_i8mf2(v101, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v107 = __riscv_vle8_v_i8mf2(v105, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v108 = __riscv_vmul_vv_i8mf2(v106, v107, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v109 = __riscv_vsext_vf4_i32m2(v108, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v110 = __riscv_vfcvt_f_x_v_f32m2(v109, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v111 = __riscv_vfmul_vf_f32m2(v110, v25, 8);
    float* v112 = v12 + 24;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v112, v111, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v113 = v9[67];
    int v114 = (int) v113;
    int v115 = v114 & 15;
    float v116 = (float) v115;
    float v117 = 0.5f + v116;
    float v118 = v13 * v117;
    float v119 = v118 * 0.25f;
    int v120 = v114 >> 4;
    float v121 = (float) v120;
    float v122 = 0.5f + v121;
    float v123 = v13 * v122;
    float v124 = v123 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v125 = v9[11];
    uint32_t v126 = (uint32_t) v125;
    uint32_t v127 = v126 << 8u;
    const uint8_t v128 = v9[10];
    uint32_t v129 = (uint32_t) v128;
    uint32_t v130 = v129 | v127;
    uint32_t v131 = v130 & 511u;
    size_t v132 = (size_t) v131;
    size_t v133 = v132 * 8;
    const uint8_t* v134 = v6 + v133;
    const int8_t* v135 = (const int8_t*) v134;
    uint32_t v136 = v130 >> 9u;
    size_t v137 = (size_t) v136;
    size_t v138 = v137 * 8;
    const int8_t* v139 = weft_iq2xs_signs64 + v138;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v140 = __riscv_vle8_v_i8mf2(v135, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v141 = __riscv_vle8_v_i8mf2(v139, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v142 = __riscv_vmul_vv_i8mf2(v140, v141, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v143 = __riscv_vsext_vf4_i32m2(v142, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v144 = __riscv_vfcvt_f_x_v_f32m2(v143, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v145 = __riscv_vfmul_vf_f32m2(v144, v119, 8);
    float* v146 = v12 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v146, v145, 8);
    const uint8_t v147 = v9[13];
    uint32_t v148 = (uint32_t) v147;
    uint32_t v149 = v148 << 8u;
    const uint8_t v150 = v9[12];
    uint32_t v151 = (uint32_t) v150;
    uint32_t v152 = v151 | v149;
    uint32_t v153 = v152 & 511u;
    size_t v154 = (size_t) v153;
    size_t v155 = v154 * 8;
    const uint8_t* v156 = v6 + v155;
    const int8_t* v157 = (const int8_t*) v156;
    uint32_t v158 = v152 >> 9u;
    size_t v159 = (size_t) v158;
    size_t v160 = v159 * 8;
    const int8_t* v161 = weft_iq2xs_signs64 + v160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v162 = __riscv_vle8_v_i8mf2(v157, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v163 = __riscv_vle8_v_i8mf2(v161, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v164 = __riscv_vmul_vv_i8mf2(v162, v163, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v165 = __riscv_vsext_vf4_i32m2(v164, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v166 = __riscv_vfcvt_f_x_v_f32m2(v165, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v167 = __riscv_vfmul_vf_f32m2(v166, v119, 8);
    float* v168 = v12 + 40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v168, v167, 8);
    const uint8_t v169 = v9[15];
    uint32_t v170 = (uint32_t) v169;
    uint32_t v171 = v170 << 8u;
    const uint8_t v172 = v9[14];
    uint32_t v173 = (uint32_t) v172;
    uint32_t v174 = v173 | v171;
    uint32_t v175 = v174 & 511u;
    size_t v176 = (size_t) v175;
    size_t v177 = v176 * 8;
    const uint8_t* v178 = v6 + v177;
    const int8_t* v179 = (const int8_t*) v178;
    uint32_t v180 = v174 >> 9u;
    size_t v181 = (size_t) v180;
    size_t v182 = v181 * 8;
    const int8_t* v183 = weft_iq2xs_signs64 + v182;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v184 = __riscv_vle8_v_i8mf2(v179, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v185 = __riscv_vle8_v_i8mf2(v183, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v186 = __riscv_vmul_vv_i8mf2(v184, v185, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v187 = __riscv_vsext_vf4_i32m2(v186, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v188 = __riscv_vfcvt_f_x_v_f32m2(v187, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v189 = __riscv_vfmul_vf_f32m2(v188, v124, 8);
    float* v190 = v12 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v190, v189, 8);
    const uint8_t v191 = v9[17];
    uint32_t v192 = (uint32_t) v191;
    uint32_t v193 = v192 << 8u;
    const uint8_t v194 = v9[16];
    uint32_t v195 = (uint32_t) v194;
    uint32_t v196 = v195 | v193;
    uint32_t v197 = v196 & 511u;
    size_t v198 = (size_t) v197;
    size_t v199 = v198 * 8;
    const uint8_t* v200 = v6 + v199;
    const int8_t* v201 = (const int8_t*) v200;
    uint32_t v202 = v196 >> 9u;
    size_t v203 = (size_t) v202;
    size_t v204 = v203 * 8;
    const int8_t* v205 = weft_iq2xs_signs64 + v204;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v206 = __riscv_vle8_v_i8mf2(v201, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v207 = __riscv_vle8_v_i8mf2(v205, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v208 = __riscv_vmul_vv_i8mf2(v206, v207, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v209 = __riscv_vsext_vf4_i32m2(v208, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v210 = __riscv_vfcvt_f_x_v_f32m2(v209, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v211 = __riscv_vfmul_vf_f32m2(v210, v124, 8);
    float* v212 = v12 + 56;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v212, v211, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v213 = v9[68];
    int v214 = (int) v213;
    int v215 = v214 & 15;
    float v216 = (float) v215;
    float v217 = 0.5f + v216;
    float v218 = v13 * v217;
    float v219 = v218 * 0.25f;
    int v220 = v214 >> 4;
    float v221 = (float) v220;
    float v222 = 0.5f + v221;
    float v223 = v13 * v222;
    float v224 = v223 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v225 = v9[19];
    uint32_t v226 = (uint32_t) v225;
    uint32_t v227 = v226 << 8u;
    const uint8_t v228 = v9[18];
    uint32_t v229 = (uint32_t) v228;
    uint32_t v230 = v229 | v227;
    uint32_t v231 = v230 & 511u;
    size_t v232 = (size_t) v231;
    size_t v233 = v232 * 8;
    const uint8_t* v234 = v6 + v233;
    const int8_t* v235 = (const int8_t*) v234;
    uint32_t v236 = v230 >> 9u;
    size_t v237 = (size_t) v236;
    size_t v238 = v237 * 8;
    const int8_t* v239 = weft_iq2xs_signs64 + v238;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v240 = __riscv_vle8_v_i8mf2(v235, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v241 = __riscv_vle8_v_i8mf2(v239, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v242 = __riscv_vmul_vv_i8mf2(v240, v241, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v243 = __riscv_vsext_vf4_i32m2(v242, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v244 = __riscv_vfcvt_f_x_v_f32m2(v243, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v245 = __riscv_vfmul_vf_f32m2(v244, v219, 8);
    float* v246 = v12 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v246, v245, 8);
    const uint8_t v247 = v9[21];
    uint32_t v248 = (uint32_t) v247;
    uint32_t v249 = v248 << 8u;
    const uint8_t v250 = v9[20];
    uint32_t v251 = (uint32_t) v250;
    uint32_t v252 = v251 | v249;
    uint32_t v253 = v252 & 511u;
    size_t v254 = (size_t) v253;
    size_t v255 = v254 * 8;
    const uint8_t* v256 = v6 + v255;
    const int8_t* v257 = (const int8_t*) v256;
    uint32_t v258 = v252 >> 9u;
    size_t v259 = (size_t) v258;
    size_t v260 = v259 * 8;
    const int8_t* v261 = weft_iq2xs_signs64 + v260;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v262 = __riscv_vle8_v_i8mf2(v257, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v263 = __riscv_vle8_v_i8mf2(v261, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v264 = __riscv_vmul_vv_i8mf2(v262, v263, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v265 = __riscv_vsext_vf4_i32m2(v264, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v266 = __riscv_vfcvt_f_x_v_f32m2(v265, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v267 = __riscv_vfmul_vf_f32m2(v266, v219, 8);
    float* v268 = v12 + 72;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v268, v267, 8);
    const uint8_t v269 = v9[23];
    uint32_t v270 = (uint32_t) v269;
    uint32_t v271 = v270 << 8u;
    const uint8_t v272 = v9[22];
    uint32_t v273 = (uint32_t) v272;
    uint32_t v274 = v273 | v271;
    uint32_t v275 = v274 & 511u;
    size_t v276 = (size_t) v275;
    size_t v277 = v276 * 8;
    const uint8_t* v278 = v6 + v277;
    const int8_t* v279 = (const int8_t*) v278;
    uint32_t v280 = v274 >> 9u;
    size_t v281 = (size_t) v280;
    size_t v282 = v281 * 8;
    const int8_t* v283 = weft_iq2xs_signs64 + v282;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v284 = __riscv_vle8_v_i8mf2(v279, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v285 = __riscv_vle8_v_i8mf2(v283, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v286 = __riscv_vmul_vv_i8mf2(v284, v285, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v287 = __riscv_vsext_vf4_i32m2(v286, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v288 = __riscv_vfcvt_f_x_v_f32m2(v287, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v289 = __riscv_vfmul_vf_f32m2(v288, v224, 8);
    float* v290 = v12 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v290, v289, 8);
    const uint8_t v291 = v9[25];
    uint32_t v292 = (uint32_t) v291;
    uint32_t v293 = v292 << 8u;
    const uint8_t v294 = v9[24];
    uint32_t v295 = (uint32_t) v294;
    uint32_t v296 = v295 | v293;
    uint32_t v297 = v296 & 511u;
    size_t v298 = (size_t) v297;
    size_t v299 = v298 * 8;
    const uint8_t* v300 = v6 + v299;
    const int8_t* v301 = (const int8_t*) v300;
    uint32_t v302 = v296 >> 9u;
    size_t v303 = (size_t) v302;
    size_t v304 = v303 * 8;
    const int8_t* v305 = weft_iq2xs_signs64 + v304;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v306 = __riscv_vle8_v_i8mf2(v301, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v307 = __riscv_vle8_v_i8mf2(v305, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v308 = __riscv_vmul_vv_i8mf2(v306, v307, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v309 = __riscv_vsext_vf4_i32m2(v308, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v310 = __riscv_vfcvt_f_x_v_f32m2(v309, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v311 = __riscv_vfmul_vf_f32m2(v310, v224, 8);
    float* v312 = v12 + 88;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v312, v311, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v313 = v9[69];
    int v314 = (int) v313;
    int v315 = v314 & 15;
    float v316 = (float) v315;
    float v317 = 0.5f + v316;
    float v318 = v13 * v317;
    float v319 = v318 * 0.25f;
    int v320 = v314 >> 4;
    float v321 = (float) v320;
    float v322 = 0.5f + v321;
    float v323 = v13 * v322;
    float v324 = v323 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v325 = v9[27];
    uint32_t v326 = (uint32_t) v325;
    uint32_t v327 = v326 << 8u;
    const uint8_t v328 = v9[26];
    uint32_t v329 = (uint32_t) v328;
    uint32_t v330 = v329 | v327;
    uint32_t v331 = v330 & 511u;
    size_t v332 = (size_t) v331;
    size_t v333 = v332 * 8;
    const uint8_t* v334 = v6 + v333;
    const int8_t* v335 = (const int8_t*) v334;
    uint32_t v336 = v330 >> 9u;
    size_t v337 = (size_t) v336;
    size_t v338 = v337 * 8;
    const int8_t* v339 = weft_iq2xs_signs64 + v338;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v340 = __riscv_vle8_v_i8mf2(v335, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v341 = __riscv_vle8_v_i8mf2(v339, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v342 = __riscv_vmul_vv_i8mf2(v340, v341, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v343 = __riscv_vsext_vf4_i32m2(v342, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v344 = __riscv_vfcvt_f_x_v_f32m2(v343, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v345 = __riscv_vfmul_vf_f32m2(v344, v319, 8);
    float* v346 = v12 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v346, v345, 8);
    const uint8_t v347 = v9[29];
    uint32_t v348 = (uint32_t) v347;
    uint32_t v349 = v348 << 8u;
    const uint8_t v350 = v9[28];
    uint32_t v351 = (uint32_t) v350;
    uint32_t v352 = v351 | v349;
    uint32_t v353 = v352 & 511u;
    size_t v354 = (size_t) v353;
    size_t v355 = v354 * 8;
    const uint8_t* v356 = v6 + v355;
    const int8_t* v357 = (const int8_t*) v356;
    uint32_t v358 = v352 >> 9u;
    size_t v359 = (size_t) v358;
    size_t v360 = v359 * 8;
    const int8_t* v361 = weft_iq2xs_signs64 + v360;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v362 = __riscv_vle8_v_i8mf2(v357, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v363 = __riscv_vle8_v_i8mf2(v361, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v364 = __riscv_vmul_vv_i8mf2(v362, v363, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v365 = __riscv_vsext_vf4_i32m2(v364, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v366 = __riscv_vfcvt_f_x_v_f32m2(v365, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v367 = __riscv_vfmul_vf_f32m2(v366, v319, 8);
    float* v368 = v12 + 104;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v368, v367, 8);
    const uint8_t v369 = v9[31];
    uint32_t v370 = (uint32_t) v369;
    uint32_t v371 = v370 << 8u;
    const uint8_t v372 = v9[30];
    uint32_t v373 = (uint32_t) v372;
    uint32_t v374 = v373 | v371;
    uint32_t v375 = v374 & 511u;
    size_t v376 = (size_t) v375;
    size_t v377 = v376 * 8;
    const uint8_t* v378 = v6 + v377;
    const int8_t* v379 = (const int8_t*) v378;
    uint32_t v380 = v374 >> 9u;
    size_t v381 = (size_t) v380;
    size_t v382 = v381 * 8;
    const int8_t* v383 = weft_iq2xs_signs64 + v382;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v384 = __riscv_vle8_v_i8mf2(v379, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v385 = __riscv_vle8_v_i8mf2(v383, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v386 = __riscv_vmul_vv_i8mf2(v384, v385, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v387 = __riscv_vsext_vf4_i32m2(v386, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v388 = __riscv_vfcvt_f_x_v_f32m2(v387, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v389 = __riscv_vfmul_vf_f32m2(v388, v324, 8);
    float* v390 = v12 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v390, v389, 8);
    const uint8_t v391 = v9[33];
    uint32_t v392 = (uint32_t) v391;
    uint32_t v393 = v392 << 8u;
    const uint8_t v394 = v9[32];
    uint32_t v395 = (uint32_t) v394;
    uint32_t v396 = v395 | v393;
    uint32_t v397 = v396 & 511u;
    size_t v398 = (size_t) v397;
    size_t v399 = v398 * 8;
    const uint8_t* v400 = v6 + v399;
    const int8_t* v401 = (const int8_t*) v400;
    uint32_t v402 = v396 >> 9u;
    size_t v403 = (size_t) v402;
    size_t v404 = v403 * 8;
    const int8_t* v405 = weft_iq2xs_signs64 + v404;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v406 = __riscv_vle8_v_i8mf2(v401, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v407 = __riscv_vle8_v_i8mf2(v405, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v408 = __riscv_vmul_vv_i8mf2(v406, v407, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v409 = __riscv_vsext_vf4_i32m2(v408, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v410 = __riscv_vfcvt_f_x_v_f32m2(v409, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v411 = __riscv_vfmul_vf_f32m2(v410, v324, 8);
    float* v412 = v12 + 120;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v412, v411, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v413 = v9[70];
    int v414 = (int) v413;
    int v415 = v414 & 15;
    float v416 = (float) v415;
    float v417 = 0.5f + v416;
    float v418 = v13 * v417;
    float v419 = v418 * 0.25f;
    int v420 = v414 >> 4;
    float v421 = (float) v420;
    float v422 = 0.5f + v421;
    float v423 = v13 * v422;
    float v424 = v423 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v425 = v9[35];
    uint32_t v426 = (uint32_t) v425;
    uint32_t v427 = v426 << 8u;
    const uint8_t v428 = v9[34];
    uint32_t v429 = (uint32_t) v428;
    uint32_t v430 = v429 | v427;
    uint32_t v431 = v430 & 511u;
    size_t v432 = (size_t) v431;
    size_t v433 = v432 * 8;
    const uint8_t* v434 = v6 + v433;
    const int8_t* v435 = (const int8_t*) v434;
    uint32_t v436 = v430 >> 9u;
    size_t v437 = (size_t) v436;
    size_t v438 = v437 * 8;
    const int8_t* v439 = weft_iq2xs_signs64 + v438;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v440 = __riscv_vle8_v_i8mf2(v435, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v441 = __riscv_vle8_v_i8mf2(v439, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v442 = __riscv_vmul_vv_i8mf2(v440, v441, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v443 = __riscv_vsext_vf4_i32m2(v442, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v444 = __riscv_vfcvt_f_x_v_f32m2(v443, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v445 = __riscv_vfmul_vf_f32m2(v444, v419, 8);
    float* v446 = v12 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v446, v445, 8);
    const uint8_t v447 = v9[37];
    uint32_t v448 = (uint32_t) v447;
    uint32_t v449 = v448 << 8u;
    const uint8_t v450 = v9[36];
    uint32_t v451 = (uint32_t) v450;
    uint32_t v452 = v451 | v449;
    uint32_t v453 = v452 & 511u;
    size_t v454 = (size_t) v453;
    size_t v455 = v454 * 8;
    const uint8_t* v456 = v6 + v455;
    const int8_t* v457 = (const int8_t*) v456;
    uint32_t v458 = v452 >> 9u;
    size_t v459 = (size_t) v458;
    size_t v460 = v459 * 8;
    const int8_t* v461 = weft_iq2xs_signs64 + v460;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v462 = __riscv_vle8_v_i8mf2(v457, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v463 = __riscv_vle8_v_i8mf2(v461, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v464 = __riscv_vmul_vv_i8mf2(v462, v463, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v465 = __riscv_vsext_vf4_i32m2(v464, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v466 = __riscv_vfcvt_f_x_v_f32m2(v465, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v467 = __riscv_vfmul_vf_f32m2(v466, v419, 8);
    float* v468 = v12 + 136;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v468, v467, 8);
    const uint8_t v469 = v9[39];
    uint32_t v470 = (uint32_t) v469;
    uint32_t v471 = v470 << 8u;
    const uint8_t v472 = v9[38];
    uint32_t v473 = (uint32_t) v472;
    uint32_t v474 = v473 | v471;
    uint32_t v475 = v474 & 511u;
    size_t v476 = (size_t) v475;
    size_t v477 = v476 * 8;
    const uint8_t* v478 = v6 + v477;
    const int8_t* v479 = (const int8_t*) v478;
    uint32_t v480 = v474 >> 9u;
    size_t v481 = (size_t) v480;
    size_t v482 = v481 * 8;
    const int8_t* v483 = weft_iq2xs_signs64 + v482;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v484 = __riscv_vle8_v_i8mf2(v479, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v485 = __riscv_vle8_v_i8mf2(v483, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v486 = __riscv_vmul_vv_i8mf2(v484, v485, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v487 = __riscv_vsext_vf4_i32m2(v486, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v488 = __riscv_vfcvt_f_x_v_f32m2(v487, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v489 = __riscv_vfmul_vf_f32m2(v488, v424, 8);
    float* v490 = v12 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v490, v489, 8);
    const uint8_t v491 = v9[41];
    uint32_t v492 = (uint32_t) v491;
    uint32_t v493 = v492 << 8u;
    const uint8_t v494 = v9[40];
    uint32_t v495 = (uint32_t) v494;
    uint32_t v496 = v495 | v493;
    uint32_t v497 = v496 & 511u;
    size_t v498 = (size_t) v497;
    size_t v499 = v498 * 8;
    const uint8_t* v500 = v6 + v499;
    const int8_t* v501 = (const int8_t*) v500;
    uint32_t v502 = v496 >> 9u;
    size_t v503 = (size_t) v502;
    size_t v504 = v503 * 8;
    const int8_t* v505 = weft_iq2xs_signs64 + v504;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v506 = __riscv_vle8_v_i8mf2(v501, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v507 = __riscv_vle8_v_i8mf2(v505, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v508 = __riscv_vmul_vv_i8mf2(v506, v507, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v509 = __riscv_vsext_vf4_i32m2(v508, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v510 = __riscv_vfcvt_f_x_v_f32m2(v509, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v511 = __riscv_vfmul_vf_f32m2(v510, v424, 8);
    float* v512 = v12 + 152;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v512, v511, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v513 = v9[71];
    int v514 = (int) v513;
    int v515 = v514 & 15;
    float v516 = (float) v515;
    float v517 = 0.5f + v516;
    float v518 = v13 * v517;
    float v519 = v518 * 0.25f;
    int v520 = v514 >> 4;
    float v521 = (float) v520;
    float v522 = 0.5f + v521;
    float v523 = v13 * v522;
    float v524 = v523 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v525 = v9[43];
    uint32_t v526 = (uint32_t) v525;
    uint32_t v527 = v526 << 8u;
    const uint8_t v528 = v9[42];
    uint32_t v529 = (uint32_t) v528;
    uint32_t v530 = v529 | v527;
    uint32_t v531 = v530 & 511u;
    size_t v532 = (size_t) v531;
    size_t v533 = v532 * 8;
    const uint8_t* v534 = v6 + v533;
    const int8_t* v535 = (const int8_t*) v534;
    uint32_t v536 = v530 >> 9u;
    size_t v537 = (size_t) v536;
    size_t v538 = v537 * 8;
    const int8_t* v539 = weft_iq2xs_signs64 + v538;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v540 = __riscv_vle8_v_i8mf2(v535, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v541 = __riscv_vle8_v_i8mf2(v539, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v542 = __riscv_vmul_vv_i8mf2(v540, v541, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v543 = __riscv_vsext_vf4_i32m2(v542, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v544 = __riscv_vfcvt_f_x_v_f32m2(v543, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v545 = __riscv_vfmul_vf_f32m2(v544, v519, 8);
    float* v546 = v12 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v546, v545, 8);
    const uint8_t v547 = v9[45];
    uint32_t v548 = (uint32_t) v547;
    uint32_t v549 = v548 << 8u;
    const uint8_t v550 = v9[44];
    uint32_t v551 = (uint32_t) v550;
    uint32_t v552 = v551 | v549;
    uint32_t v553 = v552 & 511u;
    size_t v554 = (size_t) v553;
    size_t v555 = v554 * 8;
    const uint8_t* v556 = v6 + v555;
    const int8_t* v557 = (const int8_t*) v556;
    uint32_t v558 = v552 >> 9u;
    size_t v559 = (size_t) v558;
    size_t v560 = v559 * 8;
    const int8_t* v561 = weft_iq2xs_signs64 + v560;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v562 = __riscv_vle8_v_i8mf2(v557, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v563 = __riscv_vle8_v_i8mf2(v561, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v564 = __riscv_vmul_vv_i8mf2(v562, v563, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v565 = __riscv_vsext_vf4_i32m2(v564, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v566 = __riscv_vfcvt_f_x_v_f32m2(v565, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v567 = __riscv_vfmul_vf_f32m2(v566, v519, 8);
    float* v568 = v12 + 168;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v568, v567, 8);
    const uint8_t v569 = v9[47];
    uint32_t v570 = (uint32_t) v569;
    uint32_t v571 = v570 << 8u;
    const uint8_t v572 = v9[46];
    uint32_t v573 = (uint32_t) v572;
    uint32_t v574 = v573 | v571;
    uint32_t v575 = v574 & 511u;
    size_t v576 = (size_t) v575;
    size_t v577 = v576 * 8;
    const uint8_t* v578 = v6 + v577;
    const int8_t* v579 = (const int8_t*) v578;
    uint32_t v580 = v574 >> 9u;
    size_t v581 = (size_t) v580;
    size_t v582 = v581 * 8;
    const int8_t* v583 = weft_iq2xs_signs64 + v582;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v584 = __riscv_vle8_v_i8mf2(v579, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v585 = __riscv_vle8_v_i8mf2(v583, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v586 = __riscv_vmul_vv_i8mf2(v584, v585, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v587 = __riscv_vsext_vf4_i32m2(v586, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v588 = __riscv_vfcvt_f_x_v_f32m2(v587, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v589 = __riscv_vfmul_vf_f32m2(v588, v524, 8);
    float* v590 = v12 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v590, v589, 8);
    const uint8_t v591 = v9[49];
    uint32_t v592 = (uint32_t) v591;
    uint32_t v593 = v592 << 8u;
    const uint8_t v594 = v9[48];
    uint32_t v595 = (uint32_t) v594;
    uint32_t v596 = v595 | v593;
    uint32_t v597 = v596 & 511u;
    size_t v598 = (size_t) v597;
    size_t v599 = v598 * 8;
    const uint8_t* v600 = v6 + v599;
    const int8_t* v601 = (const int8_t*) v600;
    uint32_t v602 = v596 >> 9u;
    size_t v603 = (size_t) v602;
    size_t v604 = v603 * 8;
    const int8_t* v605 = weft_iq2xs_signs64 + v604;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v606 = __riscv_vle8_v_i8mf2(v601, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v607 = __riscv_vle8_v_i8mf2(v605, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v608 = __riscv_vmul_vv_i8mf2(v606, v607, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v609 = __riscv_vsext_vf4_i32m2(v608, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v610 = __riscv_vfcvt_f_x_v_f32m2(v609, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v611 = __riscv_vfmul_vf_f32m2(v610, v524, 8);
    float* v612 = v12 + 184;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v612, v611, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v613 = v9[72];
    int v614 = (int) v613;
    int v615 = v614 & 15;
    float v616 = (float) v615;
    float v617 = 0.5f + v616;
    float v618 = v13 * v617;
    float v619 = v618 * 0.25f;
    int v620 = v614 >> 4;
    float v621 = (float) v620;
    float v622 = 0.5f + v621;
    float v623 = v13 * v622;
    float v624 = v623 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v625 = v9[51];
    uint32_t v626 = (uint32_t) v625;
    uint32_t v627 = v626 << 8u;
    const uint8_t v628 = v9[50];
    uint32_t v629 = (uint32_t) v628;
    uint32_t v630 = v629 | v627;
    uint32_t v631 = v630 & 511u;
    size_t v632 = (size_t) v631;
    size_t v633 = v632 * 8;
    const uint8_t* v634 = v6 + v633;
    const int8_t* v635 = (const int8_t*) v634;
    uint32_t v636 = v630 >> 9u;
    size_t v637 = (size_t) v636;
    size_t v638 = v637 * 8;
    const int8_t* v639 = weft_iq2xs_signs64 + v638;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v640 = __riscv_vle8_v_i8mf2(v635, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v641 = __riscv_vle8_v_i8mf2(v639, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v642 = __riscv_vmul_vv_i8mf2(v640, v641, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v643 = __riscv_vsext_vf4_i32m2(v642, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v644 = __riscv_vfcvt_f_x_v_f32m2(v643, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v645 = __riscv_vfmul_vf_f32m2(v644, v619, 8);
    float* v646 = v12 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v646, v645, 8);
    const uint8_t v647 = v9[53];
    uint32_t v648 = (uint32_t) v647;
    uint32_t v649 = v648 << 8u;
    const uint8_t v650 = v9[52];
    uint32_t v651 = (uint32_t) v650;
    uint32_t v652 = v651 | v649;
    uint32_t v653 = v652 & 511u;
    size_t v654 = (size_t) v653;
    size_t v655 = v654 * 8;
    const uint8_t* v656 = v6 + v655;
    const int8_t* v657 = (const int8_t*) v656;
    uint32_t v658 = v652 >> 9u;
    size_t v659 = (size_t) v658;
    size_t v660 = v659 * 8;
    const int8_t* v661 = weft_iq2xs_signs64 + v660;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v662 = __riscv_vle8_v_i8mf2(v657, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v663 = __riscv_vle8_v_i8mf2(v661, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v664 = __riscv_vmul_vv_i8mf2(v662, v663, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v665 = __riscv_vsext_vf4_i32m2(v664, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v666 = __riscv_vfcvt_f_x_v_f32m2(v665, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v667 = __riscv_vfmul_vf_f32m2(v666, v619, 8);
    float* v668 = v12 + 200;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v668, v667, 8);
    const uint8_t v669 = v9[55];
    uint32_t v670 = (uint32_t) v669;
    uint32_t v671 = v670 << 8u;
    const uint8_t v672 = v9[54];
    uint32_t v673 = (uint32_t) v672;
    uint32_t v674 = v673 | v671;
    uint32_t v675 = v674 & 511u;
    size_t v676 = (size_t) v675;
    size_t v677 = v676 * 8;
    const uint8_t* v678 = v6 + v677;
    const int8_t* v679 = (const int8_t*) v678;
    uint32_t v680 = v674 >> 9u;
    size_t v681 = (size_t) v680;
    size_t v682 = v681 * 8;
    const int8_t* v683 = weft_iq2xs_signs64 + v682;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v684 = __riscv_vle8_v_i8mf2(v679, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v685 = __riscv_vle8_v_i8mf2(v683, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v686 = __riscv_vmul_vv_i8mf2(v684, v685, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v687 = __riscv_vsext_vf4_i32m2(v686, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v688 = __riscv_vfcvt_f_x_v_f32m2(v687, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v689 = __riscv_vfmul_vf_f32m2(v688, v624, 8);
    float* v690 = v12 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v690, v689, 8);
    const uint8_t v691 = v9[57];
    uint32_t v692 = (uint32_t) v691;
    uint32_t v693 = v692 << 8u;
    const uint8_t v694 = v9[56];
    uint32_t v695 = (uint32_t) v694;
    uint32_t v696 = v695 | v693;
    uint32_t v697 = v696 & 511u;
    size_t v698 = (size_t) v697;
    size_t v699 = v698 * 8;
    const uint8_t* v700 = v6 + v699;
    const int8_t* v701 = (const int8_t*) v700;
    uint32_t v702 = v696 >> 9u;
    size_t v703 = (size_t) v702;
    size_t v704 = v703 * 8;
    const int8_t* v705 = weft_iq2xs_signs64 + v704;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v706 = __riscv_vle8_v_i8mf2(v701, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v707 = __riscv_vle8_v_i8mf2(v705, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v708 = __riscv_vmul_vv_i8mf2(v706, v707, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v709 = __riscv_vsext_vf4_i32m2(v708, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v710 = __riscv_vfcvt_f_x_v_f32m2(v709, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v711 = __riscv_vfmul_vf_f32m2(v710, v624, 8);
    float* v712 = v12 + 216;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v712, v711, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_scale
    const uint8_t v713 = v9[73];
    int v714 = (int) v713;
    int v715 = v714 & 15;
    float v716 = (float) v715;
    float v717 = 0.5f + v716;
    float v718 = v13 * v717;
    float v719 = v718 * 0.25f;
    int v720 = v714 >> 4;
    float v721 = (float) v720;
    float v722 = 0.5f + v721;
    float v723 = v13 * v722;
    float v724 = v723 * 0.25f;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=grid_sign_subblock
    const uint8_t v725 = v9[59];
    uint32_t v726 = (uint32_t) v725;
    uint32_t v727 = v726 << 8u;
    const uint8_t v728 = v9[58];
    uint32_t v729 = (uint32_t) v728;
    uint32_t v730 = v729 | v727;
    uint32_t v731 = v730 & 511u;
    size_t v732 = (size_t) v731;
    size_t v733 = v732 * 8;
    const uint8_t* v734 = v6 + v733;
    const int8_t* v735 = (const int8_t*) v734;
    uint32_t v736 = v730 >> 9u;
    size_t v737 = (size_t) v736;
    size_t v738 = v737 * 8;
    const int8_t* v739 = weft_iq2xs_signs64 + v738;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v740 = __riscv_vle8_v_i8mf2(v735, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v741 = __riscv_vle8_v_i8mf2(v739, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v742 = __riscv_vmul_vv_i8mf2(v740, v741, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v743 = __riscv_vsext_vf4_i32m2(v742, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v744 = __riscv_vfcvt_f_x_v_f32m2(v743, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v745 = __riscv_vfmul_vf_f32m2(v744, v719, 8);
    float* v746 = v12 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v746, v745, 8);
    const uint8_t v747 = v9[61];
    uint32_t v748 = (uint32_t) v747;
    uint32_t v749 = v748 << 8u;
    const uint8_t v750 = v9[60];
    uint32_t v751 = (uint32_t) v750;
    uint32_t v752 = v751 | v749;
    uint32_t v753 = v752 & 511u;
    size_t v754 = (size_t) v753;
    size_t v755 = v754 * 8;
    const uint8_t* v756 = v6 + v755;
    const int8_t* v757 = (const int8_t*) v756;
    uint32_t v758 = v752 >> 9u;
    size_t v759 = (size_t) v758;
    size_t v760 = v759 * 8;
    const int8_t* v761 = weft_iq2xs_signs64 + v760;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v762 = __riscv_vle8_v_i8mf2(v757, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v763 = __riscv_vle8_v_i8mf2(v761, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v764 = __riscv_vmul_vv_i8mf2(v762, v763, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v765 = __riscv_vsext_vf4_i32m2(v764, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v766 = __riscv_vfcvt_f_x_v_f32m2(v765, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v767 = __riscv_vfmul_vf_f32m2(v766, v719, 8);
    float* v768 = v12 + 232;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v768, v767, 8);
    const uint8_t v769 = v9[63];
    uint32_t v770 = (uint32_t) v769;
    uint32_t v771 = v770 << 8u;
    const uint8_t v772 = v9[62];
    uint32_t v773 = (uint32_t) v772;
    uint32_t v774 = v773 | v771;
    uint32_t v775 = v774 & 511u;
    size_t v776 = (size_t) v775;
    size_t v777 = v776 * 8;
    const uint8_t* v778 = v6 + v777;
    const int8_t* v779 = (const int8_t*) v778;
    uint32_t v780 = v774 >> 9u;
    size_t v781 = (size_t) v780;
    size_t v782 = v781 * 8;
    const int8_t* v783 = weft_iq2xs_signs64 + v782;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v784 = __riscv_vle8_v_i8mf2(v779, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v785 = __riscv_vle8_v_i8mf2(v783, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v786 = __riscv_vmul_vv_i8mf2(v784, v785, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v787 = __riscv_vsext_vf4_i32m2(v786, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v788 = __riscv_vfcvt_f_x_v_f32m2(v787, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v789 = __riscv_vfmul_vf_f32m2(v788, v724, 8);
    float* v790 = v12 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v790, v789, 8);
    const uint8_t v791 = v9[65];
    uint32_t v792 = (uint32_t) v791;
    uint32_t v793 = v792 << 8u;
    const uint8_t v794 = v9[64];
    uint32_t v795 = (uint32_t) v794;
    uint32_t v796 = v795 | v793;
    uint32_t v797 = v796 & 511u;
    size_t v798 = (size_t) v797;
    size_t v799 = v798 * 8;
    const uint8_t* v800 = v6 + v799;
    const int8_t* v801 = (const int8_t*) v800;
    uint32_t v802 = v796 >> 9u;
    size_t v803 = (size_t) v802;
    size_t v804 = v803 * 8;
    const int8_t* v805 = weft_iq2xs_signs64 + v804;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v806 = __riscv_vle8_v_i8mf2(v801, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
    vint8mf2_t v807 = __riscv_vle8_v_i8mf2(v805, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8mf2
    vint8mf2_t v808 = __riscv_vmul_vv_i8mf2(v806, v807, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m2
    vint32m2_t v809 = __riscv_vsext_vf4_i32m2(v808, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v810 = __riscv_vfcvt_f_x_v_f32m2(v809, 8);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v811 = __riscv_vfmul_vf_f32m2(v810, v724, 8);
    float* v812 = v12 + 248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v812, v811, 8);
  }
  return;
}


