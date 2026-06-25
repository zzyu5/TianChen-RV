#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_ggml_vec_dot_iq2_xs_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int64_t tcrv_iq2xs_grid[512] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL, 0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL, 0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL, 0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL, 0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL, 0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL, 0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL, 0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL, 0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL};
  static const int8_t tcrv_iq2xs_signs64[1024] = {1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_table_i64_view
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=signs_table_i64_view
  const int64_t* v8 = (const int64_t*) tcrv_iq2xs_signs64;
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v10 = v9 * 74;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v12 = v9 * 292;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v15 = (const float*) v13;
    const float v16 = v15[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    float v17 = v14 * v16;
    const uint8_t* v18 = v11 + 2;
    const uint8_t* v19 = (const uint8_t*) v18;
    const uint8_t* v20 = v11 + 66;
    const uint8_t* v21 = (const uint8_t*) v20;
    const uint8_t* v22 = v13 + 4;
    const int8_t* v23 = (const int8_t*) v22;
    // tcrv_emitc.local_variable=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v24;
    v24 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t v25 = v21[0];
    int v26 = (int) v25;
    int v27 = v26 & 15;
    int v28 = v26 >> 4;
    int v29 = v27 * 2;
    int v30 = v29 + 1;
    int v31 = v28 * 2;
    int v32 = v31 + 1;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v33[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v34[2];
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
    v33[0] = v44;
    uint32_t v45 = v40 >> 9u;
    int v46 = (int) v45;
    int v47 = v46 * 8;
    uint16_t v48 = (uint16_t) v47;
    v34[0] = v48;
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
    v33[1] = v58;
    uint32_t v59 = v54 >> 9u;
    int v60 = (int) v59;
    int v61 = v60 * 8;
    uint16_t v62 = (uint16_t) v61;
    v34[1] = v62;
    uint16_t* v63 = &v33[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v64 = __riscv_vle16_v_u16mf4(v63, 2);
    uint16_t* v65 = &v34[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v66 = __riscv_vle16_v_u16mf4(v65, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v67 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v64, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v68 = __riscv_vreinterpret_v_i64m1_i8m1(v67);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v69 = __riscv_vluxei16_v_i64m1(v8, v66, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v70 = __riscv_vreinterpret_v_i64m1_i8m1(v69);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v71 = __riscv_vle8_v_i8m1(v23, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v72 = __riscv_vmul_vv_i8m1(v68, v70, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v73 = __riscv_vwmul_vv_i16m2(v72, v71, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v74 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v75 = __riscv_vwredsum_vs_i16m2_i32m1(v73, v74, 16);
    const int8_t* v76 = v23 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v77 = __riscv_vmv_x_s_i32m1_i32(v75);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v78 = v24;
    int32_t v79 = (int32_t) v30;
    int32_t v80 = v77 * v79;
    int32_t v81 = v78 + v80;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v81;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v82[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v83[2];
    const uint8_t v84 = v19[4];
    uint32_t v85 = (uint32_t) v84;
    const uint8_t v86 = v19[5];
    uint32_t v87 = (uint32_t) v86;
    uint32_t v88 = v87 << 8u;
    uint32_t v89 = v85 | v88;
    uint32_t v90 = v89 & 511u;
    int v91 = (int) v90;
    int v92 = v91 * 8;
    uint16_t v93 = (uint16_t) v92;
    v82[0] = v93;
    uint32_t v94 = v89 >> 9u;
    int v95 = (int) v94;
    int v96 = v95 * 8;
    uint16_t v97 = (uint16_t) v96;
    v83[0] = v97;
    const uint8_t v98 = v19[6];
    uint32_t v99 = (uint32_t) v98;
    const uint8_t v100 = v19[7];
    uint32_t v101 = (uint32_t) v100;
    uint32_t v102 = v101 << 8u;
    uint32_t v103 = v99 | v102;
    uint32_t v104 = v103 & 511u;
    int v105 = (int) v104;
    int v106 = v105 * 8;
    uint16_t v107 = (uint16_t) v106;
    v82[1] = v107;
    uint32_t v108 = v103 >> 9u;
    int v109 = (int) v108;
    int v110 = v109 * 8;
    uint16_t v111 = (uint16_t) v110;
    v83[1] = v111;
    uint16_t* v112 = &v82[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v113 = __riscv_vle16_v_u16mf4(v112, 2);
    uint16_t* v114 = &v83[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v115 = __riscv_vle16_v_u16mf4(v114, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v116 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v113, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v117 = __riscv_vreinterpret_v_i64m1_i8m1(v116);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v118 = __riscv_vluxei16_v_i64m1(v8, v115, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v119 = __riscv_vreinterpret_v_i64m1_i8m1(v118);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v120 = __riscv_vle8_v_i8m1(v76, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v121 = __riscv_vmul_vv_i8m1(v117, v119, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v122 = __riscv_vwmul_vv_i16m2(v121, v120, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v123 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v124 = __riscv_vwredsum_vs_i16m2_i32m1(v122, v123, 16);
    const int8_t* v125 = v76 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v126 = __riscv_vmv_x_s_i32m1_i32(v124);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v127 = v24;
    int32_t v128 = (int32_t) v32;
    int32_t v129 = v126 * v128;
    int32_t v130 = v127 + v129;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v130;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v131 = v19 + 8;
    const uint8_t v132 = v21[1];
    int v133 = (int) v132;
    int v134 = v133 & 15;
    int v135 = v133 >> 4;
    int v136 = v134 * 2;
    int v137 = v136 + 1;
    int v138 = v135 * 2;
    int v139 = v138 + 1;
    const int8_t* v140 = v23 + 32;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v141[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v142[2];
    const uint8_t v143 = v131[0];
    uint32_t v144 = (uint32_t) v143;
    const uint8_t v145 = v131[1];
    uint32_t v146 = (uint32_t) v145;
    uint32_t v147 = v146 << 8u;
    uint32_t v148 = v144 | v147;
    uint32_t v149 = v148 & 511u;
    int v150 = (int) v149;
    int v151 = v150 * 8;
    uint16_t v152 = (uint16_t) v151;
    v141[0] = v152;
    uint32_t v153 = v148 >> 9u;
    int v154 = (int) v153;
    int v155 = v154 * 8;
    uint16_t v156 = (uint16_t) v155;
    v142[0] = v156;
    const uint8_t v157 = v131[2];
    uint32_t v158 = (uint32_t) v157;
    const uint8_t v159 = v131[3];
    uint32_t v160 = (uint32_t) v159;
    uint32_t v161 = v160 << 8u;
    uint32_t v162 = v158 | v161;
    uint32_t v163 = v162 & 511u;
    int v164 = (int) v163;
    int v165 = v164 * 8;
    uint16_t v166 = (uint16_t) v165;
    v141[1] = v166;
    uint32_t v167 = v162 >> 9u;
    int v168 = (int) v167;
    int v169 = v168 * 8;
    uint16_t v170 = (uint16_t) v169;
    v142[1] = v170;
    uint16_t* v171 = &v141[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v172 = __riscv_vle16_v_u16mf4(v171, 2);
    uint16_t* v173 = &v142[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v174 = __riscv_vle16_v_u16mf4(v173, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v175 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v172, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v176 = __riscv_vreinterpret_v_i64m1_i8m1(v175);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v177 = __riscv_vluxei16_v_i64m1(v8, v174, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v178 = __riscv_vreinterpret_v_i64m1_i8m1(v177);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v179 = __riscv_vle8_v_i8m1(v140, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v180 = __riscv_vmul_vv_i8m1(v176, v178, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v181 = __riscv_vwmul_vv_i16m2(v180, v179, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v182 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v183 = __riscv_vwredsum_vs_i16m2_i32m1(v181, v182, 16);
    const int8_t* v184 = v140 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v185 = __riscv_vmv_x_s_i32m1_i32(v183);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v186 = v24;
    int32_t v187 = (int32_t) v137;
    int32_t v188 = v185 * v187;
    int32_t v189 = v186 + v188;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v189;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v190[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v191[2];
    const uint8_t v192 = v131[4];
    uint32_t v193 = (uint32_t) v192;
    const uint8_t v194 = v131[5];
    uint32_t v195 = (uint32_t) v194;
    uint32_t v196 = v195 << 8u;
    uint32_t v197 = v193 | v196;
    uint32_t v198 = v197 & 511u;
    int v199 = (int) v198;
    int v200 = v199 * 8;
    uint16_t v201 = (uint16_t) v200;
    v190[0] = v201;
    uint32_t v202 = v197 >> 9u;
    int v203 = (int) v202;
    int v204 = v203 * 8;
    uint16_t v205 = (uint16_t) v204;
    v191[0] = v205;
    const uint8_t v206 = v131[6];
    uint32_t v207 = (uint32_t) v206;
    const uint8_t v208 = v131[7];
    uint32_t v209 = (uint32_t) v208;
    uint32_t v210 = v209 << 8u;
    uint32_t v211 = v207 | v210;
    uint32_t v212 = v211 & 511u;
    int v213 = (int) v212;
    int v214 = v213 * 8;
    uint16_t v215 = (uint16_t) v214;
    v190[1] = v215;
    uint32_t v216 = v211 >> 9u;
    int v217 = (int) v216;
    int v218 = v217 * 8;
    uint16_t v219 = (uint16_t) v218;
    v191[1] = v219;
    uint16_t* v220 = &v190[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v221 = __riscv_vle16_v_u16mf4(v220, 2);
    uint16_t* v222 = &v191[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v223 = __riscv_vle16_v_u16mf4(v222, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v224 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v221, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v225 = __riscv_vreinterpret_v_i64m1_i8m1(v224);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v226 = __riscv_vluxei16_v_i64m1(v8, v223, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v227 = __riscv_vreinterpret_v_i64m1_i8m1(v226);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v228 = __riscv_vle8_v_i8m1(v184, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v229 = __riscv_vmul_vv_i8m1(v225, v227, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v230 = __riscv_vwmul_vv_i16m2(v229, v228, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v231 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v232 = __riscv_vwredsum_vs_i16m2_i32m1(v230, v231, 16);
    const int8_t* v233 = v184 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v234 = __riscv_vmv_x_s_i32m1_i32(v232);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v235 = v24;
    int32_t v236 = (int32_t) v139;
    int32_t v237 = v234 * v236;
    int32_t v238 = v235 + v237;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v238;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v239 = v19 + 16;
    const uint8_t v240 = v21[2];
    int v241 = (int) v240;
    int v242 = v241 & 15;
    int v243 = v241 >> 4;
    int v244 = v242 * 2;
    int v245 = v244 + 1;
    int v246 = v243 * 2;
    int v247 = v246 + 1;
    const int8_t* v248 = v23 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v249[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v250[2];
    const uint8_t v251 = v239[0];
    uint32_t v252 = (uint32_t) v251;
    const uint8_t v253 = v239[1];
    uint32_t v254 = (uint32_t) v253;
    uint32_t v255 = v254 << 8u;
    uint32_t v256 = v252 | v255;
    uint32_t v257 = v256 & 511u;
    int v258 = (int) v257;
    int v259 = v258 * 8;
    uint16_t v260 = (uint16_t) v259;
    v249[0] = v260;
    uint32_t v261 = v256 >> 9u;
    int v262 = (int) v261;
    int v263 = v262 * 8;
    uint16_t v264 = (uint16_t) v263;
    v250[0] = v264;
    const uint8_t v265 = v239[2];
    uint32_t v266 = (uint32_t) v265;
    const uint8_t v267 = v239[3];
    uint32_t v268 = (uint32_t) v267;
    uint32_t v269 = v268 << 8u;
    uint32_t v270 = v266 | v269;
    uint32_t v271 = v270 & 511u;
    int v272 = (int) v271;
    int v273 = v272 * 8;
    uint16_t v274 = (uint16_t) v273;
    v249[1] = v274;
    uint32_t v275 = v270 >> 9u;
    int v276 = (int) v275;
    int v277 = v276 * 8;
    uint16_t v278 = (uint16_t) v277;
    v250[1] = v278;
    uint16_t* v279 = &v249[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v280 = __riscv_vle16_v_u16mf4(v279, 2);
    uint16_t* v281 = &v250[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v282 = __riscv_vle16_v_u16mf4(v281, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v283 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v280, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v284 = __riscv_vreinterpret_v_i64m1_i8m1(v283);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v285 = __riscv_vluxei16_v_i64m1(v8, v282, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v286 = __riscv_vreinterpret_v_i64m1_i8m1(v285);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v287 = __riscv_vle8_v_i8m1(v248, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v288 = __riscv_vmul_vv_i8m1(v284, v286, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v289 = __riscv_vwmul_vv_i16m2(v288, v287, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v290 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v291 = __riscv_vwredsum_vs_i16m2_i32m1(v289, v290, 16);
    const int8_t* v292 = v248 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v293 = __riscv_vmv_x_s_i32m1_i32(v291);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v294 = v24;
    int32_t v295 = (int32_t) v245;
    int32_t v296 = v293 * v295;
    int32_t v297 = v294 + v296;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v297;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v298[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v299[2];
    const uint8_t v300 = v239[4];
    uint32_t v301 = (uint32_t) v300;
    const uint8_t v302 = v239[5];
    uint32_t v303 = (uint32_t) v302;
    uint32_t v304 = v303 << 8u;
    uint32_t v305 = v301 | v304;
    uint32_t v306 = v305 & 511u;
    int v307 = (int) v306;
    int v308 = v307 * 8;
    uint16_t v309 = (uint16_t) v308;
    v298[0] = v309;
    uint32_t v310 = v305 >> 9u;
    int v311 = (int) v310;
    int v312 = v311 * 8;
    uint16_t v313 = (uint16_t) v312;
    v299[0] = v313;
    const uint8_t v314 = v239[6];
    uint32_t v315 = (uint32_t) v314;
    const uint8_t v316 = v239[7];
    uint32_t v317 = (uint32_t) v316;
    uint32_t v318 = v317 << 8u;
    uint32_t v319 = v315 | v318;
    uint32_t v320 = v319 & 511u;
    int v321 = (int) v320;
    int v322 = v321 * 8;
    uint16_t v323 = (uint16_t) v322;
    v298[1] = v323;
    uint32_t v324 = v319 >> 9u;
    int v325 = (int) v324;
    int v326 = v325 * 8;
    uint16_t v327 = (uint16_t) v326;
    v299[1] = v327;
    uint16_t* v328 = &v298[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v329 = __riscv_vle16_v_u16mf4(v328, 2);
    uint16_t* v330 = &v299[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v331 = __riscv_vle16_v_u16mf4(v330, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v332 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v329, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v333 = __riscv_vreinterpret_v_i64m1_i8m1(v332);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v334 = __riscv_vluxei16_v_i64m1(v8, v331, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v335 = __riscv_vreinterpret_v_i64m1_i8m1(v334);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v336 = __riscv_vle8_v_i8m1(v292, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v337 = __riscv_vmul_vv_i8m1(v333, v335, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v338 = __riscv_vwmul_vv_i16m2(v337, v336, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v339 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v340 = __riscv_vwredsum_vs_i16m2_i32m1(v338, v339, 16);
    const int8_t* v341 = v292 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v342 = __riscv_vmv_x_s_i32m1_i32(v340);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v343 = v24;
    int32_t v344 = (int32_t) v247;
    int32_t v345 = v342 * v344;
    int32_t v346 = v343 + v345;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v346;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v347 = v19 + 24;
    const uint8_t v348 = v21[3];
    int v349 = (int) v348;
    int v350 = v349 & 15;
    int v351 = v349 >> 4;
    int v352 = v350 * 2;
    int v353 = v352 + 1;
    int v354 = v351 * 2;
    int v355 = v354 + 1;
    const int8_t* v356 = v23 + 96;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v357[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v358[2];
    const uint8_t v359 = v347[0];
    uint32_t v360 = (uint32_t) v359;
    const uint8_t v361 = v347[1];
    uint32_t v362 = (uint32_t) v361;
    uint32_t v363 = v362 << 8u;
    uint32_t v364 = v360 | v363;
    uint32_t v365 = v364 & 511u;
    int v366 = (int) v365;
    int v367 = v366 * 8;
    uint16_t v368 = (uint16_t) v367;
    v357[0] = v368;
    uint32_t v369 = v364 >> 9u;
    int v370 = (int) v369;
    int v371 = v370 * 8;
    uint16_t v372 = (uint16_t) v371;
    v358[0] = v372;
    const uint8_t v373 = v347[2];
    uint32_t v374 = (uint32_t) v373;
    const uint8_t v375 = v347[3];
    uint32_t v376 = (uint32_t) v375;
    uint32_t v377 = v376 << 8u;
    uint32_t v378 = v374 | v377;
    uint32_t v379 = v378 & 511u;
    int v380 = (int) v379;
    int v381 = v380 * 8;
    uint16_t v382 = (uint16_t) v381;
    v357[1] = v382;
    uint32_t v383 = v378 >> 9u;
    int v384 = (int) v383;
    int v385 = v384 * 8;
    uint16_t v386 = (uint16_t) v385;
    v358[1] = v386;
    uint16_t* v387 = &v357[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v388 = __riscv_vle16_v_u16mf4(v387, 2);
    uint16_t* v389 = &v358[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v390 = __riscv_vle16_v_u16mf4(v389, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v391 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v388, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v392 = __riscv_vreinterpret_v_i64m1_i8m1(v391);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v393 = __riscv_vluxei16_v_i64m1(v8, v390, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v394 = __riscv_vreinterpret_v_i64m1_i8m1(v393);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v395 = __riscv_vle8_v_i8m1(v356, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v396 = __riscv_vmul_vv_i8m1(v392, v394, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v397 = __riscv_vwmul_vv_i16m2(v396, v395, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v398 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v399 = __riscv_vwredsum_vs_i16m2_i32m1(v397, v398, 16);
    const int8_t* v400 = v356 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v401 = __riscv_vmv_x_s_i32m1_i32(v399);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v402 = v24;
    int32_t v403 = (int32_t) v353;
    int32_t v404 = v401 * v403;
    int32_t v405 = v402 + v404;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v405;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v406[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v407[2];
    const uint8_t v408 = v347[4];
    uint32_t v409 = (uint32_t) v408;
    const uint8_t v410 = v347[5];
    uint32_t v411 = (uint32_t) v410;
    uint32_t v412 = v411 << 8u;
    uint32_t v413 = v409 | v412;
    uint32_t v414 = v413 & 511u;
    int v415 = (int) v414;
    int v416 = v415 * 8;
    uint16_t v417 = (uint16_t) v416;
    v406[0] = v417;
    uint32_t v418 = v413 >> 9u;
    int v419 = (int) v418;
    int v420 = v419 * 8;
    uint16_t v421 = (uint16_t) v420;
    v407[0] = v421;
    const uint8_t v422 = v347[6];
    uint32_t v423 = (uint32_t) v422;
    const uint8_t v424 = v347[7];
    uint32_t v425 = (uint32_t) v424;
    uint32_t v426 = v425 << 8u;
    uint32_t v427 = v423 | v426;
    uint32_t v428 = v427 & 511u;
    int v429 = (int) v428;
    int v430 = v429 * 8;
    uint16_t v431 = (uint16_t) v430;
    v406[1] = v431;
    uint32_t v432 = v427 >> 9u;
    int v433 = (int) v432;
    int v434 = v433 * 8;
    uint16_t v435 = (uint16_t) v434;
    v407[1] = v435;
    uint16_t* v436 = &v406[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v437 = __riscv_vle16_v_u16mf4(v436, 2);
    uint16_t* v438 = &v407[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v439 = __riscv_vle16_v_u16mf4(v438, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v440 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v437, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v441 = __riscv_vreinterpret_v_i64m1_i8m1(v440);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v442 = __riscv_vluxei16_v_i64m1(v8, v439, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v443 = __riscv_vreinterpret_v_i64m1_i8m1(v442);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v444 = __riscv_vle8_v_i8m1(v400, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v445 = __riscv_vmul_vv_i8m1(v441, v443, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v446 = __riscv_vwmul_vv_i16m2(v445, v444, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v447 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v448 = __riscv_vwredsum_vs_i16m2_i32m1(v446, v447, 16);
    const int8_t* v449 = v400 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v450 = __riscv_vmv_x_s_i32m1_i32(v448);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v451 = v24;
    int32_t v452 = (int32_t) v355;
    int32_t v453 = v450 * v452;
    int32_t v454 = v451 + v453;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v454;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v455 = v19 + 32;
    const uint8_t v456 = v21[4];
    int v457 = (int) v456;
    int v458 = v457 & 15;
    int v459 = v457 >> 4;
    int v460 = v458 * 2;
    int v461 = v460 + 1;
    int v462 = v459 * 2;
    int v463 = v462 + 1;
    const int8_t* v464 = v23 + 128;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v465[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v466[2];
    const uint8_t v467 = v455[0];
    uint32_t v468 = (uint32_t) v467;
    const uint8_t v469 = v455[1];
    uint32_t v470 = (uint32_t) v469;
    uint32_t v471 = v470 << 8u;
    uint32_t v472 = v468 | v471;
    uint32_t v473 = v472 & 511u;
    int v474 = (int) v473;
    int v475 = v474 * 8;
    uint16_t v476 = (uint16_t) v475;
    v465[0] = v476;
    uint32_t v477 = v472 >> 9u;
    int v478 = (int) v477;
    int v479 = v478 * 8;
    uint16_t v480 = (uint16_t) v479;
    v466[0] = v480;
    const uint8_t v481 = v455[2];
    uint32_t v482 = (uint32_t) v481;
    const uint8_t v483 = v455[3];
    uint32_t v484 = (uint32_t) v483;
    uint32_t v485 = v484 << 8u;
    uint32_t v486 = v482 | v485;
    uint32_t v487 = v486 & 511u;
    int v488 = (int) v487;
    int v489 = v488 * 8;
    uint16_t v490 = (uint16_t) v489;
    v465[1] = v490;
    uint32_t v491 = v486 >> 9u;
    int v492 = (int) v491;
    int v493 = v492 * 8;
    uint16_t v494 = (uint16_t) v493;
    v466[1] = v494;
    uint16_t* v495 = &v465[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v496 = __riscv_vle16_v_u16mf4(v495, 2);
    uint16_t* v497 = &v466[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v498 = __riscv_vle16_v_u16mf4(v497, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v499 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v496, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v500 = __riscv_vreinterpret_v_i64m1_i8m1(v499);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v501 = __riscv_vluxei16_v_i64m1(v8, v498, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v502 = __riscv_vreinterpret_v_i64m1_i8m1(v501);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v503 = __riscv_vle8_v_i8m1(v464, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v504 = __riscv_vmul_vv_i8m1(v500, v502, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v505 = __riscv_vwmul_vv_i16m2(v504, v503, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v506 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v507 = __riscv_vwredsum_vs_i16m2_i32m1(v505, v506, 16);
    const int8_t* v508 = v464 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v509 = __riscv_vmv_x_s_i32m1_i32(v507);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v510 = v24;
    int32_t v511 = (int32_t) v461;
    int32_t v512 = v509 * v511;
    int32_t v513 = v510 + v512;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v513;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v514[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v515[2];
    const uint8_t v516 = v455[4];
    uint32_t v517 = (uint32_t) v516;
    const uint8_t v518 = v455[5];
    uint32_t v519 = (uint32_t) v518;
    uint32_t v520 = v519 << 8u;
    uint32_t v521 = v517 | v520;
    uint32_t v522 = v521 & 511u;
    int v523 = (int) v522;
    int v524 = v523 * 8;
    uint16_t v525 = (uint16_t) v524;
    v514[0] = v525;
    uint32_t v526 = v521 >> 9u;
    int v527 = (int) v526;
    int v528 = v527 * 8;
    uint16_t v529 = (uint16_t) v528;
    v515[0] = v529;
    const uint8_t v530 = v455[6];
    uint32_t v531 = (uint32_t) v530;
    const uint8_t v532 = v455[7];
    uint32_t v533 = (uint32_t) v532;
    uint32_t v534 = v533 << 8u;
    uint32_t v535 = v531 | v534;
    uint32_t v536 = v535 & 511u;
    int v537 = (int) v536;
    int v538 = v537 * 8;
    uint16_t v539 = (uint16_t) v538;
    v514[1] = v539;
    uint32_t v540 = v535 >> 9u;
    int v541 = (int) v540;
    int v542 = v541 * 8;
    uint16_t v543 = (uint16_t) v542;
    v515[1] = v543;
    uint16_t* v544 = &v514[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v545 = __riscv_vle16_v_u16mf4(v544, 2);
    uint16_t* v546 = &v515[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v547 = __riscv_vle16_v_u16mf4(v546, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v548 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v545, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v549 = __riscv_vreinterpret_v_i64m1_i8m1(v548);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v550 = __riscv_vluxei16_v_i64m1(v8, v547, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v551 = __riscv_vreinterpret_v_i64m1_i8m1(v550);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v552 = __riscv_vle8_v_i8m1(v508, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v553 = __riscv_vmul_vv_i8m1(v549, v551, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v554 = __riscv_vwmul_vv_i16m2(v553, v552, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v555 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v556 = __riscv_vwredsum_vs_i16m2_i32m1(v554, v555, 16);
    const int8_t* v557 = v508 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v558 = __riscv_vmv_x_s_i32m1_i32(v556);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v559 = v24;
    int32_t v560 = (int32_t) v463;
    int32_t v561 = v558 * v560;
    int32_t v562 = v559 + v561;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v562;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v563 = v19 + 40;
    const uint8_t v564 = v21[5];
    int v565 = (int) v564;
    int v566 = v565 & 15;
    int v567 = v565 >> 4;
    int v568 = v566 * 2;
    int v569 = v568 + 1;
    int v570 = v567 * 2;
    int v571 = v570 + 1;
    const int8_t* v572 = v23 + 160;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v573[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v574[2];
    const uint8_t v575 = v563[0];
    uint32_t v576 = (uint32_t) v575;
    const uint8_t v577 = v563[1];
    uint32_t v578 = (uint32_t) v577;
    uint32_t v579 = v578 << 8u;
    uint32_t v580 = v576 | v579;
    uint32_t v581 = v580 & 511u;
    int v582 = (int) v581;
    int v583 = v582 * 8;
    uint16_t v584 = (uint16_t) v583;
    v573[0] = v584;
    uint32_t v585 = v580 >> 9u;
    int v586 = (int) v585;
    int v587 = v586 * 8;
    uint16_t v588 = (uint16_t) v587;
    v574[0] = v588;
    const uint8_t v589 = v563[2];
    uint32_t v590 = (uint32_t) v589;
    const uint8_t v591 = v563[3];
    uint32_t v592 = (uint32_t) v591;
    uint32_t v593 = v592 << 8u;
    uint32_t v594 = v590 | v593;
    uint32_t v595 = v594 & 511u;
    int v596 = (int) v595;
    int v597 = v596 * 8;
    uint16_t v598 = (uint16_t) v597;
    v573[1] = v598;
    uint32_t v599 = v594 >> 9u;
    int v600 = (int) v599;
    int v601 = v600 * 8;
    uint16_t v602 = (uint16_t) v601;
    v574[1] = v602;
    uint16_t* v603 = &v573[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v604 = __riscv_vle16_v_u16mf4(v603, 2);
    uint16_t* v605 = &v574[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v606 = __riscv_vle16_v_u16mf4(v605, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v607 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v604, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v608 = __riscv_vreinterpret_v_i64m1_i8m1(v607);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v609 = __riscv_vluxei16_v_i64m1(v8, v606, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v610 = __riscv_vreinterpret_v_i64m1_i8m1(v609);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v611 = __riscv_vle8_v_i8m1(v572, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v612 = __riscv_vmul_vv_i8m1(v608, v610, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v613 = __riscv_vwmul_vv_i16m2(v612, v611, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v614 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v615 = __riscv_vwredsum_vs_i16m2_i32m1(v613, v614, 16);
    const int8_t* v616 = v572 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v617 = __riscv_vmv_x_s_i32m1_i32(v615);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v618 = v24;
    int32_t v619 = (int32_t) v569;
    int32_t v620 = v617 * v619;
    int32_t v621 = v618 + v620;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v621;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v622[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v623[2];
    const uint8_t v624 = v563[4];
    uint32_t v625 = (uint32_t) v624;
    const uint8_t v626 = v563[5];
    uint32_t v627 = (uint32_t) v626;
    uint32_t v628 = v627 << 8u;
    uint32_t v629 = v625 | v628;
    uint32_t v630 = v629 & 511u;
    int v631 = (int) v630;
    int v632 = v631 * 8;
    uint16_t v633 = (uint16_t) v632;
    v622[0] = v633;
    uint32_t v634 = v629 >> 9u;
    int v635 = (int) v634;
    int v636 = v635 * 8;
    uint16_t v637 = (uint16_t) v636;
    v623[0] = v637;
    const uint8_t v638 = v563[6];
    uint32_t v639 = (uint32_t) v638;
    const uint8_t v640 = v563[7];
    uint32_t v641 = (uint32_t) v640;
    uint32_t v642 = v641 << 8u;
    uint32_t v643 = v639 | v642;
    uint32_t v644 = v643 & 511u;
    int v645 = (int) v644;
    int v646 = v645 * 8;
    uint16_t v647 = (uint16_t) v646;
    v622[1] = v647;
    uint32_t v648 = v643 >> 9u;
    int v649 = (int) v648;
    int v650 = v649 * 8;
    uint16_t v651 = (uint16_t) v650;
    v623[1] = v651;
    uint16_t* v652 = &v622[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v653 = __riscv_vle16_v_u16mf4(v652, 2);
    uint16_t* v654 = &v623[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v655 = __riscv_vle16_v_u16mf4(v654, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v656 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v653, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v657 = __riscv_vreinterpret_v_i64m1_i8m1(v656);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v658 = __riscv_vluxei16_v_i64m1(v8, v655, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v659 = __riscv_vreinterpret_v_i64m1_i8m1(v658);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v660 = __riscv_vle8_v_i8m1(v616, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v661 = __riscv_vmul_vv_i8m1(v657, v659, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v662 = __riscv_vwmul_vv_i16m2(v661, v660, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v663 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v664 = __riscv_vwredsum_vs_i16m2_i32m1(v662, v663, 16);
    const int8_t* v665 = v616 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v666 = __riscv_vmv_x_s_i32m1_i32(v664);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v667 = v24;
    int32_t v668 = (int32_t) v571;
    int32_t v669 = v666 * v668;
    int32_t v670 = v667 + v669;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v670;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v671 = v19 + 48;
    const uint8_t v672 = v21[6];
    int v673 = (int) v672;
    int v674 = v673 & 15;
    int v675 = v673 >> 4;
    int v676 = v674 * 2;
    int v677 = v676 + 1;
    int v678 = v675 * 2;
    int v679 = v678 + 1;
    const int8_t* v680 = v23 + 192;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v681[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v682[2];
    const uint8_t v683 = v671[0];
    uint32_t v684 = (uint32_t) v683;
    const uint8_t v685 = v671[1];
    uint32_t v686 = (uint32_t) v685;
    uint32_t v687 = v686 << 8u;
    uint32_t v688 = v684 | v687;
    uint32_t v689 = v688 & 511u;
    int v690 = (int) v689;
    int v691 = v690 * 8;
    uint16_t v692 = (uint16_t) v691;
    v681[0] = v692;
    uint32_t v693 = v688 >> 9u;
    int v694 = (int) v693;
    int v695 = v694 * 8;
    uint16_t v696 = (uint16_t) v695;
    v682[0] = v696;
    const uint8_t v697 = v671[2];
    uint32_t v698 = (uint32_t) v697;
    const uint8_t v699 = v671[3];
    uint32_t v700 = (uint32_t) v699;
    uint32_t v701 = v700 << 8u;
    uint32_t v702 = v698 | v701;
    uint32_t v703 = v702 & 511u;
    int v704 = (int) v703;
    int v705 = v704 * 8;
    uint16_t v706 = (uint16_t) v705;
    v681[1] = v706;
    uint32_t v707 = v702 >> 9u;
    int v708 = (int) v707;
    int v709 = v708 * 8;
    uint16_t v710 = (uint16_t) v709;
    v682[1] = v710;
    uint16_t* v711 = &v681[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v712 = __riscv_vle16_v_u16mf4(v711, 2);
    uint16_t* v713 = &v682[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v714 = __riscv_vle16_v_u16mf4(v713, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v715 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v712, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v716 = __riscv_vreinterpret_v_i64m1_i8m1(v715);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v717 = __riscv_vluxei16_v_i64m1(v8, v714, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v718 = __riscv_vreinterpret_v_i64m1_i8m1(v717);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v719 = __riscv_vle8_v_i8m1(v680, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v720 = __riscv_vmul_vv_i8m1(v716, v718, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v721 = __riscv_vwmul_vv_i16m2(v720, v719, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v722 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v723 = __riscv_vwredsum_vs_i16m2_i32m1(v721, v722, 16);
    const int8_t* v724 = v680 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v725 = __riscv_vmv_x_s_i32m1_i32(v723);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v726 = v24;
    int32_t v727 = (int32_t) v677;
    int32_t v728 = v725 * v727;
    int32_t v729 = v726 + v728;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v729;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v730[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v731[2];
    const uint8_t v732 = v671[4];
    uint32_t v733 = (uint32_t) v732;
    const uint8_t v734 = v671[5];
    uint32_t v735 = (uint32_t) v734;
    uint32_t v736 = v735 << 8u;
    uint32_t v737 = v733 | v736;
    uint32_t v738 = v737 & 511u;
    int v739 = (int) v738;
    int v740 = v739 * 8;
    uint16_t v741 = (uint16_t) v740;
    v730[0] = v741;
    uint32_t v742 = v737 >> 9u;
    int v743 = (int) v742;
    int v744 = v743 * 8;
    uint16_t v745 = (uint16_t) v744;
    v731[0] = v745;
    const uint8_t v746 = v671[6];
    uint32_t v747 = (uint32_t) v746;
    const uint8_t v748 = v671[7];
    uint32_t v749 = (uint32_t) v748;
    uint32_t v750 = v749 << 8u;
    uint32_t v751 = v747 | v750;
    uint32_t v752 = v751 & 511u;
    int v753 = (int) v752;
    int v754 = v753 * 8;
    uint16_t v755 = (uint16_t) v754;
    v730[1] = v755;
    uint32_t v756 = v751 >> 9u;
    int v757 = (int) v756;
    int v758 = v757 * 8;
    uint16_t v759 = (uint16_t) v758;
    v731[1] = v759;
    uint16_t* v760 = &v730[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v761 = __riscv_vle16_v_u16mf4(v760, 2);
    uint16_t* v762 = &v731[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v763 = __riscv_vle16_v_u16mf4(v762, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v764 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v761, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v765 = __riscv_vreinterpret_v_i64m1_i8m1(v764);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v766 = __riscv_vluxei16_v_i64m1(v8, v763, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v767 = __riscv_vreinterpret_v_i64m1_i8m1(v766);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v768 = __riscv_vle8_v_i8m1(v724, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v769 = __riscv_vmul_vv_i8m1(v765, v767, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v770 = __riscv_vwmul_vv_i16m2(v769, v768, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v771 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v772 = __riscv_vwredsum_vs_i16m2_i32m1(v770, v771, 16);
    const int8_t* v773 = v724 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v774 = __riscv_vmv_x_s_i32m1_i32(v772);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v775 = v24;
    int32_t v776 = (int32_t) v679;
    int32_t v777 = v774 * v776;
    int32_t v778 = v775 + v777;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v778;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_explicit_scales
    const uint8_t* v779 = v19 + 56;
    const uint8_t v780 = v21[7];
    int v781 = (int) v780;
    int v782 = v781 & 15;
    int v783 = v781 >> 4;
    int v784 = v782 * 2;
    int v785 = v784 + 1;
    int v786 = v783 * 2;
    int v787 = v786 + 1;
    const int8_t* v788 = v23 + 224;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v789[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v790[2];
    const uint8_t v791 = v779[0];
    uint32_t v792 = (uint32_t) v791;
    const uint8_t v793 = v779[1];
    uint32_t v794 = (uint32_t) v793;
    uint32_t v795 = v794 << 8u;
    uint32_t v796 = v792 | v795;
    uint32_t v797 = v796 & 511u;
    int v798 = (int) v797;
    int v799 = v798 * 8;
    uint16_t v800 = (uint16_t) v799;
    v789[0] = v800;
    uint32_t v801 = v796 >> 9u;
    int v802 = (int) v801;
    int v803 = v802 * 8;
    uint16_t v804 = (uint16_t) v803;
    v790[0] = v804;
    const uint8_t v805 = v779[2];
    uint32_t v806 = (uint32_t) v805;
    const uint8_t v807 = v779[3];
    uint32_t v808 = (uint32_t) v807;
    uint32_t v809 = v808 << 8u;
    uint32_t v810 = v806 | v809;
    uint32_t v811 = v810 & 511u;
    int v812 = (int) v811;
    int v813 = v812 * 8;
    uint16_t v814 = (uint16_t) v813;
    v789[1] = v814;
    uint32_t v815 = v810 >> 9u;
    int v816 = (int) v815;
    int v817 = v816 * 8;
    uint16_t v818 = (uint16_t) v817;
    v790[1] = v818;
    uint16_t* v819 = &v789[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v820 = __riscv_vle16_v_u16mf4(v819, 2);
    uint16_t* v821 = &v790[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v822 = __riscv_vle16_v_u16mf4(v821, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v823 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v820, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v824 = __riscv_vreinterpret_v_i64m1_i8m1(v823);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v825 = __riscv_vluxei16_v_i64m1(v8, v822, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v826 = __riscv_vreinterpret_v_i64m1_i8m1(v825);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v827 = __riscv_vle8_v_i8m1(v788, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v828 = __riscv_vmul_vv_i8m1(v824, v826, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v829 = __riscv_vwmul_vv_i16m2(v828, v827, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v830 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v831 = __riscv_vwredsum_vs_i16m2_i32m1(v829, v830, 16);
    const int8_t* v832 = v788 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v833 = __riscv_vmv_x_s_i32m1_i32(v831);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v834 = v24;
    int32_t v835 = (int32_t) v785;
    int32_t v836 = v833 * v835;
    int32_t v837 = v834 + v836;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v837;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=grid_sign_half
    // tcrv_emitc.local_variable=gridoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v838[2];
    // tcrv_emitc.local_variable=signoff source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    uint16_t v839[2];
    const uint8_t v840 = v779[4];
    uint32_t v841 = (uint32_t) v840;
    const uint8_t v842 = v779[5];
    uint32_t v843 = (uint32_t) v842;
    uint32_t v844 = v843 << 8u;
    uint32_t v845 = v841 | v844;
    uint32_t v846 = v845 & 511u;
    int v847 = (int) v846;
    int v848 = v847 * 8;
    uint16_t v849 = (uint16_t) v848;
    v838[0] = v849;
    uint32_t v850 = v845 >> 9u;
    int v851 = (int) v850;
    int v852 = v851 * 8;
    uint16_t v853 = (uint16_t) v852;
    v839[0] = v853;
    const uint8_t v854 = v779[6];
    uint32_t v855 = (uint32_t) v854;
    const uint8_t v856 = v779[7];
    uint32_t v857 = (uint32_t) v856;
    uint32_t v858 = v857 << 8u;
    uint32_t v859 = v855 | v858;
    uint32_t v860 = v859 & 511u;
    int v861 = (int) v860;
    int v862 = v861 * 8;
    uint16_t v863 = (uint16_t) v862;
    v838[1] = v863;
    uint32_t v864 = v859 >> 9u;
    int v865 = (int) v864;
    int v866 = v865 * 8;
    uint16_t v867 = (uint16_t) v866;
    v839[1] = v867;
    uint16_t* v868 = &v838[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v869 = __riscv_vle16_v_u16mf4(v868, 2);
    uint16_t* v870 = &v839[0];
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_u16mf4
    vuint16mf4_t v871 = __riscv_vle16_v_u16mf4(v870, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v872 = __riscv_vluxei16_v_i64m1(tcrv_iq2xs_grid, v869, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v873 = __riscv_vreinterpret_v_i64m1_i8m1(v872);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vluxei16_v_i64m1
    vint64m1_t v874 = __riscv_vluxei16_v_i64m1(v8, v871, 2);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_i64m1_i8m1
    vint8m1_t v875 = __riscv_vreinterpret_v_i64m1_i8m1(v874);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v876 = __riscv_vle8_v_i8m1(v832, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vv_i8m1
    vint8m1_t v877 = __riscv_vmul_vv_i8m1(v873, v875, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v878 = __riscv_vwmul_vv_i16m2(v877, v876, 16);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v879 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v880 = __riscv_vwredsum_vs_i16m2_i32m1(v878, v879, 16);
    const int8_t* v881 = v832 + 16;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v882 = __riscv_vmv_x_s_i32m1_i32(v880);
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bsum_accumulate
    int32_t v883 = v24;
    int32_t v884 = (int32_t) v787;
    int32_t v885 = v882 * v884;
    int32_t v886 = v883 + v885;
    // tcrv_emitc.assign target=bsum source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v24 = v886;
    // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v887 = v24;
    float v888 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v888 + v17 * (float) v887;
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq2_xs_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v889 = v6;
  float v890 = 0.125f * v889;
  v2[0] = v890;
  return;
}


