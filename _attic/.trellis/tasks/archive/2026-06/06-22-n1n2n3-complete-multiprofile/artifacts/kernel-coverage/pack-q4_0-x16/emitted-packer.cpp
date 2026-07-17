#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16(size_t v1, const uint8_t* v2, uint8_t* v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_block_loop
  for (size_t v5 = 0; v5 < v1; v5 += 1) {
    size_t v6 = v5 * 288;
    size_t v7 = v5 * 288;
    // tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_scales
    for (size_t v8 = 0; v8 < 16; v8 += 1) {
      size_t v9 = v8 * 18;
      size_t v10 = v6 + v9;
      size_t v11 = v8 * 2;
      size_t v12 = v7 + v11;
      size_t v13 = v10 + 0;
      const uint8_t v14 = v2[v13];
      uint8_t v15 = (uint8_t) v14;
      size_t v16 = v12 + 0;
      v3[v16] = v15;
      size_t v17 = v10 + 1;
      const uint8_t v18 = v2[v17];
      uint8_t v19 = (uint8_t) v18;
      size_t v20 = v12 + 1;
      v3[v20] = v19;
    }
    // tcrv_emitc.source_op=tcrv_rvv.pack_q4_0_to_q4_0x16 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=pack_quants_xor
    size_t v21 = v7 + 32;
    for (size_t v22 = 0; v22 < 16; v22 += 1) {
      size_t v23 = v22 * 16;
      size_t v24 = v21 + v23;
      size_t v25 = 2 + v22;
      for (size_t v26 = 0; v26 < 16; v26 += 1) {
        size_t v27 = v26 * 18;
        size_t v28 = v6 + v27;
        size_t v29 = v28 + v25;
        size_t v30 = v24 + v26;
        const uint8_t v31 = v2[v29];
        uint8_t v32 = (uint8_t) v31;
        uint8_t v33 = v32 ^ 136;
        v3[v30] = v33;
      }
    }
  }
  return;
}


