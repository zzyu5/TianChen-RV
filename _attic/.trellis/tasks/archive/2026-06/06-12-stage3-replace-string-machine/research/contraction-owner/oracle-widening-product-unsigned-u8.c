#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(const uint8_t* v1, const uint8_t* v2, uint16_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16mf2
  size_t v5 = __riscv_vsetvl_e16mf2(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e16mf2
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e16mf2(v7);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
    const uint8_t* v9 = v1 + v6;
    vuint8mf4_t v10 = __riscv_vle8_v_u8mf4(v9, v8);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
    const uint8_t* v11 = v2 + v6;
    vuint8mf4_t v12 = __riscv_vle8_v_u8mf4(v11, v8);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vv_u16mf2
    vuint16mf2_t v13 = __riscv_vwmulu_vv_u16mf2(v10, v12, v8);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse16_v_u16mf2
    uint16_t* v14 = v3 + v6;
    __riscv_vse16_v_u16mf2(v14, v13, v8);
  }
  return;
}


