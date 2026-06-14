#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_unsigned_product_reduce_kernel_explicit_selected_body_rvv_unsigned_product_reduce(const uint8_t* v1, const uint8_t* v2, const uint32_t* v3, uint32_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v5);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m1
  const uint32_t v7 = v3[0];
  vuint32m1_t v8 = __riscv_vmv_v_x_u32m1(v7, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_u32m1
  __riscv_vse32_v_u32m1(v4, v8, 1);
  for (size_t v9 = 0; v9 < v5; v9 += v6) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v10 = v5 - v9;
    size_t v11 = __riscv_vsetvl_e32m1(v10);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
    const uint8_t* v12 = v1 + v9;
    vuint8mf4_t v13 = __riscv_vle8_v_u8mf4(v12, v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
    const uint8_t* v14 = v2 + v9;
    vuint8mf4_t v15 = __riscv_vle8_v_u8mf4(v14, v11);
    // tcrv_emitc.source_op=tcrv_rvv.widening_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vv_u16mf2
    vuint16mf2_t v16 = __riscv_vwmulu_vv_u16mf2(v13, v15, v11);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m1
    uint32_t v17 = v4[0];
    vuint32m1_t v18 = __riscv_vmv_v_x_u32m1(v17, 1);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsumu_vs_u16mf2_u32m1
    vuint32m1_t v19 = __riscv_vwredsumu_vs_u16mf2_u32m1(v16, v18, v11);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_u32m1
    __riscv_vse32_v_u32m1(v4, v19, 1);
  }
  return;
}


