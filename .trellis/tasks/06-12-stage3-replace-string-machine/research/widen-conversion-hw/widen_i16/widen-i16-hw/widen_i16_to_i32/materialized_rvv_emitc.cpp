#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_pre_realized_body_widen_i16_to_i32_kernel_pre_realized_body_rvv_widen_i16_to_i32(const int16_t* v1, int32_t* v2, size_t v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v3);
  for (size_t v5 = 0; v5 < v3; v5 += v4) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v6 = v3 - v5;
    size_t v7 = __riscv_vsetvl_e32m1(v6);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_i16mf2
    const int16_t* v8 = v1 + v5;
    vint16mf2_t v9 = __riscv_vle16_v_i16mf2(v8, v7);
    // tcrv_emitc.source_op=tcrv_rvv.widening_convert role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i32m1
    vint32m1_t v10 = __riscv_vwcvt_x_x_v_i32m1(v9, v7);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v11 = v2 + v5;
    __riscv_vse32_v_i32m1(v11, v10, v7);
  }
  return;
}


