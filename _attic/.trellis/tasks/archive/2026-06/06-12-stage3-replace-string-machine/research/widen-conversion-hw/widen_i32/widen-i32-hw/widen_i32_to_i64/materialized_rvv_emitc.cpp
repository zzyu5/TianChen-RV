#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_selected_body_widen_i32_to_i64_kernel_explicit_selected_body_rvv_widen_i32_to_i64(const int32_t* v1, int64_t* v2, size_t v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e64m2
  size_t v4 = __riscv_vsetvl_e64m2(v3);
  for (size_t v5 = 0; v5 < v3; v5 += v4) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e64m2
    size_t v6 = v3 - v5;
    size_t v7 = __riscv_vsetvl_e64m2(v6);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v8 = v1 + v5;
    vint32m1_t v9 = __riscv_vle32_v_i32m1(v8, v7);
    // tcrv_emitc.source_op=tcrv_rvv.widening_convert role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i64m2
    vint64m2_t v10 = __riscv_vwcvt_x_x_v_i64m2(v9, v7);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse64_v_i64m2
    int64_t* v11 = v2 + v5;
    __riscv_vse64_v_i64m2(v11, v10, v7);
  }
  return;
}


