#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_seg2_deinterleave_kernel_explicit_rvv_seg2_deinterleave(const int32_t* v1, int32_t* v2, int32_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.segment2_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlseg2e32_v_i32m1x2
    size_t v9 = v6 * 2;
    const int32_t* v10 = v1 + v9;
    vint32m1x2_t v11 = __riscv_vlseg2e32_v_i32m1x2(v10, v8);
    // tcrv_emitc.source_op=tcrv_rvv.move role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m1x2_i32m1
    vint32m1_t v12 = __riscv_vget_v_i32m1x2_i32m1(v11, 0);
    // tcrv_emitc.source_op=tcrv_rvv.move role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vget_v_i32m1x2_i32m1
    vint32m1_t v13 = __riscv_vget_v_i32m1x2_i32m1(v11, 1);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v14 = v2 + v6;
    __riscv_vse32_v_i32m1(v14, v12, v8);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    int32_t* v15 = v3 + v6;
    __riscv_vse32_v_i32m1(v15, v13, v8);
  }
  return;
}


