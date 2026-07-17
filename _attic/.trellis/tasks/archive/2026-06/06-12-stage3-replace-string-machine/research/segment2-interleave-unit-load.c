#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_seg2_interleave_kernel_explicit_rvv_seg2_interleave(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v7 = v4 - v6;
    size_t v8 = __riscv_vsetvl_e32m1(v7);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v9 = v1 + v6;
    vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
    const int32_t* v11 = v2 + v6;
    vint32m1_t v12 = __riscv_vle32_v_i32m1(v11, v8);
    // tcrv_emitc.source_op=tcrv_rvv.segment2_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vcreate_v_i32m1x2
    vint32m1x2_t v13 = __riscv_vcreate_v_i32m1x2(v10, v12);
    // tcrv_emitc.source_op=tcrv_rvv.segment2_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsseg2e32_v_i32m1x2
    size_t v14 = v6 * 2;
    int32_t* v15 = v3 + v14;
    __riscv_vsseg2e32_v_i32m1x2(v15, v13, v8);
  }
  return;
}


