#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_explicit_strided_dot_kernel_rvv_explicit_strided_input_dot(const int16_t* v1, const int16_t* v2, const int32_t* v3, int32_t* v4, size_t v5, size_t v6, size_t v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v5);
  // tcrv_emitc.source_op=tcrv_rvv.widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v9 = v3[0];
  vint32m1_t v10 = __riscv_vmv_v_x_i32m1(v9, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v4, v10, 1);
  for (size_t v11 = 0; v11 < v5; v11 += v8) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v12 = v5 - v11;
    size_t v13 = __riscv_vsetvl_e32m1(v12);
    // tcrv_emitc.source_op=tcrv_rvv.strided_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_i16mf2
    size_t v14 = v11 * v6;
    const int16_t* v15 = v1 + v14;
    ptrdiff_t v16 = (ptrdiff_t) v6;
    ptrdiff_t v17 = (ptrdiff_t) 2;
    ptrdiff_t v18 = v16 * v17;
    vint16mf2_t v19 = __riscv_vlse16_v_i16mf2(v15, v18, v13);
    // tcrv_emitc.source_op=tcrv_rvv.strided_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlse16_v_i16mf2
    size_t v20 = v11 * v7;
    const int16_t* v21 = v2 + v20;
    ptrdiff_t v22 = (ptrdiff_t) v7;
    ptrdiff_t v23 = (ptrdiff_t) 2;
    ptrdiff_t v24 = v22 * v23;
    vint16mf2_t v25 = __riscv_vlse16_v_i16mf2(v21, v24, v13);
    // tcrv_emitc.source_op=tcrv_rvv.widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i32m1
    vint32m1_t v26 = __riscv_vwmul_vv_i32m1(v19, v25, v13);
    // tcrv_emitc.source_op=tcrv_rvv.widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v27 = v4[0];
    vint32m1_t v28 = __riscv_vmv_v_x_i32m1(v27, 1);
    // tcrv_emitc.source_op=tcrv_rvv.widening_dot_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
    vint32m1_t v29 = __riscv_vredsum_vs_i32m1_i32m1(v26, v28, v13);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v4, v29, 1);
  }
  return;
}


