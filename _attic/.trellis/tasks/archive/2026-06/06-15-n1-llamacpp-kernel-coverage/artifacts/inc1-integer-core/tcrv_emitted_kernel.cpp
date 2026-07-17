#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(const int8_t* v1, const int8_t* v2, const int8_t* v3, const int32_t* v4, int32_t* v5, size_t v6) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v7 = __riscv_vsetvl_e32m1(v6);
  // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v8 = v4[0];
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(v8, 1);
  // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v5, v9, 1);
  for (size_t v10 = 0; v10 < v6; v10 += v7) {
    // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v11 = v6 - v10;
    size_t v12 = __riscv_vsetvl_e32m1(v11);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v13 = v1 + v10;
    vint8mf4_t v14 = __riscv_vle8_v_i8mf4(v13, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v15 = v2 + v10;
    vint8mf4_t v16 = __riscv_vle8_v_i8mf4(v15, v12);
    // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
    const int8_t* v17 = v3 + v10;
    vint8mf4_t v18 = __riscv_vle8_v_i8mf4(v17, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8mf4
    vint8mf4_t v19 = __riscv_vxor_vx_i8mf4(v14, 0x88, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf4
    vint8mf4_t v20 = __riscv_vsll_vx_i8mf4(v19, 4, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
    vint8mf4_t v21 = __riscv_vsra_vx_i8mf4(v20, 4, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
    vint8mf4_t v22 = __riscv_vsra_vx_i8mf4(v19, 4, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
    vint16mf2_t v23 = __riscv_vwmul_vv_i16mf2(v21, v16, v12);
    // tcrv_emitc.source_op=tcrv_rvv.packed_i4_offset_binary_x_i8_product role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
    vint16mf2_t v24 = __riscv_vwmacc_vv_i16mf2(v23, v22, v18, v12);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v25 = v5[0];
    vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v25, 1);
    // tcrv_emitc.source_op=tcrv_rvv.standalone_reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
    vint32m1_t v27 = __riscv_vwredsum_vs_i16mf2_i32m1(v24, v26, v12);
    // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v5, v27, 1);
  }
  return;
}


