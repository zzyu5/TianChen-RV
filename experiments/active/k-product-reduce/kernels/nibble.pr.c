#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_rvv_unsigned_nibble_q8_1_integer_core_kernel_rvv_unsigned_nibble_q8_1_integer_core(const uint8_t* v1, const int8_t* v2, const int8_t* v3, const int32_t* v4, int32_t* v5, size_t v6) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
  size_t v7 = __riscv_vsetvl_e8m1(v6);
  // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v8 = v4[0];
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(v8, 1);
  // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v5, v9, 1);
  for (size_t v10 = 0; v10 < v6; v10 += v7) {
    // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v11 = v6 - v10;
    size_t v12 = __riscv_vsetvl_e8m1(v11);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    const uint8_t* v13 = v1 + v10;
    vuint8m1_t v14 = __riscv_vle8_v_u8m1(v13, v12);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    const int8_t* v15 = v2 + v10;
    vint8m1_t v16 = __riscv_vle8_v_i8m1(v15, v12);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    const int8_t* v17 = v3 + v10;
    vint8m1_t v18 = __riscv_vle8_v_i8m1(v17, v12);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v19 = __riscv_vand_vx_u8m1(v14, 0x0F, v12);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v20 = __riscv_vsrl_vx_u8m1(v14, 0x04, v12);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v21 = __riscv_vreinterpret_v_u8m1_i8m1(v19);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v22 = __riscv_vreinterpret_v_u8m1_i8m1(v20);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v23 = __riscv_vwmul_vv_i16m2(v21, v16, v12);
    // weft_emitc.source_op=weft_rvv.unsigned_nibble_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v24 = __riscv_vwmacc_vv_i16m2(v23, v22, v18, v12);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v25 = v5[0];
    vint32m1_t v26 = __riscv_vmv_v_x_i32m1(v25, 1);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v27 = __riscv_vwredsum_vs_i16m2_i32m1(v24, v26, v12);
    // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v5, v27, 1);
  }
  return;
}


