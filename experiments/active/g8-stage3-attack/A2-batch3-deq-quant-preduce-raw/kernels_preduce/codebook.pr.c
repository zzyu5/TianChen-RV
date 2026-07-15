#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_rvv_codebook_q8_0_integer_core_kernel_rvv_codebook_q8_0_integer_core(const uint8_t* v1, const int8_t* v2, const int8_t* v3, const int32_t* v4, int32_t* v5, size_t v6) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v7 = __riscv_vsetvl_e32m1(v6);
  // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
  const int32_t v8 = v4[0];
  vint32m1_t v9 = __riscv_vmv_v_x_i32m1(v8, 1);
  // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
  __riscv_vse32_v_i32m1(v5, v9, 1);
  for (size_t v10 = 0; v10 < v6; v10 += v7) {
    // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v11 = v6 - v10;
    size_t v12 = __riscv_vsetvl_e32m1(v11);
    static const int8_t weft_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    // weft_emitc.source_op=weft_rvv.codebook_table_broadcast role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
    vint8m1_t v13 = __riscv_vle8_v_i8m1(weft_iq4_nl_kvalues, 16);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    const uint8_t* v14 = v1 + v10;
    vuint8m1_t v15 = __riscv_vle8_v_u8m1(v14, v12);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    const int8_t* v16 = v2 + v10;
    vint8m1_t v17 = __riscv_vle8_v_i8m1(v16, v12);
    // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    const int8_t* v18 = v3 + v10;
    vint8m1_t v19 = __riscv_vle8_v_i8m1(v18, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v20 = __riscv_vand_vx_u8m1(v15, 0x0F, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v21 = __riscv_vsrl_vx_u8m1(v15, 0x04, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v22 = __riscv_vrgather_vv_i8m1(v13, v20, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v23 = __riscv_vrgather_vv_i8m1(v13, v21, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v24 = __riscv_vwmul_vv_i16m2(v22, v17, v12);
    // weft_emitc.source_op=weft_rvv.codebook_gather_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v25 = __riscv_vwmacc_vv_i16m2(v24, v23, v19, v12);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v26 = v5[0];
    vint32m1_t v27 = __riscv_vmv_v_x_i32m1(v26, 1);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v28 = __riscv_vwredsum_vs_i16m2_i32m1(v25, v27, v12);
    // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v5, v28, 1);
  }
  return;
}


