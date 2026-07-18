#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_rvv_five_bit_q5_0_integer_core_kernel_rvv_five_bit_q5_0_integer_core(const uint8_t* v1, const int8_t* v2, const int8_t* v3, const int32_t* v4, int32_t* v5, size_t v6) {
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
    const uint8_t* v19 = v1 + 2;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    const uint8_t* v21 = v1 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v23 = __riscv_vand_vx_u8m1(v14, 0x0F, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v24 = __riscv_vsrl_vx_u8m1(v14, 0x04, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v25 = __riscv_vid_v_u16m2(v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v26 = __riscv_vadd_vx_u16m2(v25, 0, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v27 = __riscv_vmv_v_x_u16m2(v20, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v28 = __riscv_vsrl_vv_u16m2(v27, v26, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v29 = __riscv_vand_vx_u16m2(v28, 0x1, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v30 = __riscv_vsll_vx_u16m2(v29, 0x4, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v31 = __riscv_vncvt_x_x_w_u8m1(v30, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v32 = __riscv_vid_v_u16m2(v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v33 = __riscv_vadd_vx_u16m2(v32, 0, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v34 = __riscv_vmv_v_x_u16m2(v22, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v35 = __riscv_vsrl_vv_u16m2(v34, v33, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v36 = __riscv_vand_vx_u16m2(v35, 0x1, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v37 = __riscv_vsll_vx_u16m2(v36, 0x4, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v38 = __riscv_vncvt_x_x_w_u8m1(v37, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v39 = __riscv_vor_vv_u8m1(v23, v31, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v40 = __riscv_vor_vv_u8m1(v24, v38, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v41 = __riscv_vreinterpret_v_u8m1_i8m1(v39);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v42 = __riscv_vsub_vx_i8m1(v41, 16, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v43 = __riscv_vreinterpret_v_u8m1_i8m1(v40);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
    vint8m1_t v44 = __riscv_vsub_vx_i8m1(v43, 16, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v45 = __riscv_vwmul_vv_i16m2(v42, v16, v12);
    // weft_emitc.source_op=weft_rvv.five_bit_offset_binary_x_i8_product role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v46 = __riscv_vwmacc_vv_i16m2(v45, v44, v18, v12);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v47 = v5[0];
    vint32m1_t v48 = __riscv_vmv_v_x_i32m1(v47, 1);
    // weft_emitc.source_op=weft_rvv.standalone_reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v49 = __riscv_vwredsum_vs_i16m2_i32m1(v46, v48, v12);
    // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
    __riscv_vse32_v_i32m1(v5, v49, 1);
  }
  return;
}


