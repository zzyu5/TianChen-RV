#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_mxfp4_kernel_dequant_mxfp4(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_dequant_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
  vint8m1_t v5 = __riscv_vle8_v_i8m1(weft_dequant_mxfp4_kvalues, 16);
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 32;
  for (size_t v7 = 0; v7 < v6; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v8 = v7 * 17;
    const uint8_t* v9 = v2 + v8;
    size_t v10 = v7 * 32;
    float* v11 = v3 + v10;
    float* v12 = (float*) v11;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=mxfp4_decode
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_exponent_load
    const uint8_t* v13 = (const uint8_t*) v9;
    const uint8_t v14 = v13[0];
    uint32_t v15 = (uint32_t) v14;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
    bool v16 = v15 < 2;
    uint32_t v17 = v15 & 0x1F;
    uint32_t v18 = 0x00200000u << v17;
    uint32_t v19 = v15 - 1;
    uint32_t v20 = v19 << 23;
    uint32_t v21 = v16 ? v18 : v20;
    uint32_t v22;
    v22 = v21;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=e8m0_reinterpret_float
    const uint32_t* v23 = &v22;
    const float* v24 = (const float*) v23;
    const float v25 = v24[0];
    float v26 = (float) v25;
    const uint8_t* v27 = v9 + 1;
    const uint8_t* v28 = (const uint8_t*) v27;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v30 = __riscv_vand_vx_u8m1(v29, 15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v31 = __riscv_vsrl_vx_u8m1(v29, 4, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v32 = __riscv_vrgather_vv_i8m1(v5, v30, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v33 = __riscv_vsext_vf4_i32m4(v32, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v34 = __riscv_vfcvt_f_x_v_f32m4(v33, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v35 = __riscv_vfmul_vf_f32m4(v34, v26, 16);
    float* v36 = v12 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v36, v35, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v37 = __riscv_vrgather_vv_i8m1(v5, v31, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf4_i32m4
    vint32m4_t v38 = __riscv_vsext_vf4_i32m4(v37, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v39 = __riscv_vfcvt_f_x_v_f32m4(v38, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v40 = __riscv_vfmul_vf_f32m4(v39, v26, 16);
    float* v41 = v12 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v41, v40, 16);
  }
  return;
}


