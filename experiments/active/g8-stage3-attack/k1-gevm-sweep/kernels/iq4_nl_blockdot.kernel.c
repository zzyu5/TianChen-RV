#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, const int32_t* v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  static const int8_t weft_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v7;
  v7 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v8 = v1 / 32;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v9 = __riscv_vle8_v_i8m1(weft_iq4_nl_kvalues, 16);
  for (size_t v10 = 0; v10 < v8; v10 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 18;
    const uint8_t* v12 = v3 + v11;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // weft_emitc.local_variable=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v18 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = v19 + 0;
    const uint8_t* v21 = (const uint8_t*) v20;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v22 = __riscv_vle8_v_u8m1(v21, v18);
    const uint8_t* v23 = v14 + 2;
    const uint8_t* v24 = v23 + 0;
    const int8_t* v25 = (const int8_t*) v24;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v26 = __riscv_vle8_v_i8m1(v25, v18);
    const uint8_t* v27 = v14 + 18;
    const uint8_t* v28 = v27 + 0;
    const int8_t* v29 = (const int8_t*) v28;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v30 = __riscv_vle8_v_i8m1(v29, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v31 = __riscv_vand_vx_u8m1(v22, 0x0F, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v32 = __riscv_vsrl_vx_u8m1(v22, 0x04, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v33 = __riscv_vrgather_vv_i8m1(v9, v31, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v34 = __riscv_vrgather_vv_i8m1(v9, v32, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v35 = __riscv_vwmul_vv_i16m2(v33, v26, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v36 = __riscv_vwmacc_vv_i16m2(v35, v34, v30, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v37 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v38 = __riscv_vwredsum_vs_i16m2_i32m1(v36, v37, v18);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v39 = __riscv_vmv_x_s_i32m1_i32(v38);
    // weft_emitc.assign target=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v17 = v39;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v40 = v17;
    float v41 = v7;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v7 = v41 + (float) v40 * (v15 * v16);
  }
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v42 = v7;
  v2[0] = v42;
  return;
}


