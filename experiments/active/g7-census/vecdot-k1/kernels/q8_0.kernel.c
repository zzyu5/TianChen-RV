#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
  size_t v10 = __riscv_vsetvl_e8m2(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v11;
  v11 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v12 = v1 / 32;
  for (size_t v13 = 0; v13 < v12; v13 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_x
    size_t v14 = v13 * 34;
    const uint8_t* v15 = v4 + v14;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_y
    size_t v16 = v13 * 34;
    const uint8_t* v17 = v6 + v16;
    // weft_emitc.local_variable=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v18;
    v18 = 0;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v19 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v20 = v15 + 2;
    const uint8_t* v21 = v20 + 0;
    const int8_t* v22 = (const int8_t*) v21;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v23 = __riscv_vle8_v_i8m2(v22, v19);
    const uint8_t* v24 = v17 + 2;
    const uint8_t* v25 = v24 + 0;
    const int8_t* v26 = (const int8_t*) v25;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v27 = __riscv_vle8_v_i8m2(v26, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v28 = __riscv_vwmul_vv_i16m4(v23, v27, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v29 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v30 = __riscv_vwredsum_vs_i16m4_i32m1(v28, v29, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v31 = __riscv_vmv_x_s_i32m1_i32(v30);
    // weft_emitc.assign target=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v18 = v31;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v32 = (float)*(const _Float16 *)(v15);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v33 = (float)*(const _Float16 *)(v17);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v34 = v18;
    float v35 = v11;
    float v36 = (float) v34;
    float v37 = v36 * v32;
    float v38 = v37 * v33;
    float v39 = v35 + v38;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v11 = v39;
  }
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v40 = v11;
  v2[0] = v40;
  return;
}


