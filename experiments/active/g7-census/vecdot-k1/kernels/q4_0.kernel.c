#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
  size_t v10 = __riscv_vsetvl_e8m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v11;
  v11 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v12 = v1 / 32;
  for (size_t v13 = 0; v13 < v12; v13 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_x
    size_t v14 = v13 * 18;
    const uint8_t* v15 = v4 + v14;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_y
    size_t v16 = v13 * 34;
    const uint8_t* v17 = v6 + v16;
    // weft_emitc.local_variable=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v18;
    v18 = 0;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v19 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v20 = v15 + 2;
    const uint8_t* v21 = v20 + 0;
    const int8_t* v22 = (const int8_t*) v21;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v23 = __riscv_vle8_v_i8m1(v22, v19);
    const uint8_t* v24 = v17 + 2;
    const uint8_t* v25 = v24 + 0;
    const int8_t* v26 = (const int8_t*) v25;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v27 = __riscv_vle8_v_i8m1(v26, v19);
    const uint8_t* v28 = v17 + 18;
    const uint8_t* v29 = v28 + 0;
    const int8_t* v30 = (const int8_t*) v29;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v31 = __riscv_vle8_v_i8m1(v30, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v32 = __riscv_vxor_vx_i8m1(v23, 0x88, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v33 = __riscv_vsll_vx_i8m1(v32, 4, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v34 = __riscv_vsra_vx_i8m1(v33, 4, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v35 = __riscv_vsra_vx_i8m1(v32, 4, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v36 = __riscv_vwmul_vv_i16m2(v34, v27, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v37 = __riscv_vwmacc_vv_i16m2(v36, v35, v31, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v38 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v39 = __riscv_vwredsum_vs_i16m2_i32m1(v37, v38, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v40 = __riscv_vmv_x_s_i32m1_i32(v39);
    // weft_emitc.assign target=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v18 = v40;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v15);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v17);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v43 = v18;
    float v44 = v11;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v11 = v44 + ((float) v43 * v41) * v42;
  }
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v45 = v11;
  v2[0] = v45;
  return;
}


