#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
  size_t v10 = __riscv_vsetvl_e8m2(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v11;
  v11 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v12 = v1 / 32;
  for (size_t v13 = 0; v13 < v12; v13 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v14 = v13 * 34;
    const uint8_t* v15 = v4 + v14;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v16 = v13 * 34;
    const uint8_t* v17 = v6 + v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v15);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v19 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v20;
    v20 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v21 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v22 = v15 + 2;
    const uint8_t* v23 = v22 + 0;
    const int8_t* v24 = (const int8_t*) v23;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v25 = __riscv_vle8_v_i8m2(v24, v21);
    const uint8_t* v26 = v17 + 2;
    const uint8_t* v27 = v26 + 0;
    const int8_t* v28 = (const int8_t*) v27;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v29 = __riscv_vle8_v_i8m2(v28, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v30 = __riscv_vwmul_vv_i16m4(v25, v29, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v31 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v32 = __riscv_vwredsum_vs_i16m4_i32m1(v30, v31, v21);
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v33 = __riscv_vmv_x_s_i32m1_i32(v32);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v20 = v33;
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v34 = v20;
    float v35 = v11;
    float v36 = (float) v34;
    float v37 = v36 * v18;
    float v38 = v37 * v19;
    float v39 = v35 + v38;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v39;
  }
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v40 = v11;
  v2[0] = v40;
  return;
}


