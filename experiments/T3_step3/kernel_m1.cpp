#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
  size_t v10 = __riscv_vsetvl_e8m1(v1);
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
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v21 = __riscv_vsetvl_e8m1(32);
    for (size_t v22 = 0; v22 < 32; v22 += v21) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v23 = 32 - v22;
      size_t v24 = __riscv_vsetvl_e8m1(v23);
      const uint8_t* v25 = v15 + 2;
      const uint8_t* v26 = v25 + v22;
      const int8_t* v27 = (const int8_t*) v26;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v28 = __riscv_vle8_v_i8m1(v27, v24);
      const uint8_t* v29 = v17 + 2;
      const uint8_t* v30 = v29 + v22;
      const int8_t* v31 = (const int8_t*) v30;
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v32 = __riscv_vle8_v_i8m1(v31, v24);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v33 = __riscv_vwmul_vv_i16m2(v28, v32, v24);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v34 = v20;
      vint32m1_t v35 = __riscv_vmv_v_x_i32m1(v34, 1);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v36 = __riscv_vwredsum_vs_i16m2_i32m1(v33, v35, v24);
      // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v37 = __riscv_vmv_x_s_i32m1_i32(v36);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
      v20 = v37;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v38 = v20;
    float v39 = v11;
    float v40 = (float) v38;
    float v41 = v40 * v18;
    float v42 = v41 * v19;
    float v43 = v39 + v42;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
    v11 = v43;
  }
  // tcrv_emitc.source_op=tcrv_rvv.typed_flat_block_dot_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v44 = v11;
  v2[0] = v44;
  return;
}


