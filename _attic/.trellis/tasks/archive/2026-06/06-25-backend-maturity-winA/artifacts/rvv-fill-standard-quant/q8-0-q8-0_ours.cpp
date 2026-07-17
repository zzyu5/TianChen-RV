#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 34;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v16 = __riscv_vsetvl_e8m2(32);
    for (size_t v17 = 0; v17 < 32; v17 += v16) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v18 = 32 - v17;
      size_t v19 = __riscv_vsetvl_e8m2(v18);
      const uint8_t* v20 = v10 + 2;
      const uint8_t* v21 = v20 + v17;
      const int8_t* v22 = (const int8_t*) v21;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v23 = __riscv_vle8_v_i8m2(v22, v19);
      const uint8_t* v24 = v12 + 2;
      const uint8_t* v25 = v24 + v17;
      const int8_t* v26 = (const int8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v27 = __riscv_vle8_v_i8m2(v26, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v28 = __riscv_vwmul_vv_i16m4(v23, v27, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v29 = v15;
      vint32m1_t v30 = __riscv_vmv_v_x_i32m1(v29, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v31 = __riscv_vwredsum_vs_i16m4_i32m1(v28, v30, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v32 = __riscv_vmv_x_s_i32m1_i32(v31);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v15 = v32;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v33 = v15;
    float v34 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v34 + (float) v33 * (v13 * v14);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v35 = v6;
  v2[0] = v35;
  return;
}


