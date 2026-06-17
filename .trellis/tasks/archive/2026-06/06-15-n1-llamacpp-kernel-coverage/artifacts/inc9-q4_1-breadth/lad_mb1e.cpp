#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb1e(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 20;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 36;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    const uint8_t* v15 = v10 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v15);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v19;
    v19 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v20 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v21 = v10 + 4;
    const uint8_t* v22 = v21 + 0;
    const uint8_t* v23 = (const uint8_t*) v22;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v24 = __riscv_vle8_v_u8m1(v23, v20);
    const uint8_t* v25 = v12 + 4;
    const uint8_t* v26 = v25 + 0;
    const int8_t* v27 = (const int8_t*) v26;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v28 = __riscv_vle8_v_i8m1(v27, v20);
    const uint8_t* v29 = v12 + 20;
    const uint8_t* v30 = v29 + 0;
    const int8_t* v31 = (const int8_t*) v30;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v32 = __riscv_vle8_v_i8m1(v31, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v33 = __riscv_vand_vx_u8m1(v24, 0x0F, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v34 = __riscv_vsrl_vx_u8m1(v24, 0x04, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v35 = __riscv_vreinterpret_v_u8m1_i8m1(v33);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v36 = __riscv_vreinterpret_v_u8m1_i8m1(v34);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v37 = __riscv_vwmul_vv_i16m2(v35, v28, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v38 = __riscv_vwmacc_vv_i16m2(v37, v36, v32, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v39 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v40 = __riscv_vwredsum_vs_i16m2_i32m1(v38, v39, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v41 = __riscv_vmv_x_s_i32m1_i32(v40);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v19 = v41;
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v42 = v19;
    float v43 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v43 + ((v13 * v14) * (float) v42 + v16 * v18);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v44 = v6;
  v2[0] = v44;
  return;
}


