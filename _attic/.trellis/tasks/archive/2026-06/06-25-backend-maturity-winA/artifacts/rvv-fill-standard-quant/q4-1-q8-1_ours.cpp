#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v20 = __riscv_vsetvl_e32m1(16);
    for (size_t v21 = 0; v21 < 16; v21 += v20) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v22 = 16 - v21;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      const uint8_t* v24 = v10 + 4;
      const uint8_t* v25 = v24 + v21;
      const uint8_t* v26 = (const uint8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v27 = __riscv_vle8_v_u8mf4(v26, v23);
      const uint8_t* v28 = v12 + 4;
      const uint8_t* v29 = v28 + v21;
      const int8_t* v30 = (const int8_t*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v23);
      const uint8_t* v32 = v12 + 20;
      const uint8_t* v33 = v32 + v21;
      const int8_t* v34 = (const int8_t*) v33;
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v36 = __riscv_vand_vx_u8mf4(v27, 0x0F, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v37 = __riscv_vsrl_vx_u8mf4(v27, 0x04, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v38 = __riscv_vreinterpret_v_u8mf4_i8mf4(v36);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v39 = __riscv_vreinterpret_v_u8mf4_i8mf4(v37);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v40 = __riscv_vwmul_vv_i16mf2(v38, v31, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v41 = __riscv_vwmacc_vv_i16mf2(v40, v39, v35, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v42 = v19;
      vint32m1_t v43 = __riscv_vmv_v_x_i32m1(v42, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v44 = __riscv_vwredsum_vs_i16mf2_i32m1(v41, v43, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v45 = __riscv_vmv_x_s_i32m1_i32(v44);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v19 = v45;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v46 = v19;
    float v47 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v47 + ((v13 * v14) * (float) v46 + v16 * v18);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v48 = v6;
  v2[0] = v48;
  return;
}


