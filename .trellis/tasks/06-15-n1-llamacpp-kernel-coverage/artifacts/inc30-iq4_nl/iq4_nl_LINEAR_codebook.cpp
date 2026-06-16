#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int8_t tcrv_iq4_nl_kvalues[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v8 = __riscv_vle8_v_i8m1(tcrv_iq4_nl_kvalues, 16);
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v10 = v9 * 18;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v12 = v9 * 34;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v13);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v16;
    v16 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v17 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v18 = v11 + 2;
    const uint8_t* v19 = v18 + 0;
    const uint8_t* v20 = (const uint8_t*) v19;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v21 = __riscv_vle8_v_u8m1(v20, v17);
    const uint8_t* v22 = v13 + 2;
    const uint8_t* v23 = v22 + 0;
    const int8_t* v24 = (const int8_t*) v23;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v25 = __riscv_vle8_v_i8m1(v24, v17);
    const uint8_t* v26 = v13 + 18;
    const uint8_t* v27 = v26 + 0;
    const int8_t* v28 = (const int8_t*) v27;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v30 = __riscv_vand_vx_u8m1(v21, 0x0F, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v31 = __riscv_vsrl_vx_u8m1(v21, 0x04, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v32 = __riscv_vrgather_vv_i8m1(v8, v30, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v33 = __riscv_vrgather_vv_i8m1(v8, v31, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v34 = __riscv_vwmul_vv_i16m2(v32, v25, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v35 = __riscv_vwmacc_vv_i16m2(v34, v33, v29, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v36 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v37 = __riscv_vwredsum_vs_i16m2_i32m1(v35, v36, v17);
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v38 = __riscv_vmv_x_s_i32m1_i32(v37);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v16 = v38;
    // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v39 = v16;
    float v40 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v40 + (float) v39 * (v14 * v15);
  }
  // tcrv_emitc.source_op=tcrv_rvv.iq4_nl_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v41 = v6;
  v2[0] = v41;
  return;
}


