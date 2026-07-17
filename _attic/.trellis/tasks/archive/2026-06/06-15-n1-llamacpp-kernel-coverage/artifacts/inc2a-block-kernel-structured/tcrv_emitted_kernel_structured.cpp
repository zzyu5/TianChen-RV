#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10;
  v10 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v11 = v1 / 32;
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v13 = v12 * 18;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v15 = v12 * 34;
    const uint8_t* v16 = v6 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v17 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v16);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v19;
    v19 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v20 = __riscv_vsetvl_e32m1(16);
    for (size_t v21 = 0; v21 < 16; v21 += v20) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v22 = 16 - v21;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      const uint8_t* v24 = v14 + 2;
      const uint8_t* v25 = v24 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v26 = __riscv_vle8_v_i8mf4(v25, v23);
      const uint8_t* v27 = v16 + 2;
      const uint8_t* v28 = v27 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v23);
      const uint8_t* v30 = v16 + 18;
      const uint8_t* v31 = v30 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v32 = __riscv_vle8_v_i8mf4(v31, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8mf4
      vint8mf4_t v33 = __riscv_vxor_vx_i8mf4(v26, 0x88, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf4
      vint8mf4_t v34 = __riscv_vsll_vx_i8mf4(v33, 4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
      vint8mf4_t v35 = __riscv_vsra_vx_i8mf4(v34, 4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf4
      vint8mf4_t v36 = __riscv_vsra_vx_i8mf4(v33, 4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v37 = __riscv_vwmul_vv_i16mf2(v35, v29, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v38 = __riscv_vwmacc_vv_i16mf2(v37, v36, v32, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v39 = v19;
      vint32m1_t v40 = __riscv_vmv_v_x_i32m1(v39, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v41 = __riscv_vwredsum_vs_i16mf2_i32m1(v38, v40, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v42 = __riscv_vmv_x_s_i32m1_i32(v41);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v19 = v42;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v43 = v19;
    float v44 = (float) v43;
    float v45 = v44 * v17;
    float v46 = v45 * v18;
    float v47 = v10;
    float v48 = v47 + v46;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v48;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v49 = v10;
  v2[0] = v49;
  return;
}


