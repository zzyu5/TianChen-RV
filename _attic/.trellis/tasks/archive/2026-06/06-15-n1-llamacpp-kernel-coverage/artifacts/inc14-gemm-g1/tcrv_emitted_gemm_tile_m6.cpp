#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_gemm_m6_gemm_m6_v(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v7[6];
  v7[0] = 0.0f;
  v7[1] = 0.0f;
  v7[2] = 0.0f;
  v7[3] = 0.0f;
  v7[4] = 0.0f;
  v7[5] = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v8 = v1 / 32;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v10 = v9 * 18;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v13 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v14 = v11 + 2;
    const int8_t* v15 = (const int8_t*) v14;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v16 = __riscv_vle8_v_i8m1(v15, v13);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
    vint8m1_t v17 = __riscv_vxor_vx_i8m1(v16, 0x88, v13);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
    vint8m1_t v18 = __riscv_vsll_vx_i8m1(v17, 4, v13);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v19 = __riscv_vsra_vx_i8m1(v18, 4, v13);
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
    vint8m1_t v20 = __riscv_vsra_vx_i8m1(v17, 4, v13);
    for (size_t v21 = 0; v21 < 6; v21 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=column_base_y
      size_t v22 = v21 * v5;
      const uint8_t* v23 = v4 + v22;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
      size_t v24 = v9 * 34;
      const uint8_t* v25 = v23 + v24;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
      float v26 = (float)*(const _Float16 *)(v25);
      const uint8_t* v27 = v25 + 2;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v13);
      const uint8_t* v30 = v25 + 18;
      const int8_t* v31 = (const int8_t*) v30;
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v32 = __riscv_vle8_v_i8m1(v31, v13);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v33 = __riscv_vwmul_vv_i16m2(v19, v29, v13);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v34 = __riscv_vwmacc_vv_i16m2(v33, v20, v32, v13);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      vint32m1_t v35 = __riscv_vmv_v_x_i32m1(0, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v36 = __riscv_vwredsum_vs_i16m2_i32m1(v34, v35, v13);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v37 = __riscv_vmv_x_s_i32m1_i32(v36);
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
      float v38 = v7[v21];
      // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface
      v7[v21] = v38 + ((float) v37 * v12) * v26;
    }
  }
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v39 = v7[0];
  v2[0] = v39;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v40 = v7[1];
  v2[1] = v40;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v41 = v7[2];
  v2[2] = v41;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v42 = v7[3];
  v2[3] = v42;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v43 = v7[4];
  v2[4] = v43;
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm_tile role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v44 = v7[5];
  v2[5] = v44;
  return;
}


