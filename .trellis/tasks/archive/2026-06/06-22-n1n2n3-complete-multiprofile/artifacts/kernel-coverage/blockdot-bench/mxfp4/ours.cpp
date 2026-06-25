#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_mxfp4_q8_0_kernel_ggml_vec_dot_mxfp4_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const int8_t tcrv_mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=codebook_table_load
  vint8m1_t v8 = __riscv_vle8_v_i8m1(tcrv_mxfp4_kvalues, 16);
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v10 = v9 * 17;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v12 = v9 * 34;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=e8m0_exponent_load
    const uint8_t* v14 = (const uint8_t*) v11;
    const uint8_t v15 = v14[0];
    uint32_t v16 = (uint32_t) v15;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=e8m0_to_fp32_half_bits
    bool v17 = v16 < 2;
    uint32_t v18 = v16 & 0x1F;
    uint32_t v19 = 0x00200000u << v18;
    uint32_t v20 = v16 - 1;
    uint32_t v21 = v20 << 23;
    uint32_t v22 = v17 ? v19 : v21;
    uint32_t v23;
    v23 = v22;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=e8m0_reinterpret_float
    const uint32_t* v24 = &v23;
    const float* v25 = (const float*) v24;
    const float v26 = v25[0];
    float v27 = (float) v26;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v28 = (float)*(const _Float16 *)(v13);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v29;
    v29 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v30 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v31 = v11 + 1;
    const uint8_t* v32 = v31 + 0;
    const uint8_t* v33 = (const uint8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v34 = __riscv_vle8_v_u8m1(v33, v30);
    const uint8_t* v35 = v13 + 2;
    const uint8_t* v36 = v35 + 0;
    const int8_t* v37 = (const int8_t*) v36;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v38 = __riscv_vle8_v_i8m1(v37, v30);
    const uint8_t* v39 = v13 + 18;
    const uint8_t* v40 = v39 + 0;
    const int8_t* v41 = (const int8_t*) v40;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v42 = __riscv_vle8_v_i8m1(v41, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v43 = __riscv_vand_vx_u8m1(v34, 0x0F, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v44 = __riscv_vsrl_vx_u8m1(v34, 0x04, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v45 = __riscv_vrgather_vv_i8m1(v8, v43, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vrgather_vv_i8m1
    vint8m1_t v46 = __riscv_vrgather_vv_i8m1(v8, v44, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v47 = __riscv_vwmul_vv_i16m2(v45, v38, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v48 = __riscv_vwmacc_vv_i16m2(v47, v46, v42, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v49 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v50 = __riscv_vwredsum_vs_i16m2_i32m1(v48, v49, v30);
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v51 = __riscv_vmv_x_s_i32m1_i32(v50);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v29 = v51;
    // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v52 = v29;
    float v53 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v53 + (float) v52 * (v27 * v28);
  }
  // tcrv_emitc.source_op=tcrv_rvv.mxfp4_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v54 = v6;
  v2[0] = v54;
  return;
}


