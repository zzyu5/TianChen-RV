#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 22;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v15 = v10 + 2;
    uint32_t v16 = (uint16_t)*(const uint16_t *)(v15);
    const uint8_t* v17 = v10 + 4;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v19;
    v19 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v20 = __riscv_vsetvl_e32m1(16);
    for (size_t v21 = 0; v21 < 16; v21 += v20) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v22 = 16 - v21;
      size_t v23 = __riscv_vsetvl_e32m1(v22);
      const uint8_t* v24 = v10 + 6;
      const uint8_t* v25 = v24 + v21;
      const uint8_t* v26 = (const uint8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v27 = __riscv_vle8_v_u8mf4(v26, v23);
      const uint8_t* v28 = v12 + 2;
      const uint8_t* v29 = v28 + v21;
      const int8_t* v30 = (const int8_t*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v31 = __riscv_vle8_v_i8mf4(v30, v23);
      const uint8_t* v32 = v12 + 18;
      const uint8_t* v33 = v32 + v21;
      const int8_t* v34 = (const int8_t*) v33;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v36 = __riscv_vand_vx_u8mf4(v27, 0x0F, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v37 = __riscv_vsrl_vx_u8mf4(v27, 0x04, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v38 = __riscv_vid_v_u16mf2(v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v39 = __riscv_vadd_vx_u16mf2(v38, v21, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v40 = __riscv_vmv_v_x_u16mf2(v16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v41 = __riscv_vsrl_vv_u16mf2(v40, v39, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v42 = __riscv_vand_vx_u16mf2(v41, 0x1, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v43 = __riscv_vsll_vx_u16mf2(v42, 0x4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v44 = __riscv_vncvt_x_x_w_u8mf4(v43, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v45 = __riscv_vid_v_u16mf2(v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v46 = __riscv_vadd_vx_u16mf2(v45, v21, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v47 = __riscv_vmv_v_x_u16mf2(v18, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v48 = __riscv_vsrl_vv_u16mf2(v47, v46, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v49 = __riscv_vand_vx_u16mf2(v48, 0x1, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v50 = __riscv_vsll_vx_u16mf2(v49, 0x4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v51 = __riscv_vncvt_x_x_w_u8mf4(v50, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v52 = __riscv_vor_vv_u8mf4(v36, v44, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v53 = __riscv_vor_vv_u8mf4(v37, v51, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v54 = __riscv_vreinterpret_v_u8mf4_i8mf4(v52);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v55 = __riscv_vsub_vx_i8mf4(v54, 16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v56 = __riscv_vreinterpret_v_u8mf4_i8mf4(v53);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v57 = __riscv_vsub_vx_i8mf4(v56, 16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v58 = __riscv_vwmul_vv_i16mf2(v55, v31, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v59 = __riscv_vwmacc_vv_i16mf2(v58, v57, v35, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v60 = v19;
      vint32m1_t v61 = __riscv_vmv_v_x_i32m1(v60, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v62 = __riscv_vwredsum_vs_i16mf2_i32m1(v59, v61, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v63 = __riscv_vmv_x_s_i32m1_i32(v62);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v19 = v63;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v64 = v19;
    float v65 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v65 + (v13 * v14) * (float) v64;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v66 = v6;
  v2[0] = v66;
  return;
}


