#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_mf4_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_mf4_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 22;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v17 = v12 + 2;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    const uint8_t* v19 = v12 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v22 = __riscv_vsetvl_e32m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      const uint8_t* v26 = v12 + 6;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v29 = __riscv_vle8_v_u8mf4(v28, v25);
      const uint8_t* v30 = v14 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v25);
      const uint8_t* v34 = v14 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v38 = __riscv_vand_vx_u8mf4(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v39 = __riscv_vsrl_vx_u8mf4(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v40 = __riscv_vid_v_u16mf2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v41 = __riscv_vadd_vx_u16mf2(v40, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v42 = __riscv_vmv_v_x_u16mf2(v18, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v43 = __riscv_vsrl_vv_u16mf2(v42, v41, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v44 = __riscv_vand_vx_u16mf2(v43, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v45 = __riscv_vsll_vx_u16mf2(v44, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v46 = __riscv_vncvt_x_x_w_u8mf4(v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v47 = __riscv_vid_v_u16mf2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v48 = __riscv_vadd_vx_u16mf2(v47, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v49 = __riscv_vmv_v_x_u16mf2(v20, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v50 = __riscv_vsrl_vv_u16mf2(v49, v48, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v51 = __riscv_vand_vx_u16mf2(v50, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v52 = __riscv_vsll_vx_u16mf2(v51, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v53 = __riscv_vncvt_x_x_w_u8mf4(v52, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v54 = __riscv_vor_vv_u8mf4(v38, v46, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v55 = __riscv_vor_vv_u8mf4(v39, v53, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v56 = __riscv_vreinterpret_v_u8mf4_i8mf4(v54);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v57 = __riscv_vsub_vx_i8mf4(v56, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v58 = __riscv_vreinterpret_v_u8mf4_i8mf4(v55);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v59 = __riscv_vsub_vx_i8mf4(v58, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v60 = __riscv_vwmul_vv_i16mf2(v57, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v61 = __riscv_vwmacc_vv_i16mf2(v60, v59, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v21;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16mf2_i32m1(v61, v63, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v10 + 1;
    size_t v67 = v66 * 22;
    const uint8_t* v68 = v3 + v67;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v69 = v10 + 1;
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v72 = (float)*(const _Float16 *)(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v74 = v68 + 2;
    uint32_t v75 = (uint16_t)*(const uint16_t *)(v74);
    const uint8_t* v76 = v68 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v78;
    v78 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v79 = __riscv_vsetvl_e32m1(16);
    for (size_t v80 = 0; v80 < 16; v80 += v79) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v81 = 16 - v80;
      size_t v82 = __riscv_vsetvl_e32m1(v81);
      const uint8_t* v83 = v68 + 6;
      const uint8_t* v84 = v83 + v80;
      const uint8_t* v85 = (const uint8_t*) v84;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v86 = __riscv_vle8_v_u8mf4(v85, v82);
      const uint8_t* v87 = v71 + 2;
      const uint8_t* v88 = v87 + v80;
      const int8_t* v89 = (const int8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v90 = __riscv_vle8_v_i8mf4(v89, v82);
      const uint8_t* v91 = v71 + 18;
      const uint8_t* v92 = v91 + v80;
      const int8_t* v93 = (const int8_t*) v92;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v94 = __riscv_vle8_v_i8mf4(v93, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v95 = __riscv_vand_vx_u8mf4(v86, 0x0F, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v96 = __riscv_vsrl_vx_u8mf4(v86, 0x04, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v97 = __riscv_vid_v_u16mf2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v98 = __riscv_vadd_vx_u16mf2(v97, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v99 = __riscv_vmv_v_x_u16mf2(v75, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v100 = __riscv_vsrl_vv_u16mf2(v99, v98, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v101 = __riscv_vand_vx_u16mf2(v100, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v102 = __riscv_vsll_vx_u16mf2(v101, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v103 = __riscv_vncvt_x_x_w_u8mf4(v102, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v104 = __riscv_vid_v_u16mf2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v105 = __riscv_vadd_vx_u16mf2(v104, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v106 = __riscv_vmv_v_x_u16mf2(v77, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v107 = __riscv_vsrl_vv_u16mf2(v106, v105, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v108 = __riscv_vand_vx_u16mf2(v107, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v109 = __riscv_vsll_vx_u16mf2(v108, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v110 = __riscv_vncvt_x_x_w_u8mf4(v109, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v111 = __riscv_vor_vv_u8mf4(v95, v103, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v112 = __riscv_vor_vv_u8mf4(v96, v110, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v113 = __riscv_vreinterpret_v_u8mf4_i8mf4(v111);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v114 = __riscv_vsub_vx_i8mf4(v113, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v115 = __riscv_vreinterpret_v_u8mf4_i8mf4(v112);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v116 = __riscv_vsub_vx_i8mf4(v115, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v117 = __riscv_vwmul_vv_i16mf2(v114, v90, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v118 = __riscv_vwmacc_vv_i16mf2(v117, v116, v94, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v119 = v78;
      vint32m1_t v120 = __riscv_vmv_v_x_i32m1(v119, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v121 = __riscv_vwredsum_vs_i16mf2_i32m1(v118, v120, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v122 = __riscv_vmv_x_s_i32m1_i32(v121);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v78 = v122;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v123 = v21;
    float v124 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v124 + (v15 * v16) * (float) v123;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v125 = v78;
    float v126 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v126 + (v72 * v73) * (float) v125;
  }
  for (size_t v127 = v9; v127 < v7; v127 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v128 = v127 * 22;
    const uint8_t* v129 = v3 + v128;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v130 = v127 * 34;
    const uint8_t* v131 = v4 + v130;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v132 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v131);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v134 = v129 + 2;
    uint32_t v135 = (uint16_t)*(const uint16_t *)(v134);
    const uint8_t* v136 = v129 + 4;
    uint32_t v137 = (uint16_t)*(const uint16_t *)(v136);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v138;
    v138 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v139 = __riscv_vsetvl_e32m1(16);
    for (size_t v140 = 0; v140 < 16; v140 += v139) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v141 = 16 - v140;
      size_t v142 = __riscv_vsetvl_e32m1(v141);
      const uint8_t* v143 = v129 + 6;
      const uint8_t* v144 = v143 + v140;
      const uint8_t* v145 = (const uint8_t*) v144;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v146 = __riscv_vle8_v_u8mf4(v145, v142);
      const uint8_t* v147 = v131 + 2;
      const uint8_t* v148 = v147 + v140;
      const int8_t* v149 = (const int8_t*) v148;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v150 = __riscv_vle8_v_i8mf4(v149, v142);
      const uint8_t* v151 = v131 + 18;
      const uint8_t* v152 = v151 + v140;
      const int8_t* v153 = (const int8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v154 = __riscv_vle8_v_i8mf4(v153, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v155 = __riscv_vand_vx_u8mf4(v146, 0x0F, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v156 = __riscv_vsrl_vx_u8mf4(v146, 0x04, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v157 = __riscv_vid_v_u16mf2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v158 = __riscv_vadd_vx_u16mf2(v157, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v159 = __riscv_vmv_v_x_u16mf2(v135, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v160 = __riscv_vsrl_vv_u16mf2(v159, v158, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v161 = __riscv_vand_vx_u16mf2(v160, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v162 = __riscv_vsll_vx_u16mf2(v161, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v163 = __riscv_vncvt_x_x_w_u8mf4(v162, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v164 = __riscv_vid_v_u16mf2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v165 = __riscv_vadd_vx_u16mf2(v164, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v166 = __riscv_vmv_v_x_u16mf2(v137, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v167 = __riscv_vsrl_vv_u16mf2(v166, v165, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v168 = __riscv_vand_vx_u16mf2(v167, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v169 = __riscv_vsll_vx_u16mf2(v168, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v170 = __riscv_vncvt_x_x_w_u8mf4(v169, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v171 = __riscv_vor_vv_u8mf4(v155, v163, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v172 = __riscv_vor_vv_u8mf4(v156, v170, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v173 = __riscv_vreinterpret_v_u8mf4_i8mf4(v171);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v174 = __riscv_vsub_vx_i8mf4(v173, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v175 = __riscv_vreinterpret_v_u8mf4_i8mf4(v172);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v176 = __riscv_vsub_vx_i8mf4(v175, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v177 = __riscv_vwmul_vv_i16mf2(v174, v150, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v178 = __riscv_vwmacc_vv_i16mf2(v177, v176, v154, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v179 = v138;
      vint32m1_t v180 = __riscv_vmv_v_x_i32m1(v179, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v181 = __riscv_vwredsum_vs_i16mf2_i32m1(v178, v180, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v182 = __riscv_vmv_x_s_i32m1_i32(v181);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v138 = v182;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v183 = v138;
    float v184 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v184 + (v132 * v133) * (float) v183;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v185 = v6;
  v2[0] = v185;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_mf4_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 22;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v17 = v12 + 2;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    const uint8_t* v19 = v12 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v22 = __riscv_vsetvl_e32m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e32m1(v24);
      const uint8_t* v26 = v12 + 6;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v29 = __riscv_vle8_v_u8mf4(v28, v25);
      const uint8_t* v30 = v14 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v33 = __riscv_vle8_v_i8mf4(v32, v25);
      const uint8_t* v34 = v14 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v38 = __riscv_vand_vx_u8mf4(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v39 = __riscv_vsrl_vx_u8mf4(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v40 = __riscv_vid_v_u16mf2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v41 = __riscv_vadd_vx_u16mf2(v40, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v42 = __riscv_vmv_v_x_u16mf2(v18, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v43 = __riscv_vsrl_vv_u16mf2(v42, v41, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v44 = __riscv_vand_vx_u16mf2(v43, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v45 = __riscv_vsll_vx_u16mf2(v44, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v46 = __riscv_vncvt_x_x_w_u8mf4(v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v47 = __riscv_vid_v_u16mf2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v48 = __riscv_vadd_vx_u16mf2(v47, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v49 = __riscv_vmv_v_x_u16mf2(v20, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v50 = __riscv_vsrl_vv_u16mf2(v49, v48, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v51 = __riscv_vand_vx_u16mf2(v50, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v52 = __riscv_vsll_vx_u16mf2(v51, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v53 = __riscv_vncvt_x_x_w_u8mf4(v52, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v54 = __riscv_vor_vv_u8mf4(v38, v46, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v55 = __riscv_vor_vv_u8mf4(v39, v53, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v56 = __riscv_vreinterpret_v_u8mf4_i8mf4(v54);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v57 = __riscv_vsub_vx_i8mf4(v56, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v58 = __riscv_vreinterpret_v_u8mf4_i8mf4(v55);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v59 = __riscv_vsub_vx_i8mf4(v58, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v60 = __riscv_vwmul_vv_i16mf2(v57, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v61 = __riscv_vwmacc_vv_i16mf2(v60, v59, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v21;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16mf2_i32m1(v61, v63, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v10 + 1;
    size_t v67 = v66 * 22;
    const uint8_t* v68 = v3 + v67;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v69 = v10 + 1;
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v72 = (float)*(const _Float16 *)(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v74 = v68 + 2;
    uint32_t v75 = (uint16_t)*(const uint16_t *)(v74);
    const uint8_t* v76 = v68 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v78;
    v78 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v79 = __riscv_vsetvl_e32m1(16);
    for (size_t v80 = 0; v80 < 16; v80 += v79) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v81 = 16 - v80;
      size_t v82 = __riscv_vsetvl_e32m1(v81);
      const uint8_t* v83 = v68 + 6;
      const uint8_t* v84 = v83 + v80;
      const uint8_t* v85 = (const uint8_t*) v84;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v86 = __riscv_vle8_v_u8mf4(v85, v82);
      const uint8_t* v87 = v71 + 2;
      const uint8_t* v88 = v87 + v80;
      const int8_t* v89 = (const int8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v90 = __riscv_vle8_v_i8mf4(v89, v82);
      const uint8_t* v91 = v71 + 18;
      const uint8_t* v92 = v91 + v80;
      const int8_t* v93 = (const int8_t*) v92;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v94 = __riscv_vle8_v_i8mf4(v93, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v95 = __riscv_vand_vx_u8mf4(v86, 0x0F, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v96 = __riscv_vsrl_vx_u8mf4(v86, 0x04, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v97 = __riscv_vid_v_u16mf2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v98 = __riscv_vadd_vx_u16mf2(v97, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v99 = __riscv_vmv_v_x_u16mf2(v75, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v100 = __riscv_vsrl_vv_u16mf2(v99, v98, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v101 = __riscv_vand_vx_u16mf2(v100, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v102 = __riscv_vsll_vx_u16mf2(v101, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v103 = __riscv_vncvt_x_x_w_u8mf4(v102, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v104 = __riscv_vid_v_u16mf2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v105 = __riscv_vadd_vx_u16mf2(v104, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v106 = __riscv_vmv_v_x_u16mf2(v77, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v107 = __riscv_vsrl_vv_u16mf2(v106, v105, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v108 = __riscv_vand_vx_u16mf2(v107, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v109 = __riscv_vsll_vx_u16mf2(v108, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v110 = __riscv_vncvt_x_x_w_u8mf4(v109, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v111 = __riscv_vor_vv_u8mf4(v95, v103, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v112 = __riscv_vor_vv_u8mf4(v96, v110, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v113 = __riscv_vreinterpret_v_u8mf4_i8mf4(v111);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v114 = __riscv_vsub_vx_i8mf4(v113, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v115 = __riscv_vreinterpret_v_u8mf4_i8mf4(v112);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v116 = __riscv_vsub_vx_i8mf4(v115, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v117 = __riscv_vwmul_vv_i16mf2(v114, v90, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v118 = __riscv_vwmacc_vv_i16mf2(v117, v116, v94, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v119 = v78;
      vint32m1_t v120 = __riscv_vmv_v_x_i32m1(v119, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v121 = __riscv_vwredsum_vs_i16mf2_i32m1(v118, v120, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v122 = __riscv_vmv_x_s_i32m1_i32(v121);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v78 = v122;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v123 = v10 + 2;
    size_t v124 = v123 * 22;
    const uint8_t* v125 = v3 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v126 = v10 + 2;
    size_t v127 = v126 * 34;
    const uint8_t* v128 = v4 + v127;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v129 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v130 = (float)*(const _Float16 *)(v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v131 = v125 + 2;
    uint32_t v132 = (uint16_t)*(const uint16_t *)(v131);
    const uint8_t* v133 = v125 + 4;
    uint32_t v134 = (uint16_t)*(const uint16_t *)(v133);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v135;
    v135 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v136 = __riscv_vsetvl_e32m1(16);
    for (size_t v137 = 0; v137 < 16; v137 += v136) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v138 = 16 - v137;
      size_t v139 = __riscv_vsetvl_e32m1(v138);
      const uint8_t* v140 = v125 + 6;
      const uint8_t* v141 = v140 + v137;
      const uint8_t* v142 = (const uint8_t*) v141;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v143 = __riscv_vle8_v_u8mf4(v142, v139);
      const uint8_t* v144 = v128 + 2;
      const uint8_t* v145 = v144 + v137;
      const int8_t* v146 = (const int8_t*) v145;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v147 = __riscv_vle8_v_i8mf4(v146, v139);
      const uint8_t* v148 = v128 + 18;
      const uint8_t* v149 = v148 + v137;
      const int8_t* v150 = (const int8_t*) v149;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v151 = __riscv_vle8_v_i8mf4(v150, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v152 = __riscv_vand_vx_u8mf4(v143, 0x0F, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v153 = __riscv_vsrl_vx_u8mf4(v143, 0x04, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v154 = __riscv_vid_v_u16mf2(v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v155 = __riscv_vadd_vx_u16mf2(v154, v137, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v156 = __riscv_vmv_v_x_u16mf2(v132, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v157 = __riscv_vsrl_vv_u16mf2(v156, v155, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v158 = __riscv_vand_vx_u16mf2(v157, 0x1, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v159 = __riscv_vsll_vx_u16mf2(v158, 0x4, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v160 = __riscv_vncvt_x_x_w_u8mf4(v159, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v161 = __riscv_vid_v_u16mf2(v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v162 = __riscv_vadd_vx_u16mf2(v161, v137, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v163 = __riscv_vmv_v_x_u16mf2(v134, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v164 = __riscv_vsrl_vv_u16mf2(v163, v162, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v165 = __riscv_vand_vx_u16mf2(v164, 0x1, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v166 = __riscv_vsll_vx_u16mf2(v165, 0x4, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v167 = __riscv_vncvt_x_x_w_u8mf4(v166, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v168 = __riscv_vor_vv_u8mf4(v152, v160, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v169 = __riscv_vor_vv_u8mf4(v153, v167, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v170 = __riscv_vreinterpret_v_u8mf4_i8mf4(v168);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v171 = __riscv_vsub_vx_i8mf4(v170, 16, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v172 = __riscv_vreinterpret_v_u8mf4_i8mf4(v169);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v173 = __riscv_vsub_vx_i8mf4(v172, 16, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v174 = __riscv_vwmul_vv_i16mf2(v171, v147, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v175 = __riscv_vwmacc_vv_i16mf2(v174, v173, v151, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v176 = v135;
      vint32m1_t v177 = __riscv_vmv_v_x_i32m1(v176, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v178 = __riscv_vwredsum_vs_i16mf2_i32m1(v175, v177, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v179 = __riscv_vmv_x_s_i32m1_i32(v178);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v135 = v179;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v180 = v10 + 3;
    size_t v181 = v180 * 22;
    const uint8_t* v182 = v3 + v181;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v183 = v10 + 3;
    size_t v184 = v183 * 34;
    const uint8_t* v185 = v4 + v184;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v186 = (float)*(const _Float16 *)(v182);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v187 = (float)*(const _Float16 *)(v185);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v188 = v182 + 2;
    uint32_t v189 = (uint16_t)*(const uint16_t *)(v188);
    const uint8_t* v190 = v182 + 4;
    uint32_t v191 = (uint16_t)*(const uint16_t *)(v190);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v192;
    v192 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v193 = __riscv_vsetvl_e32m1(16);
    for (size_t v194 = 0; v194 < 16; v194 += v193) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v195 = 16 - v194;
      size_t v196 = __riscv_vsetvl_e32m1(v195);
      const uint8_t* v197 = v182 + 6;
      const uint8_t* v198 = v197 + v194;
      const uint8_t* v199 = (const uint8_t*) v198;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v200 = __riscv_vle8_v_u8mf4(v199, v196);
      const uint8_t* v201 = v185 + 2;
      const uint8_t* v202 = v201 + v194;
      const int8_t* v203 = (const int8_t*) v202;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v204 = __riscv_vle8_v_i8mf4(v203, v196);
      const uint8_t* v205 = v185 + 18;
      const uint8_t* v206 = v205 + v194;
      const int8_t* v207 = (const int8_t*) v206;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v208 = __riscv_vle8_v_i8mf4(v207, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v209 = __riscv_vand_vx_u8mf4(v200, 0x0F, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v210 = __riscv_vsrl_vx_u8mf4(v200, 0x04, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v211 = __riscv_vid_v_u16mf2(v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v212 = __riscv_vadd_vx_u16mf2(v211, v194, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v213 = __riscv_vmv_v_x_u16mf2(v189, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v214 = __riscv_vsrl_vv_u16mf2(v213, v212, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v215 = __riscv_vand_vx_u16mf2(v214, 0x1, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v216 = __riscv_vsll_vx_u16mf2(v215, 0x4, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v217 = __riscv_vncvt_x_x_w_u8mf4(v216, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v218 = __riscv_vid_v_u16mf2(v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v219 = __riscv_vadd_vx_u16mf2(v218, v194, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v220 = __riscv_vmv_v_x_u16mf2(v191, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v221 = __riscv_vsrl_vv_u16mf2(v220, v219, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v222 = __riscv_vand_vx_u16mf2(v221, 0x1, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v223 = __riscv_vsll_vx_u16mf2(v222, 0x4, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v224 = __riscv_vncvt_x_x_w_u8mf4(v223, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v225 = __riscv_vor_vv_u8mf4(v209, v217, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v226 = __riscv_vor_vv_u8mf4(v210, v224, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v227 = __riscv_vreinterpret_v_u8mf4_i8mf4(v225);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v228 = __riscv_vsub_vx_i8mf4(v227, 16, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v229 = __riscv_vreinterpret_v_u8mf4_i8mf4(v226);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v230 = __riscv_vsub_vx_i8mf4(v229, 16, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v231 = __riscv_vwmul_vv_i16mf2(v228, v204, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v232 = __riscv_vwmacc_vv_i16mf2(v231, v230, v208, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v233 = v192;
      vint32m1_t v234 = __riscv_vmv_v_x_i32m1(v233, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v235 = __riscv_vwredsum_vs_i16mf2_i32m1(v232, v234, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v236 = __riscv_vmv_x_s_i32m1_i32(v235);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v192 = v236;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v237 = v21;
    float v238 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v238 + (v15 * v16) * (float) v237;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v239 = v78;
    float v240 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v240 + (v72 * v73) * (float) v239;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v241 = v135;
    float v242 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v242 + (v129 * v130) * (float) v241;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v243 = v192;
    float v244 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v244 + (v186 * v187) * (float) v243;
  }
  for (size_t v245 = v9; v245 < v7; v245 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v246 = v245 * 22;
    const uint8_t* v247 = v3 + v246;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v248 = v245 * 34;
    const uint8_t* v249 = v4 + v248;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v250 = (float)*(const _Float16 *)(v247);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v251 = (float)*(const _Float16 *)(v249);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v252 = v247 + 2;
    uint32_t v253 = (uint16_t)*(const uint16_t *)(v252);
    const uint8_t* v254 = v247 + 4;
    uint32_t v255 = (uint16_t)*(const uint16_t *)(v254);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v256;
    v256 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v257 = __riscv_vsetvl_e32m1(16);
    for (size_t v258 = 0; v258 < 16; v258 += v257) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v259 = 16 - v258;
      size_t v260 = __riscv_vsetvl_e32m1(v259);
      const uint8_t* v261 = v247 + 6;
      const uint8_t* v262 = v261 + v258;
      const uint8_t* v263 = (const uint8_t*) v262;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v264 = __riscv_vle8_v_u8mf4(v263, v260);
      const uint8_t* v265 = v249 + 2;
      const uint8_t* v266 = v265 + v258;
      const int8_t* v267 = (const int8_t*) v266;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v268 = __riscv_vle8_v_i8mf4(v267, v260);
      const uint8_t* v269 = v249 + 18;
      const uint8_t* v270 = v269 + v258;
      const int8_t* v271 = (const int8_t*) v270;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v272 = __riscv_vle8_v_i8mf4(v271, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v273 = __riscv_vand_vx_u8mf4(v264, 0x0F, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v274 = __riscv_vsrl_vx_u8mf4(v264, 0x04, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v275 = __riscv_vid_v_u16mf2(v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v276 = __riscv_vadd_vx_u16mf2(v275, v258, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v277 = __riscv_vmv_v_x_u16mf2(v253, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v278 = __riscv_vsrl_vv_u16mf2(v277, v276, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v279 = __riscv_vand_vx_u16mf2(v278, 0x1, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v280 = __riscv_vsll_vx_u16mf2(v279, 0x4, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v281 = __riscv_vncvt_x_x_w_u8mf4(v280, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v282 = __riscv_vid_v_u16mf2(v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v283 = __riscv_vadd_vx_u16mf2(v282, v258, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v284 = __riscv_vmv_v_x_u16mf2(v255, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v285 = __riscv_vsrl_vv_u16mf2(v284, v283, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v286 = __riscv_vand_vx_u16mf2(v285, 0x1, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v287 = __riscv_vsll_vx_u16mf2(v286, 0x4, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v288 = __riscv_vncvt_x_x_w_u8mf4(v287, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v289 = __riscv_vor_vv_u8mf4(v273, v281, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v290 = __riscv_vor_vv_u8mf4(v274, v288, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v291 = __riscv_vreinterpret_v_u8mf4_i8mf4(v289);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v292 = __riscv_vsub_vx_i8mf4(v291, 16, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v293 = __riscv_vreinterpret_v_u8mf4_i8mf4(v290);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf4
      vint8mf4_t v294 = __riscv_vsub_vx_i8mf4(v293, 16, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v295 = __riscv_vwmul_vv_i16mf2(v292, v268, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v296 = __riscv_vwmacc_vv_i16mf2(v295, v294, v272, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v297 = v256;
      vint32m1_t v298 = __riscv_vmv_v_x_i32m1(v297, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v299 = __riscv_vwredsum_vs_i16mf2_i32m1(v296, v298, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v300 = __riscv_vmv_x_s_i32m1_i32(v299);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v256 = v300;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v301 = v256;
    float v302 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v302 + (v250 * v251) * (float) v301;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v303 = v6;
  v2[0] = v303;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_m1_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v20 = __riscv_vsetvl_e8m1(16);
    for (size_t v21 = 0; v21 < 16; v21 += v20) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v22 = 16 - v21;
      size_t v23 = __riscv_vsetvl_e8m1(v22);
      const uint8_t* v24 = v10 + 6;
      const uint8_t* v25 = v24 + v21;
      const uint8_t* v26 = (const uint8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v27 = __riscv_vle8_v_u8m1(v26, v23);
      const uint8_t* v28 = v12 + 2;
      const uint8_t* v29 = v28 + v21;
      const int8_t* v30 = (const int8_t*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v31 = __riscv_vle8_v_i8m1(v30, v23);
      const uint8_t* v32 = v12 + 18;
      const uint8_t* v33 = v32 + v21;
      const int8_t* v34 = (const int8_t*) v33;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v35 = __riscv_vle8_v_i8m1(v34, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v36 = __riscv_vand_vx_u8m1(v27, 0x0F, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v37 = __riscv_vsrl_vx_u8m1(v27, 0x04, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v38 = __riscv_vid_v_u16m2(v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v39 = __riscv_vadd_vx_u16m2(v38, v21, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v40 = __riscv_vmv_v_x_u16m2(v16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v41 = __riscv_vsrl_vv_u16m2(v40, v39, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v42 = __riscv_vand_vx_u16m2(v41, 0x1, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v43 = __riscv_vsll_vx_u16m2(v42, 0x4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v44 = __riscv_vncvt_x_x_w_u8m1(v43, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v45 = __riscv_vid_v_u16m2(v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v46 = __riscv_vadd_vx_u16m2(v45, v21, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v47 = __riscv_vmv_v_x_u16m2(v18, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v48 = __riscv_vsrl_vv_u16m2(v47, v46, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v49 = __riscv_vand_vx_u16m2(v48, 0x1, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v50 = __riscv_vsll_vx_u16m2(v49, 0x4, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v51 = __riscv_vncvt_x_x_w_u8m1(v50, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v52 = __riscv_vor_vv_u8m1(v36, v44, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v53 = __riscv_vor_vv_u8m1(v37, v51, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v54 = __riscv_vreinterpret_v_u8m1_i8m1(v52);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v55 = __riscv_vsub_vx_i8m1(v54, 16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v56 = __riscv_vreinterpret_v_u8m1_i8m1(v53);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v57 = __riscv_vsub_vx_i8m1(v56, 16, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v58 = __riscv_vwmul_vv_i16m2(v55, v31, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v59 = __riscv_vwmacc_vv_i16m2(v58, v57, v35, v23);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v60 = v19;
      vint32m1_t v61 = __riscv_vmv_v_x_i32m1(v60, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v62 = __riscv_vwredsum_vs_i16m2_i32m1(v59, v61, v23);
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



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_m1_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 22;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v17 = v12 + 2;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    const uint8_t* v19 = v12 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v12 + 6;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, v25);
      const uint8_t* v30 = v14 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      const uint8_t* v34 = v14 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v38 = __riscv_vand_vx_u8m1(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v39 = __riscv_vsrl_vx_u8m1(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v40 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v41 = __riscv_vadd_vx_u16m2(v40, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v42 = __riscv_vmv_v_x_u16m2(v18, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v43 = __riscv_vsrl_vv_u16m2(v42, v41, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v44 = __riscv_vand_vx_u16m2(v43, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v45 = __riscv_vsll_vx_u16m2(v44, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v46 = __riscv_vncvt_x_x_w_u8m1(v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v47 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v48 = __riscv_vadd_vx_u16m2(v47, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v49 = __riscv_vmv_v_x_u16m2(v20, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v50 = __riscv_vsrl_vv_u16m2(v49, v48, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v51 = __riscv_vand_vx_u16m2(v50, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v52 = __riscv_vsll_vx_u16m2(v51, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v53 = __riscv_vncvt_x_x_w_u8m1(v52, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v54 = __riscv_vor_vv_u8m1(v38, v46, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v53, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v56 = __riscv_vreinterpret_v_u8m1_i8m1(v54);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v57 = __riscv_vsub_vx_i8m1(v56, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v59 = __riscv_vsub_vx_i8m1(v58, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v57, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v61 = __riscv_vwmacc_vv_i16m2(v60, v59, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v21;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16m2_i32m1(v61, v63, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v10 + 1;
    size_t v67 = v66 * 22;
    const uint8_t* v68 = v3 + v67;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v69 = v10 + 1;
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v72 = (float)*(const _Float16 *)(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v74 = v68 + 2;
    uint32_t v75 = (uint16_t)*(const uint16_t *)(v74);
    const uint8_t* v76 = v68 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v78;
    v78 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v79 = __riscv_vsetvl_e8m1(16);
    for (size_t v80 = 0; v80 < 16; v80 += v79) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v81 = 16 - v80;
      size_t v82 = __riscv_vsetvl_e8m1(v81);
      const uint8_t* v83 = v68 + 6;
      const uint8_t* v84 = v83 + v80;
      const uint8_t* v85 = (const uint8_t*) v84;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v86 = __riscv_vle8_v_u8m1(v85, v82);
      const uint8_t* v87 = v71 + 2;
      const uint8_t* v88 = v87 + v80;
      const int8_t* v89 = (const int8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v90 = __riscv_vle8_v_i8m1(v89, v82);
      const uint8_t* v91 = v71 + 18;
      const uint8_t* v92 = v91 + v80;
      const int8_t* v93 = (const int8_t*) v92;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v94 = __riscv_vle8_v_i8m1(v93, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v95 = __riscv_vand_vx_u8m1(v86, 0x0F, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v96 = __riscv_vsrl_vx_u8m1(v86, 0x04, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v97 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v98 = __riscv_vadd_vx_u16m2(v97, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v99 = __riscv_vmv_v_x_u16m2(v75, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v100 = __riscv_vsrl_vv_u16m2(v99, v98, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v101 = __riscv_vand_vx_u16m2(v100, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v102 = __riscv_vsll_vx_u16m2(v101, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v103 = __riscv_vncvt_x_x_w_u8m1(v102, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v104 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v105 = __riscv_vadd_vx_u16m2(v104, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v106 = __riscv_vmv_v_x_u16m2(v77, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v107 = __riscv_vsrl_vv_u16m2(v106, v105, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v108 = __riscv_vand_vx_u16m2(v107, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v109 = __riscv_vsll_vx_u16m2(v108, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v110 = __riscv_vncvt_x_x_w_u8m1(v109, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v103, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v112 = __riscv_vor_vv_u8m1(v96, v110, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v114 = __riscv_vsub_vx_i8m1(v113, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v115 = __riscv_vreinterpret_v_u8m1_i8m1(v112);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v116 = __riscv_vsub_vx_i8m1(v115, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v117 = __riscv_vwmul_vv_i16m2(v114, v90, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v118 = __riscv_vwmacc_vv_i16m2(v117, v116, v94, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v119 = v78;
      vint32m1_t v120 = __riscv_vmv_v_x_i32m1(v119, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v121 = __riscv_vwredsum_vs_i16m2_i32m1(v118, v120, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v122 = __riscv_vmv_x_s_i32m1_i32(v121);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v78 = v122;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v123 = v21;
    float v124 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v124 + (v15 * v16) * (float) v123;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v125 = v78;
    float v126 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v126 + (v72 * v73) * (float) v125;
  }
  for (size_t v127 = v9; v127 < v7; v127 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v128 = v127 * 22;
    const uint8_t* v129 = v3 + v128;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v130 = v127 * 34;
    const uint8_t* v131 = v4 + v130;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v132 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v131);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v134 = v129 + 2;
    uint32_t v135 = (uint16_t)*(const uint16_t *)(v134);
    const uint8_t* v136 = v129 + 4;
    uint32_t v137 = (uint16_t)*(const uint16_t *)(v136);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v138;
    v138 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v139 = __riscv_vsetvl_e8m1(16);
    for (size_t v140 = 0; v140 < 16; v140 += v139) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v141 = 16 - v140;
      size_t v142 = __riscv_vsetvl_e8m1(v141);
      const uint8_t* v143 = v129 + 6;
      const uint8_t* v144 = v143 + v140;
      const uint8_t* v145 = (const uint8_t*) v144;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v146 = __riscv_vle8_v_u8m1(v145, v142);
      const uint8_t* v147 = v131 + 2;
      const uint8_t* v148 = v147 + v140;
      const int8_t* v149 = (const int8_t*) v148;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v150 = __riscv_vle8_v_i8m1(v149, v142);
      const uint8_t* v151 = v131 + 18;
      const uint8_t* v152 = v151 + v140;
      const int8_t* v153 = (const int8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v154 = __riscv_vle8_v_i8m1(v153, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v155 = __riscv_vand_vx_u8m1(v146, 0x0F, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v156 = __riscv_vsrl_vx_u8m1(v146, 0x04, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v157 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v158 = __riscv_vadd_vx_u16m2(v157, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v159 = __riscv_vmv_v_x_u16m2(v135, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v160 = __riscv_vsrl_vv_u16m2(v159, v158, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v161 = __riscv_vand_vx_u16m2(v160, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v162 = __riscv_vsll_vx_u16m2(v161, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v163 = __riscv_vncvt_x_x_w_u8m1(v162, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v164 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v165 = __riscv_vadd_vx_u16m2(v164, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v166 = __riscv_vmv_v_x_u16m2(v137, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v167 = __riscv_vsrl_vv_u16m2(v166, v165, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v168 = __riscv_vand_vx_u16m2(v167, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v169 = __riscv_vsll_vx_u16m2(v168, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v170 = __riscv_vncvt_x_x_w_u8m1(v169, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v171 = __riscv_vor_vv_u8m1(v155, v163, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v172 = __riscv_vor_vv_u8m1(v156, v170, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v173 = __riscv_vreinterpret_v_u8m1_i8m1(v171);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v174 = __riscv_vsub_vx_i8m1(v173, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v175 = __riscv_vreinterpret_v_u8m1_i8m1(v172);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v176 = __riscv_vsub_vx_i8m1(v175, 16, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v177 = __riscv_vwmul_vv_i16m2(v174, v150, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v178 = __riscv_vwmacc_vv_i16m2(v177, v176, v154, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v179 = v138;
      vint32m1_t v180 = __riscv_vmv_v_x_i32m1(v179, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v181 = __riscv_vwredsum_vs_i16m2_i32m1(v178, v180, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v182 = __riscv_vmv_x_s_i32m1_i32(v181);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v138 = v182;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v183 = v138;
    float v184 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v184 + (v132 * v133) * (float) v183;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v185 = v6;
  v2[0] = v185;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0_m1_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 22;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v17 = v12 + 2;
    uint32_t v18 = (uint16_t)*(const uint16_t *)(v17);
    const uint8_t* v19 = v12 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(16);
    for (size_t v23 = 0; v23 < 16; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 16 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v12 + 6;
      const uint8_t* v27 = v26 + v23;
      const uint8_t* v28 = (const uint8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, v25);
      const uint8_t* v30 = v14 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      const uint8_t* v34 = v14 + 18;
      const uint8_t* v35 = v34 + v23;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v38 = __riscv_vand_vx_u8m1(v29, 0x0F, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v39 = __riscv_vsrl_vx_u8m1(v29, 0x04, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v40 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v41 = __riscv_vadd_vx_u16m2(v40, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v42 = __riscv_vmv_v_x_u16m2(v18, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v43 = __riscv_vsrl_vv_u16m2(v42, v41, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v44 = __riscv_vand_vx_u16m2(v43, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v45 = __riscv_vsll_vx_u16m2(v44, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v46 = __riscv_vncvt_x_x_w_u8m1(v45, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v47 = __riscv_vid_v_u16m2(v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v48 = __riscv_vadd_vx_u16m2(v47, v23, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v49 = __riscv_vmv_v_x_u16m2(v20, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v50 = __riscv_vsrl_vv_u16m2(v49, v48, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v51 = __riscv_vand_vx_u16m2(v50, 0x1, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v52 = __riscv_vsll_vx_u16m2(v51, 0x4, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v53 = __riscv_vncvt_x_x_w_u8m1(v52, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v54 = __riscv_vor_vv_u8m1(v38, v46, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v53, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v56 = __riscv_vreinterpret_v_u8m1_i8m1(v54);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v57 = __riscv_vsub_vx_i8m1(v56, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v59 = __riscv_vsub_vx_i8m1(v58, 16, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v57, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v61 = __riscv_vwmacc_vv_i16m2(v60, v59, v37, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v21;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16m2_i32m1(v61, v63, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v10 + 1;
    size_t v67 = v66 * 22;
    const uint8_t* v68 = v3 + v67;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v69 = v10 + 1;
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v72 = (float)*(const _Float16 *)(v68);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v74 = v68 + 2;
    uint32_t v75 = (uint16_t)*(const uint16_t *)(v74);
    const uint8_t* v76 = v68 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v78;
    v78 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v79 = __riscv_vsetvl_e8m1(16);
    for (size_t v80 = 0; v80 < 16; v80 += v79) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v81 = 16 - v80;
      size_t v82 = __riscv_vsetvl_e8m1(v81);
      const uint8_t* v83 = v68 + 6;
      const uint8_t* v84 = v83 + v80;
      const uint8_t* v85 = (const uint8_t*) v84;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v86 = __riscv_vle8_v_u8m1(v85, v82);
      const uint8_t* v87 = v71 + 2;
      const uint8_t* v88 = v87 + v80;
      const int8_t* v89 = (const int8_t*) v88;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v90 = __riscv_vle8_v_i8m1(v89, v82);
      const uint8_t* v91 = v71 + 18;
      const uint8_t* v92 = v91 + v80;
      const int8_t* v93 = (const int8_t*) v92;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v94 = __riscv_vle8_v_i8m1(v93, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v95 = __riscv_vand_vx_u8m1(v86, 0x0F, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v96 = __riscv_vsrl_vx_u8m1(v86, 0x04, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v97 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v98 = __riscv_vadd_vx_u16m2(v97, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v99 = __riscv_vmv_v_x_u16m2(v75, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v100 = __riscv_vsrl_vv_u16m2(v99, v98, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v101 = __riscv_vand_vx_u16m2(v100, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v102 = __riscv_vsll_vx_u16m2(v101, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v103 = __riscv_vncvt_x_x_w_u8m1(v102, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v104 = __riscv_vid_v_u16m2(v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v105 = __riscv_vadd_vx_u16m2(v104, v80, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v106 = __riscv_vmv_v_x_u16m2(v77, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v107 = __riscv_vsrl_vv_u16m2(v106, v105, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v108 = __riscv_vand_vx_u16m2(v107, 0x1, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v109 = __riscv_vsll_vx_u16m2(v108, 0x4, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v110 = __riscv_vncvt_x_x_w_u8m1(v109, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v103, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v112 = __riscv_vor_vv_u8m1(v96, v110, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v114 = __riscv_vsub_vx_i8m1(v113, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v115 = __riscv_vreinterpret_v_u8m1_i8m1(v112);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v116 = __riscv_vsub_vx_i8m1(v115, 16, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v117 = __riscv_vwmul_vv_i16m2(v114, v90, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v118 = __riscv_vwmacc_vv_i16m2(v117, v116, v94, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v119 = v78;
      vint32m1_t v120 = __riscv_vmv_v_x_i32m1(v119, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v121 = __riscv_vwredsum_vs_i16m2_i32m1(v118, v120, v82);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v122 = __riscv_vmv_x_s_i32m1_i32(v121);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v78 = v122;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v123 = v10 + 2;
    size_t v124 = v123 * 22;
    const uint8_t* v125 = v3 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v126 = v10 + 2;
    size_t v127 = v126 * 34;
    const uint8_t* v128 = v4 + v127;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v129 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v130 = (float)*(const _Float16 *)(v128);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v131 = v125 + 2;
    uint32_t v132 = (uint16_t)*(const uint16_t *)(v131);
    const uint8_t* v133 = v125 + 4;
    uint32_t v134 = (uint16_t)*(const uint16_t *)(v133);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v135;
    v135 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v136 = __riscv_vsetvl_e8m1(16);
    for (size_t v137 = 0; v137 < 16; v137 += v136) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v138 = 16 - v137;
      size_t v139 = __riscv_vsetvl_e8m1(v138);
      const uint8_t* v140 = v125 + 6;
      const uint8_t* v141 = v140 + v137;
      const uint8_t* v142 = (const uint8_t*) v141;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v143 = __riscv_vle8_v_u8m1(v142, v139);
      const uint8_t* v144 = v128 + 2;
      const uint8_t* v145 = v144 + v137;
      const int8_t* v146 = (const int8_t*) v145;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v147 = __riscv_vle8_v_i8m1(v146, v139);
      const uint8_t* v148 = v128 + 18;
      const uint8_t* v149 = v148 + v137;
      const int8_t* v150 = (const int8_t*) v149;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v151 = __riscv_vle8_v_i8m1(v150, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v152 = __riscv_vand_vx_u8m1(v143, 0x0F, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v153 = __riscv_vsrl_vx_u8m1(v143, 0x04, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v154 = __riscv_vid_v_u16m2(v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v155 = __riscv_vadd_vx_u16m2(v154, v137, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v156 = __riscv_vmv_v_x_u16m2(v132, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v157 = __riscv_vsrl_vv_u16m2(v156, v155, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v158 = __riscv_vand_vx_u16m2(v157, 0x1, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v159 = __riscv_vsll_vx_u16m2(v158, 0x4, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v160 = __riscv_vncvt_x_x_w_u8m1(v159, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v161 = __riscv_vid_v_u16m2(v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v162 = __riscv_vadd_vx_u16m2(v161, v137, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v163 = __riscv_vmv_v_x_u16m2(v134, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v164 = __riscv_vsrl_vv_u16m2(v163, v162, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v165 = __riscv_vand_vx_u16m2(v164, 0x1, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v166 = __riscv_vsll_vx_u16m2(v165, 0x4, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v167 = __riscv_vncvt_x_x_w_u8m1(v166, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v168 = __riscv_vor_vv_u8m1(v152, v160, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v169 = __riscv_vor_vv_u8m1(v153, v167, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v170 = __riscv_vreinterpret_v_u8m1_i8m1(v168);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v171 = __riscv_vsub_vx_i8m1(v170, 16, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v172 = __riscv_vreinterpret_v_u8m1_i8m1(v169);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v173 = __riscv_vsub_vx_i8m1(v172, 16, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v174 = __riscv_vwmul_vv_i16m2(v171, v147, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v175 = __riscv_vwmacc_vv_i16m2(v174, v173, v151, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v176 = v135;
      vint32m1_t v177 = __riscv_vmv_v_x_i32m1(v176, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v178 = __riscv_vwredsum_vs_i16m2_i32m1(v175, v177, v139);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v179 = __riscv_vmv_x_s_i32m1_i32(v178);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v135 = v179;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v180 = v10 + 3;
    size_t v181 = v180 * 22;
    const uint8_t* v182 = v3 + v181;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v183 = v10 + 3;
    size_t v184 = v183 * 34;
    const uint8_t* v185 = v4 + v184;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v186 = (float)*(const _Float16 *)(v182);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v187 = (float)*(const _Float16 *)(v185);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v188 = v182 + 2;
    uint32_t v189 = (uint16_t)*(const uint16_t *)(v188);
    const uint8_t* v190 = v182 + 4;
    uint32_t v191 = (uint16_t)*(const uint16_t *)(v190);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v192;
    v192 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v193 = __riscv_vsetvl_e8m1(16);
    for (size_t v194 = 0; v194 < 16; v194 += v193) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v195 = 16 - v194;
      size_t v196 = __riscv_vsetvl_e8m1(v195);
      const uint8_t* v197 = v182 + 6;
      const uint8_t* v198 = v197 + v194;
      const uint8_t* v199 = (const uint8_t*) v198;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v200 = __riscv_vle8_v_u8m1(v199, v196);
      const uint8_t* v201 = v185 + 2;
      const uint8_t* v202 = v201 + v194;
      const int8_t* v203 = (const int8_t*) v202;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v204 = __riscv_vle8_v_i8m1(v203, v196);
      const uint8_t* v205 = v185 + 18;
      const uint8_t* v206 = v205 + v194;
      const int8_t* v207 = (const int8_t*) v206;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v208 = __riscv_vle8_v_i8m1(v207, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v209 = __riscv_vand_vx_u8m1(v200, 0x0F, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v210 = __riscv_vsrl_vx_u8m1(v200, 0x04, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v211 = __riscv_vid_v_u16m2(v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v212 = __riscv_vadd_vx_u16m2(v211, v194, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v213 = __riscv_vmv_v_x_u16m2(v189, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v214 = __riscv_vsrl_vv_u16m2(v213, v212, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v215 = __riscv_vand_vx_u16m2(v214, 0x1, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v216 = __riscv_vsll_vx_u16m2(v215, 0x4, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v217 = __riscv_vncvt_x_x_w_u8m1(v216, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v218 = __riscv_vid_v_u16m2(v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v219 = __riscv_vadd_vx_u16m2(v218, v194, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v220 = __riscv_vmv_v_x_u16m2(v191, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v221 = __riscv_vsrl_vv_u16m2(v220, v219, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v222 = __riscv_vand_vx_u16m2(v221, 0x1, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v223 = __riscv_vsll_vx_u16m2(v222, 0x4, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v224 = __riscv_vncvt_x_x_w_u8m1(v223, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v225 = __riscv_vor_vv_u8m1(v209, v217, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v226 = __riscv_vor_vv_u8m1(v210, v224, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v227 = __riscv_vreinterpret_v_u8m1_i8m1(v225);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v228 = __riscv_vsub_vx_i8m1(v227, 16, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v229 = __riscv_vreinterpret_v_u8m1_i8m1(v226);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v230 = __riscv_vsub_vx_i8m1(v229, 16, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v231 = __riscv_vwmul_vv_i16m2(v228, v204, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v232 = __riscv_vwmacc_vv_i16m2(v231, v230, v208, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v233 = v192;
      vint32m1_t v234 = __riscv_vmv_v_x_i32m1(v233, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v235 = __riscv_vwredsum_vs_i16m2_i32m1(v232, v234, v196);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v236 = __riscv_vmv_x_s_i32m1_i32(v235);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v192 = v236;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v237 = v21;
    float v238 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v238 + (v15 * v16) * (float) v237;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v239 = v78;
    float v240 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v240 + (v72 * v73) * (float) v239;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v241 = v135;
    float v242 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v242 + (v129 * v130) * (float) v241;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v243 = v192;
    float v244 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v244 + (v186 * v187) * (float) v243;
  }
  for (size_t v245 = v9; v245 < v7; v245 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v246 = v245 * 22;
    const uint8_t* v247 = v3 + v246;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v248 = v245 * 34;
    const uint8_t* v249 = v4 + v248;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v250 = (float)*(const _Float16 *)(v247);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v251 = (float)*(const _Float16 *)(v249);
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v252 = v247 + 2;
    uint32_t v253 = (uint16_t)*(const uint16_t *)(v252);
    const uint8_t* v254 = v247 + 4;
    uint32_t v255 = (uint16_t)*(const uint16_t *)(v254);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v256;
    v256 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v257 = __riscv_vsetvl_e8m1(16);
    for (size_t v258 = 0; v258 < 16; v258 += v257) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v259 = 16 - v258;
      size_t v260 = __riscv_vsetvl_e8m1(v259);
      const uint8_t* v261 = v247 + 6;
      const uint8_t* v262 = v261 + v258;
      const uint8_t* v263 = (const uint8_t*) v262;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v264 = __riscv_vle8_v_u8m1(v263, v260);
      const uint8_t* v265 = v249 + 2;
      const uint8_t* v266 = v265 + v258;
      const int8_t* v267 = (const int8_t*) v266;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v268 = __riscv_vle8_v_i8m1(v267, v260);
      const uint8_t* v269 = v249 + 18;
      const uint8_t* v270 = v269 + v258;
      const int8_t* v271 = (const int8_t*) v270;
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v272 = __riscv_vle8_v_i8m1(v271, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v273 = __riscv_vand_vx_u8m1(v264, 0x0F, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v274 = __riscv_vsrl_vx_u8m1(v264, 0x04, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v275 = __riscv_vid_v_u16m2(v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v276 = __riscv_vadd_vx_u16m2(v275, v258, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v277 = __riscv_vmv_v_x_u16m2(v253, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v278 = __riscv_vsrl_vv_u16m2(v277, v276, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v279 = __riscv_vand_vx_u16m2(v278, 0x1, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v280 = __riscv_vsll_vx_u16m2(v279, 0x4, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v281 = __riscv_vncvt_x_x_w_u8m1(v280, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v282 = __riscv_vid_v_u16m2(v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v283 = __riscv_vadd_vx_u16m2(v282, v258, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v284 = __riscv_vmv_v_x_u16m2(v255, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v285 = __riscv_vsrl_vv_u16m2(v284, v283, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v286 = __riscv_vand_vx_u16m2(v285, 0x1, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v287 = __riscv_vsll_vx_u16m2(v286, 0x4, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v288 = __riscv_vncvt_x_x_w_u8m1(v287, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v289 = __riscv_vor_vv_u8m1(v273, v281, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v290 = __riscv_vor_vv_u8m1(v274, v288, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v291 = __riscv_vreinterpret_v_u8m1_i8m1(v289);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v292 = __riscv_vsub_vx_i8m1(v291, 16, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v293 = __riscv_vreinterpret_v_u8m1_i8m1(v290);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
      vint8m1_t v294 = __riscv_vsub_vx_i8m1(v293, 16, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v295 = __riscv_vwmul_vv_i16m2(v292, v268, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v296 = __riscv_vwmacc_vv_i16m2(v295, v294, v272, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v297 = v256;
      vint32m1_t v298 = __riscv_vmv_v_x_i32m1(v297, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v299 = __riscv_vwredsum_vs_i16m2_i32m1(v296, v298, v260);
      // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v300 = __riscv_vmv_x_s_i32m1_i32(v299);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v256 = v300;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v301 = v256;
    float v302 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v302 + (v250 * v251) * (float) v301;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v303 = v6;
  v2[0] = v303;
  return;
}


