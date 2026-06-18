#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_mf4_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 24;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 36;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    const uint8_t* v15 = v10 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v15);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v19 = v10 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    const uint8_t* v21 = v10 + 6;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v23;
    v23 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v24 = __riscv_vsetvl_e32m1(16);
    for (size_t v25 = 0; v25 < 16; v25 += v24) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v26 = 16 - v25;
      size_t v27 = __riscv_vsetvl_e32m1(v26);
      const uint8_t* v28 = v10 + 8;
      const uint8_t* v29 = v28 + v25;
      const uint8_t* v30 = (const uint8_t*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v31 = __riscv_vle8_v_u8mf4(v30, v27);
      const uint8_t* v32 = v12 + 4;
      const uint8_t* v33 = v32 + v25;
      const int8_t* v34 = (const int8_t*) v33;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v35 = __riscv_vle8_v_i8mf4(v34, v27);
      const uint8_t* v36 = v12 + 20;
      const uint8_t* v37 = v36 + v25;
      const int8_t* v38 = (const int8_t*) v37;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v39 = __riscv_vle8_v_i8mf4(v38, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v40 = __riscv_vand_vx_u8mf4(v31, 0x0F, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v41 = __riscv_vsrl_vx_u8mf4(v31, 0x04, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v42 = __riscv_vid_v_u16mf2(v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v43 = __riscv_vadd_vx_u16mf2(v42, v25, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v44 = __riscv_vmv_v_x_u16mf2(v20, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v45 = __riscv_vsrl_vv_u16mf2(v44, v43, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v46 = __riscv_vand_vx_u16mf2(v45, 0x1, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v47 = __riscv_vsll_vx_u16mf2(v46, 0x4, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v48 = __riscv_vncvt_x_x_w_u8mf4(v47, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v49 = __riscv_vid_v_u16mf2(v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v50 = __riscv_vadd_vx_u16mf2(v49, v25, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v51 = __riscv_vmv_v_x_u16mf2(v22, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v52 = __riscv_vsrl_vv_u16mf2(v51, v50, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v53 = __riscv_vand_vx_u16mf2(v52, 0x1, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v54 = __riscv_vsll_vx_u16mf2(v53, 0x4, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v55 = __riscv_vncvt_x_x_w_u8mf4(v54, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v56 = __riscv_vor_vv_u8mf4(v40, v48, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v57 = __riscv_vor_vv_u8mf4(v41, v55, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v58 = __riscv_vreinterpret_v_u8mf4_i8mf4(v56);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v59 = __riscv_vreinterpret_v_u8mf4_i8mf4(v57);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v60 = __riscv_vwmul_vv_i16mf2(v58, v35, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v61 = __riscv_vwmacc_vv_i16mf2(v60, v59, v39, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v23;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16mf2_i32m1(v61, v63, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v23 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v66 = v23;
    float v67 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v67 + ((v13 * v14) * (float) v66 + v16 * v18);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v68 = v6;
  v2[0] = v68;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_mf4_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 24;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v21 = v12 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    const uint8_t* v23 = v12 + 6;
    uint32_t v24 = (uint16_t)*(const uint16_t *)(v23);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v26 = __riscv_vsetvl_e32m1(16);
    for (size_t v27 = 0; v27 < 16; v27 += v26) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v28 = 16 - v27;
      size_t v29 = __riscv_vsetvl_e32m1(v28);
      const uint8_t* v30 = v12 + 8;
      const uint8_t* v31 = v30 + v27;
      const uint8_t* v32 = (const uint8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v33 = __riscv_vle8_v_u8mf4(v32, v29);
      const uint8_t* v34 = v14 + 4;
      const uint8_t* v35 = v34 + v27;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v29);
      const uint8_t* v38 = v14 + 20;
      const uint8_t* v39 = v38 + v27;
      const int8_t* v40 = (const int8_t*) v39;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v42 = __riscv_vand_vx_u8mf4(v33, 0x0F, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v43 = __riscv_vsrl_vx_u8mf4(v33, 0x04, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v44 = __riscv_vid_v_u16mf2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v45 = __riscv_vadd_vx_u16mf2(v44, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v46 = __riscv_vmv_v_x_u16mf2(v22, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v47 = __riscv_vsrl_vv_u16mf2(v46, v45, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v48 = __riscv_vand_vx_u16mf2(v47, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v49 = __riscv_vsll_vx_u16mf2(v48, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v50 = __riscv_vncvt_x_x_w_u8mf4(v49, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v51 = __riscv_vid_v_u16mf2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v52 = __riscv_vadd_vx_u16mf2(v51, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v53 = __riscv_vmv_v_x_u16mf2(v24, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v54 = __riscv_vsrl_vv_u16mf2(v53, v52, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v55 = __riscv_vand_vx_u16mf2(v54, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v56 = __riscv_vsll_vx_u16mf2(v55, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v57 = __riscv_vncvt_x_x_w_u8mf4(v56, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v58 = __riscv_vor_vv_u8mf4(v42, v50, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v59 = __riscv_vor_vv_u8mf4(v43, v57, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v60 = __riscv_vreinterpret_v_u8mf4_i8mf4(v58);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v61 = __riscv_vreinterpret_v_u8mf4_i8mf4(v59);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v62 = __riscv_vwmul_vv_i16mf2(v60, v37, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v63 = __riscv_vwmacc_vv_i16mf2(v62, v61, v41, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v64 = v25;
      vint32m1_t v65 = __riscv_vmv_v_x_i32m1(v64, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v66 = __riscv_vwredsum_vs_i16mf2_i32m1(v63, v65, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v67 = __riscv_vmv_x_s_i32m1_i32(v66);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v25 = v67;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v68 = v10 + 1;
    size_t v69 = v68 * 24;
    const uint8_t* v70 = v3 + v69;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v71 = v10 + 1;
    size_t v72 = v71 * 36;
    const uint8_t* v73 = v4 + v72;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v74 = (float)*(const _Float16 *)(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v73);
    const uint8_t* v76 = v70 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v77 = (float)*(const _Float16 *)(v76);
    const uint8_t* v78 = v73 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v79 = (float)*(const _Float16 *)(v78);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v80 = v70 + 4;
    uint32_t v81 = (uint16_t)*(const uint16_t *)(v80);
    const uint8_t* v82 = v70 + 6;
    uint32_t v83 = (uint16_t)*(const uint16_t *)(v82);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v84;
    v84 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v85 = __riscv_vsetvl_e32m1(16);
    for (size_t v86 = 0; v86 < 16; v86 += v85) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v87 = 16 - v86;
      size_t v88 = __riscv_vsetvl_e32m1(v87);
      const uint8_t* v89 = v70 + 8;
      const uint8_t* v90 = v89 + v86;
      const uint8_t* v91 = (const uint8_t*) v90;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v92 = __riscv_vle8_v_u8mf4(v91, v88);
      const uint8_t* v93 = v73 + 4;
      const uint8_t* v94 = v93 + v86;
      const int8_t* v95 = (const int8_t*) v94;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v96 = __riscv_vle8_v_i8mf4(v95, v88);
      const uint8_t* v97 = v73 + 20;
      const uint8_t* v98 = v97 + v86;
      const int8_t* v99 = (const int8_t*) v98;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v100 = __riscv_vle8_v_i8mf4(v99, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v101 = __riscv_vand_vx_u8mf4(v92, 0x0F, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v102 = __riscv_vsrl_vx_u8mf4(v92, 0x04, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v103 = __riscv_vid_v_u16mf2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v104 = __riscv_vadd_vx_u16mf2(v103, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v105 = __riscv_vmv_v_x_u16mf2(v81, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v106 = __riscv_vsrl_vv_u16mf2(v105, v104, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v107 = __riscv_vand_vx_u16mf2(v106, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v108 = __riscv_vsll_vx_u16mf2(v107, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v109 = __riscv_vncvt_x_x_w_u8mf4(v108, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v110 = __riscv_vid_v_u16mf2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v111 = __riscv_vadd_vx_u16mf2(v110, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v112 = __riscv_vmv_v_x_u16mf2(v83, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v113 = __riscv_vsrl_vv_u16mf2(v112, v111, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v114 = __riscv_vand_vx_u16mf2(v113, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v115 = __riscv_vsll_vx_u16mf2(v114, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v116 = __riscv_vncvt_x_x_w_u8mf4(v115, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v117 = __riscv_vor_vv_u8mf4(v101, v109, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v118 = __riscv_vor_vv_u8mf4(v102, v116, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v119 = __riscv_vreinterpret_v_u8mf4_i8mf4(v117);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v120 = __riscv_vreinterpret_v_u8mf4_i8mf4(v118);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v121 = __riscv_vwmul_vv_i16mf2(v119, v96, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v122 = __riscv_vwmacc_vv_i16mf2(v121, v120, v100, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v123 = v84;
      vint32m1_t v124 = __riscv_vmv_v_x_i32m1(v123, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v125 = __riscv_vwredsum_vs_i16mf2_i32m1(v122, v124, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v126 = __riscv_vmv_x_s_i32m1_i32(v125);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v84 = v126;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v127 = v25;
    float v128 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v128 + ((v15 * v16) * (float) v127 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v129 = v84;
    float v130 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v130 + ((v74 * v75) * (float) v129 + v77 * v79);
  }
  for (size_t v131 = v9; v131 < v7; v131 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v132 = v131 * 24;
    const uint8_t* v133 = v3 + v132;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v134 = v131 * 36;
    const uint8_t* v135 = v4 + v134;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v136 = (float)*(const _Float16 *)(v133);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v137 = (float)*(const _Float16 *)(v135);
    const uint8_t* v138 = v133 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v139 = (float)*(const _Float16 *)(v138);
    const uint8_t* v140 = v135 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v141 = (float)*(const _Float16 *)(v140);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v142 = v133 + 4;
    uint32_t v143 = (uint16_t)*(const uint16_t *)(v142);
    const uint8_t* v144 = v133 + 6;
    uint32_t v145 = (uint16_t)*(const uint16_t *)(v144);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v146;
    v146 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v147 = __riscv_vsetvl_e32m1(16);
    for (size_t v148 = 0; v148 < 16; v148 += v147) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v149 = 16 - v148;
      size_t v150 = __riscv_vsetvl_e32m1(v149);
      const uint8_t* v151 = v133 + 8;
      const uint8_t* v152 = v151 + v148;
      const uint8_t* v153 = (const uint8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v154 = __riscv_vle8_v_u8mf4(v153, v150);
      const uint8_t* v155 = v135 + 4;
      const uint8_t* v156 = v155 + v148;
      const int8_t* v157 = (const int8_t*) v156;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v158 = __riscv_vle8_v_i8mf4(v157, v150);
      const uint8_t* v159 = v135 + 20;
      const uint8_t* v160 = v159 + v148;
      const int8_t* v161 = (const int8_t*) v160;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v162 = __riscv_vle8_v_i8mf4(v161, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v163 = __riscv_vand_vx_u8mf4(v154, 0x0F, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v164 = __riscv_vsrl_vx_u8mf4(v154, 0x04, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v165 = __riscv_vid_v_u16mf2(v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v166 = __riscv_vadd_vx_u16mf2(v165, v148, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v167 = __riscv_vmv_v_x_u16mf2(v143, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v168 = __riscv_vsrl_vv_u16mf2(v167, v166, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v169 = __riscv_vand_vx_u16mf2(v168, 0x1, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v170 = __riscv_vsll_vx_u16mf2(v169, 0x4, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v171 = __riscv_vncvt_x_x_w_u8mf4(v170, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v172 = __riscv_vid_v_u16mf2(v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v173 = __riscv_vadd_vx_u16mf2(v172, v148, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v174 = __riscv_vmv_v_x_u16mf2(v145, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v175 = __riscv_vsrl_vv_u16mf2(v174, v173, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v176 = __riscv_vand_vx_u16mf2(v175, 0x1, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v177 = __riscv_vsll_vx_u16mf2(v176, 0x4, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v178 = __riscv_vncvt_x_x_w_u8mf4(v177, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v179 = __riscv_vor_vv_u8mf4(v163, v171, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v180 = __riscv_vor_vv_u8mf4(v164, v178, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v181 = __riscv_vreinterpret_v_u8mf4_i8mf4(v179);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v182 = __riscv_vreinterpret_v_u8mf4_i8mf4(v180);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v183 = __riscv_vwmul_vv_i16mf2(v181, v158, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v184 = __riscv_vwmacc_vv_i16mf2(v183, v182, v162, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v185 = v146;
      vint32m1_t v186 = __riscv_vmv_v_x_i32m1(v185, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v187 = __riscv_vwredsum_vs_i16mf2_i32m1(v184, v186, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v188 = __riscv_vmv_x_s_i32m1_i32(v187);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v146 = v188;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v189 = v146;
    float v190 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v190 + ((v136 * v137) * (float) v189 + v139 * v141);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v191 = v6;
  v2[0] = v191;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_mf4_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 24;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v21 = v12 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    const uint8_t* v23 = v12 + 6;
    uint32_t v24 = (uint16_t)*(const uint16_t *)(v23);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v26 = __riscv_vsetvl_e32m1(16);
    for (size_t v27 = 0; v27 < 16; v27 += v26) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v28 = 16 - v27;
      size_t v29 = __riscv_vsetvl_e32m1(v28);
      const uint8_t* v30 = v12 + 8;
      const uint8_t* v31 = v30 + v27;
      const uint8_t* v32 = (const uint8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v33 = __riscv_vle8_v_u8mf4(v32, v29);
      const uint8_t* v34 = v14 + 4;
      const uint8_t* v35 = v34 + v27;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v37 = __riscv_vle8_v_i8mf4(v36, v29);
      const uint8_t* v38 = v14 + 20;
      const uint8_t* v39 = v38 + v27;
      const int8_t* v40 = (const int8_t*) v39;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v41 = __riscv_vle8_v_i8mf4(v40, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v42 = __riscv_vand_vx_u8mf4(v33, 0x0F, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v43 = __riscv_vsrl_vx_u8mf4(v33, 0x04, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v44 = __riscv_vid_v_u16mf2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v45 = __riscv_vadd_vx_u16mf2(v44, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v46 = __riscv_vmv_v_x_u16mf2(v22, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v47 = __riscv_vsrl_vv_u16mf2(v46, v45, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v48 = __riscv_vand_vx_u16mf2(v47, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v49 = __riscv_vsll_vx_u16mf2(v48, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v50 = __riscv_vncvt_x_x_w_u8mf4(v49, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v51 = __riscv_vid_v_u16mf2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v52 = __riscv_vadd_vx_u16mf2(v51, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v53 = __riscv_vmv_v_x_u16mf2(v24, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v54 = __riscv_vsrl_vv_u16mf2(v53, v52, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v55 = __riscv_vand_vx_u16mf2(v54, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v56 = __riscv_vsll_vx_u16mf2(v55, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v57 = __riscv_vncvt_x_x_w_u8mf4(v56, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v58 = __riscv_vor_vv_u8mf4(v42, v50, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v59 = __riscv_vor_vv_u8mf4(v43, v57, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v60 = __riscv_vreinterpret_v_u8mf4_i8mf4(v58);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v61 = __riscv_vreinterpret_v_u8mf4_i8mf4(v59);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v62 = __riscv_vwmul_vv_i16mf2(v60, v37, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v63 = __riscv_vwmacc_vv_i16mf2(v62, v61, v41, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v64 = v25;
      vint32m1_t v65 = __riscv_vmv_v_x_i32m1(v64, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v66 = __riscv_vwredsum_vs_i16mf2_i32m1(v63, v65, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v67 = __riscv_vmv_x_s_i32m1_i32(v66);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v25 = v67;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v68 = v10 + 1;
    size_t v69 = v68 * 24;
    const uint8_t* v70 = v3 + v69;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v71 = v10 + 1;
    size_t v72 = v71 * 36;
    const uint8_t* v73 = v4 + v72;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v74 = (float)*(const _Float16 *)(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v73);
    const uint8_t* v76 = v70 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v77 = (float)*(const _Float16 *)(v76);
    const uint8_t* v78 = v73 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v79 = (float)*(const _Float16 *)(v78);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v80 = v70 + 4;
    uint32_t v81 = (uint16_t)*(const uint16_t *)(v80);
    const uint8_t* v82 = v70 + 6;
    uint32_t v83 = (uint16_t)*(const uint16_t *)(v82);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v84;
    v84 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v85 = __riscv_vsetvl_e32m1(16);
    for (size_t v86 = 0; v86 < 16; v86 += v85) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v87 = 16 - v86;
      size_t v88 = __riscv_vsetvl_e32m1(v87);
      const uint8_t* v89 = v70 + 8;
      const uint8_t* v90 = v89 + v86;
      const uint8_t* v91 = (const uint8_t*) v90;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v92 = __riscv_vle8_v_u8mf4(v91, v88);
      const uint8_t* v93 = v73 + 4;
      const uint8_t* v94 = v93 + v86;
      const int8_t* v95 = (const int8_t*) v94;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v96 = __riscv_vle8_v_i8mf4(v95, v88);
      const uint8_t* v97 = v73 + 20;
      const uint8_t* v98 = v97 + v86;
      const int8_t* v99 = (const int8_t*) v98;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v100 = __riscv_vle8_v_i8mf4(v99, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v101 = __riscv_vand_vx_u8mf4(v92, 0x0F, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v102 = __riscv_vsrl_vx_u8mf4(v92, 0x04, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v103 = __riscv_vid_v_u16mf2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v104 = __riscv_vadd_vx_u16mf2(v103, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v105 = __riscv_vmv_v_x_u16mf2(v81, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v106 = __riscv_vsrl_vv_u16mf2(v105, v104, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v107 = __riscv_vand_vx_u16mf2(v106, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v108 = __riscv_vsll_vx_u16mf2(v107, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v109 = __riscv_vncvt_x_x_w_u8mf4(v108, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v110 = __riscv_vid_v_u16mf2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v111 = __riscv_vadd_vx_u16mf2(v110, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v112 = __riscv_vmv_v_x_u16mf2(v83, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v113 = __riscv_vsrl_vv_u16mf2(v112, v111, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v114 = __riscv_vand_vx_u16mf2(v113, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v115 = __riscv_vsll_vx_u16mf2(v114, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v116 = __riscv_vncvt_x_x_w_u8mf4(v115, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v117 = __riscv_vor_vv_u8mf4(v101, v109, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v118 = __riscv_vor_vv_u8mf4(v102, v116, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v119 = __riscv_vreinterpret_v_u8mf4_i8mf4(v117);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v120 = __riscv_vreinterpret_v_u8mf4_i8mf4(v118);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v121 = __riscv_vwmul_vv_i16mf2(v119, v96, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v122 = __riscv_vwmacc_vv_i16mf2(v121, v120, v100, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v123 = v84;
      vint32m1_t v124 = __riscv_vmv_v_x_i32m1(v123, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v125 = __riscv_vwredsum_vs_i16mf2_i32m1(v122, v124, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v126 = __riscv_vmv_x_s_i32m1_i32(v125);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v84 = v126;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v127 = v10 + 2;
    size_t v128 = v127 * 24;
    const uint8_t* v129 = v3 + v128;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v130 = v10 + 2;
    size_t v131 = v130 * 36;
    const uint8_t* v132 = v4 + v131;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v134 = (float)*(const _Float16 *)(v132);
    const uint8_t* v135 = v129 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v136 = (float)*(const _Float16 *)(v135);
    const uint8_t* v137 = v132 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v138 = (float)*(const _Float16 *)(v137);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v139 = v129 + 4;
    uint32_t v140 = (uint16_t)*(const uint16_t *)(v139);
    const uint8_t* v141 = v129 + 6;
    uint32_t v142 = (uint16_t)*(const uint16_t *)(v141);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v143;
    v143 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v144 = __riscv_vsetvl_e32m1(16);
    for (size_t v145 = 0; v145 < 16; v145 += v144) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v146 = 16 - v145;
      size_t v147 = __riscv_vsetvl_e32m1(v146);
      const uint8_t* v148 = v129 + 8;
      const uint8_t* v149 = v148 + v145;
      const uint8_t* v150 = (const uint8_t*) v149;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v151 = __riscv_vle8_v_u8mf4(v150, v147);
      const uint8_t* v152 = v132 + 4;
      const uint8_t* v153 = v152 + v145;
      const int8_t* v154 = (const int8_t*) v153;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v155 = __riscv_vle8_v_i8mf4(v154, v147);
      const uint8_t* v156 = v132 + 20;
      const uint8_t* v157 = v156 + v145;
      const int8_t* v158 = (const int8_t*) v157;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v159 = __riscv_vle8_v_i8mf4(v158, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v160 = __riscv_vand_vx_u8mf4(v151, 0x0F, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v161 = __riscv_vsrl_vx_u8mf4(v151, 0x04, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v162 = __riscv_vid_v_u16mf2(v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v163 = __riscv_vadd_vx_u16mf2(v162, v145, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v164 = __riscv_vmv_v_x_u16mf2(v140, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v165 = __riscv_vsrl_vv_u16mf2(v164, v163, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v166 = __riscv_vand_vx_u16mf2(v165, 0x1, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v167 = __riscv_vsll_vx_u16mf2(v166, 0x4, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v168 = __riscv_vncvt_x_x_w_u8mf4(v167, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v169 = __riscv_vid_v_u16mf2(v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v170 = __riscv_vadd_vx_u16mf2(v169, v145, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v171 = __riscv_vmv_v_x_u16mf2(v142, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v172 = __riscv_vsrl_vv_u16mf2(v171, v170, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v173 = __riscv_vand_vx_u16mf2(v172, 0x1, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v174 = __riscv_vsll_vx_u16mf2(v173, 0x4, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v175 = __riscv_vncvt_x_x_w_u8mf4(v174, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v176 = __riscv_vor_vv_u8mf4(v160, v168, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v177 = __riscv_vor_vv_u8mf4(v161, v175, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v178 = __riscv_vreinterpret_v_u8mf4_i8mf4(v176);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v179 = __riscv_vreinterpret_v_u8mf4_i8mf4(v177);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v180 = __riscv_vwmul_vv_i16mf2(v178, v155, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v181 = __riscv_vwmacc_vv_i16mf2(v180, v179, v159, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v182 = v143;
      vint32m1_t v183 = __riscv_vmv_v_x_i32m1(v182, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v184 = __riscv_vwredsum_vs_i16mf2_i32m1(v181, v183, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v185 = __riscv_vmv_x_s_i32m1_i32(v184);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v143 = v185;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v186 = v10 + 3;
    size_t v187 = v186 * 24;
    const uint8_t* v188 = v3 + v187;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v189 = v10 + 3;
    size_t v190 = v189 * 36;
    const uint8_t* v191 = v4 + v190;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v192 = (float)*(const _Float16 *)(v188);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v193 = (float)*(const _Float16 *)(v191);
    const uint8_t* v194 = v188 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v195 = (float)*(const _Float16 *)(v194);
    const uint8_t* v196 = v191 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v197 = (float)*(const _Float16 *)(v196);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v198 = v188 + 4;
    uint32_t v199 = (uint16_t)*(const uint16_t *)(v198);
    const uint8_t* v200 = v188 + 6;
    uint32_t v201 = (uint16_t)*(const uint16_t *)(v200);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v202;
    v202 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v203 = __riscv_vsetvl_e32m1(16);
    for (size_t v204 = 0; v204 < 16; v204 += v203) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v205 = 16 - v204;
      size_t v206 = __riscv_vsetvl_e32m1(v205);
      const uint8_t* v207 = v188 + 8;
      const uint8_t* v208 = v207 + v204;
      const uint8_t* v209 = (const uint8_t*) v208;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v210 = __riscv_vle8_v_u8mf4(v209, v206);
      const uint8_t* v211 = v191 + 4;
      const uint8_t* v212 = v211 + v204;
      const int8_t* v213 = (const int8_t*) v212;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v214 = __riscv_vle8_v_i8mf4(v213, v206);
      const uint8_t* v215 = v191 + 20;
      const uint8_t* v216 = v215 + v204;
      const int8_t* v217 = (const int8_t*) v216;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v218 = __riscv_vle8_v_i8mf4(v217, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v219 = __riscv_vand_vx_u8mf4(v210, 0x0F, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v220 = __riscv_vsrl_vx_u8mf4(v210, 0x04, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v221 = __riscv_vid_v_u16mf2(v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v222 = __riscv_vadd_vx_u16mf2(v221, v204, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v223 = __riscv_vmv_v_x_u16mf2(v199, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v224 = __riscv_vsrl_vv_u16mf2(v223, v222, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v225 = __riscv_vand_vx_u16mf2(v224, 0x1, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v226 = __riscv_vsll_vx_u16mf2(v225, 0x4, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v227 = __riscv_vncvt_x_x_w_u8mf4(v226, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v228 = __riscv_vid_v_u16mf2(v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v229 = __riscv_vadd_vx_u16mf2(v228, v204, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v230 = __riscv_vmv_v_x_u16mf2(v201, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v231 = __riscv_vsrl_vv_u16mf2(v230, v229, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v232 = __riscv_vand_vx_u16mf2(v231, 0x1, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v233 = __riscv_vsll_vx_u16mf2(v232, 0x4, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v234 = __riscv_vncvt_x_x_w_u8mf4(v233, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v235 = __riscv_vor_vv_u8mf4(v219, v227, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v236 = __riscv_vor_vv_u8mf4(v220, v234, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v237 = __riscv_vreinterpret_v_u8mf4_i8mf4(v235);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v238 = __riscv_vreinterpret_v_u8mf4_i8mf4(v236);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v239 = __riscv_vwmul_vv_i16mf2(v237, v214, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v240 = __riscv_vwmacc_vv_i16mf2(v239, v238, v218, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v241 = v202;
      vint32m1_t v242 = __riscv_vmv_v_x_i32m1(v241, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v243 = __riscv_vwredsum_vs_i16mf2_i32m1(v240, v242, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v244 = __riscv_vmv_x_s_i32m1_i32(v243);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v202 = v244;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v245 = v25;
    float v246 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v246 + ((v15 * v16) * (float) v245 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v247 = v84;
    float v248 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v248 + ((v74 * v75) * (float) v247 + v77 * v79);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v249 = v143;
    float v250 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v250 + ((v133 * v134) * (float) v249 + v136 * v138);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v251 = v202;
    float v252 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v252 + ((v192 * v193) * (float) v251 + v195 * v197);
  }
  for (size_t v253 = v9; v253 < v7; v253 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v254 = v253 * 24;
    const uint8_t* v255 = v3 + v254;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v256 = v253 * 36;
    const uint8_t* v257 = v4 + v256;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v258 = (float)*(const _Float16 *)(v255);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v259 = (float)*(const _Float16 *)(v257);
    const uint8_t* v260 = v255 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v261 = (float)*(const _Float16 *)(v260);
    const uint8_t* v262 = v257 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v263 = (float)*(const _Float16 *)(v262);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v264 = v255 + 4;
    uint32_t v265 = (uint16_t)*(const uint16_t *)(v264);
    const uint8_t* v266 = v255 + 6;
    uint32_t v267 = (uint16_t)*(const uint16_t *)(v266);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v268;
    v268 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v269 = __riscv_vsetvl_e32m1(16);
    for (size_t v270 = 0; v270 < 16; v270 += v269) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v271 = 16 - v270;
      size_t v272 = __riscv_vsetvl_e32m1(v271);
      const uint8_t* v273 = v255 + 8;
      const uint8_t* v274 = v273 + v270;
      const uint8_t* v275 = (const uint8_t*) v274;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf4
      vuint8mf4_t v276 = __riscv_vle8_v_u8mf4(v275, v272);
      const uint8_t* v277 = v257 + 4;
      const uint8_t* v278 = v277 + v270;
      const int8_t* v279 = (const int8_t*) v278;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v280 = __riscv_vle8_v_i8mf4(v279, v272);
      const uint8_t* v281 = v257 + 20;
      const uint8_t* v282 = v281 + v270;
      const int8_t* v283 = (const int8_t*) v282;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v284 = __riscv_vle8_v_i8mf4(v283, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf4
      vuint8mf4_t v285 = __riscv_vand_vx_u8mf4(v276, 0x0F, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf4
      vuint8mf4_t v286 = __riscv_vsrl_vx_u8mf4(v276, 0x04, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v287 = __riscv_vid_v_u16mf2(v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v288 = __riscv_vadd_vx_u16mf2(v287, v270, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v289 = __riscv_vmv_v_x_u16mf2(v265, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v290 = __riscv_vsrl_vv_u16mf2(v289, v288, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v291 = __riscv_vand_vx_u16mf2(v290, 0x1, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v292 = __riscv_vsll_vx_u16mf2(v291, 0x4, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v293 = __riscv_vncvt_x_x_w_u8mf4(v292, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16mf2
      vuint16mf2_t v294 = __riscv_vid_v_u16mf2(v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16mf2
      vuint16mf2_t v295 = __riscv_vadd_vx_u16mf2(v294, v270, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16mf2
      vuint16mf2_t v296 = __riscv_vmv_v_x_u16mf2(v267, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16mf2
      vuint16mf2_t v297 = __riscv_vsrl_vv_u16mf2(v296, v295, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16mf2
      vuint16mf2_t v298 = __riscv_vand_vx_u16mf2(v297, 0x1, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16mf2
      vuint16mf2_t v299 = __riscv_vsll_vx_u16mf2(v298, 0x4, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf4
      vuint8mf4_t v300 = __riscv_vncvt_x_x_w_u8mf4(v299, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v301 = __riscv_vor_vv_u8mf4(v285, v293, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf4
      vuint8mf4_t v302 = __riscv_vor_vv_u8mf4(v286, v300, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v303 = __riscv_vreinterpret_v_u8mf4_i8mf4(v301);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf4_i8mf4
      vint8mf4_t v304 = __riscv_vreinterpret_v_u8mf4_i8mf4(v302);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v305 = __riscv_vwmul_vv_i16mf2(v303, v280, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16mf2
      vint16mf2_t v306 = __riscv_vwmacc_vv_i16mf2(v305, v304, v284, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v307 = v268;
      vint32m1_t v308 = __riscv_vmv_v_x_i32m1(v307, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v309 = __riscv_vwredsum_vs_i16mf2_i32m1(v306, v308, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v310 = __riscv_vmv_x_s_i32m1_i32(v309);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v268 = v310;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v311 = v268;
    float v312 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v312 + ((v258 * v259) * (float) v311 + v261 * v263);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v313 = v6;
  v2[0] = v313;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 24;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 36;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    const uint8_t* v15 = v10 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v15);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v19 = v10 + 4;
    uint32_t v20 = (uint16_t)*(const uint16_t *)(v19);
    const uint8_t* v21 = v10 + 6;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v23;
    v23 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v24 = __riscv_vsetvl_e8m1(16);
    for (size_t v25 = 0; v25 < 16; v25 += v24) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v26 = 16 - v25;
      size_t v27 = __riscv_vsetvl_e8m1(v26);
      const uint8_t* v28 = v10 + 8;
      const uint8_t* v29 = v28 + v25;
      const uint8_t* v30 = (const uint8_t*) v29;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v31 = __riscv_vle8_v_u8m1(v30, v27);
      const uint8_t* v32 = v12 + 4;
      const uint8_t* v33 = v32 + v25;
      const int8_t* v34 = (const int8_t*) v33;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v35 = __riscv_vle8_v_i8m1(v34, v27);
      const uint8_t* v36 = v12 + 20;
      const uint8_t* v37 = v36 + v25;
      const int8_t* v38 = (const int8_t*) v37;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v39 = __riscv_vle8_v_i8m1(v38, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v40 = __riscv_vand_vx_u8m1(v31, 0x0F, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v41 = __riscv_vsrl_vx_u8m1(v31, 0x04, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v42 = __riscv_vid_v_u16m2(v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v43 = __riscv_vadd_vx_u16m2(v42, v25, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v44 = __riscv_vmv_v_x_u16m2(v20, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v45 = __riscv_vsrl_vv_u16m2(v44, v43, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v46 = __riscv_vand_vx_u16m2(v45, 0x1, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v47 = __riscv_vsll_vx_u16m2(v46, 0x4, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v48 = __riscv_vncvt_x_x_w_u8m1(v47, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v49 = __riscv_vid_v_u16m2(v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v50 = __riscv_vadd_vx_u16m2(v49, v25, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v51 = __riscv_vmv_v_x_u16m2(v22, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v52 = __riscv_vsrl_vv_u16m2(v51, v50, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v53 = __riscv_vand_vx_u16m2(v52, 0x1, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v54 = __riscv_vsll_vx_u16m2(v53, 0x4, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v55 = __riscv_vncvt_x_x_w_u8m1(v54, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v56 = __riscv_vor_vv_u8m1(v40, v48, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v57 = __riscv_vor_vv_u8m1(v41, v55, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v56);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v59 = __riscv_vreinterpret_v_u8m1_i8m1(v57);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v58, v35, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v61 = __riscv_vwmacc_vv_i16m2(v60, v59, v39, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v62 = v23;
      vint32m1_t v63 = __riscv_vmv_v_x_i32m1(v62, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v64 = __riscv_vwredsum_vs_i16m2_i32m1(v61, v63, v27);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v65 = __riscv_vmv_x_s_i32m1_i32(v64);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v23 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v66 = v23;
    float v67 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v67 + ((v13 * v14) * (float) v66 + v16 * v18);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v68 = v6;
  v2[0] = v68;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 24;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v21 = v12 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    const uint8_t* v23 = v12 + 6;
    uint32_t v24 = (uint16_t)*(const uint16_t *)(v23);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v26 = __riscv_vsetvl_e8m1(16);
    for (size_t v27 = 0; v27 < 16; v27 += v26) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v28 = 16 - v27;
      size_t v29 = __riscv_vsetvl_e8m1(v28);
      const uint8_t* v30 = v12 + 8;
      const uint8_t* v31 = v30 + v27;
      const uint8_t* v32 = (const uint8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v33 = __riscv_vle8_v_u8m1(v32, v29);
      const uint8_t* v34 = v14 + 4;
      const uint8_t* v35 = v34 + v27;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v29);
      const uint8_t* v38 = v14 + 20;
      const uint8_t* v39 = v38 + v27;
      const int8_t* v40 = (const int8_t*) v39;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v41 = __riscv_vle8_v_i8m1(v40, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v42 = __riscv_vand_vx_u8m1(v33, 0x0F, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v43 = __riscv_vsrl_vx_u8m1(v33, 0x04, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v44 = __riscv_vid_v_u16m2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v45 = __riscv_vadd_vx_u16m2(v44, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v46 = __riscv_vmv_v_x_u16m2(v22, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v47 = __riscv_vsrl_vv_u16m2(v46, v45, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v48 = __riscv_vand_vx_u16m2(v47, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v49 = __riscv_vsll_vx_u16m2(v48, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v50 = __riscv_vncvt_x_x_w_u8m1(v49, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v51 = __riscv_vid_v_u16m2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v52 = __riscv_vadd_vx_u16m2(v51, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v53 = __riscv_vmv_v_x_u16m2(v24, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v54 = __riscv_vsrl_vv_u16m2(v53, v52, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v55 = __riscv_vand_vx_u16m2(v54, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v56 = __riscv_vsll_vx_u16m2(v55, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v57 = __riscv_vncvt_x_x_w_u8m1(v56, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v58 = __riscv_vor_vv_u8m1(v42, v50, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v59 = __riscv_vor_vv_u8m1(v43, v57, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v60 = __riscv_vreinterpret_v_u8m1_i8m1(v58);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v61 = __riscv_vreinterpret_v_u8m1_i8m1(v59);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v62 = __riscv_vwmul_vv_i16m2(v60, v37, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v63 = __riscv_vwmacc_vv_i16m2(v62, v61, v41, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v64 = v25;
      vint32m1_t v65 = __riscv_vmv_v_x_i32m1(v64, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v66 = __riscv_vwredsum_vs_i16m2_i32m1(v63, v65, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v67 = __riscv_vmv_x_s_i32m1_i32(v66);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v25 = v67;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v68 = v10 + 1;
    size_t v69 = v68 * 24;
    const uint8_t* v70 = v3 + v69;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v71 = v10 + 1;
    size_t v72 = v71 * 36;
    const uint8_t* v73 = v4 + v72;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v74 = (float)*(const _Float16 *)(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v73);
    const uint8_t* v76 = v70 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v77 = (float)*(const _Float16 *)(v76);
    const uint8_t* v78 = v73 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v79 = (float)*(const _Float16 *)(v78);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v80 = v70 + 4;
    uint32_t v81 = (uint16_t)*(const uint16_t *)(v80);
    const uint8_t* v82 = v70 + 6;
    uint32_t v83 = (uint16_t)*(const uint16_t *)(v82);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v84;
    v84 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v85 = __riscv_vsetvl_e8m1(16);
    for (size_t v86 = 0; v86 < 16; v86 += v85) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v87 = 16 - v86;
      size_t v88 = __riscv_vsetvl_e8m1(v87);
      const uint8_t* v89 = v70 + 8;
      const uint8_t* v90 = v89 + v86;
      const uint8_t* v91 = (const uint8_t*) v90;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v92 = __riscv_vle8_v_u8m1(v91, v88);
      const uint8_t* v93 = v73 + 4;
      const uint8_t* v94 = v93 + v86;
      const int8_t* v95 = (const int8_t*) v94;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v96 = __riscv_vle8_v_i8m1(v95, v88);
      const uint8_t* v97 = v73 + 20;
      const uint8_t* v98 = v97 + v86;
      const int8_t* v99 = (const int8_t*) v98;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v100 = __riscv_vle8_v_i8m1(v99, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v101 = __riscv_vand_vx_u8m1(v92, 0x0F, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v102 = __riscv_vsrl_vx_u8m1(v92, 0x04, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v103 = __riscv_vid_v_u16m2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v104 = __riscv_vadd_vx_u16m2(v103, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v105 = __riscv_vmv_v_x_u16m2(v81, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v106 = __riscv_vsrl_vv_u16m2(v105, v104, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v107 = __riscv_vand_vx_u16m2(v106, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v108 = __riscv_vsll_vx_u16m2(v107, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v109 = __riscv_vncvt_x_x_w_u8m1(v108, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v110 = __riscv_vid_v_u16m2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v111 = __riscv_vadd_vx_u16m2(v110, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v112 = __riscv_vmv_v_x_u16m2(v83, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v113 = __riscv_vsrl_vv_u16m2(v112, v111, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v114 = __riscv_vand_vx_u16m2(v113, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v115 = __riscv_vsll_vx_u16m2(v114, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v116 = __riscv_vncvt_x_x_w_u8m1(v115, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v117 = __riscv_vor_vv_u8m1(v101, v109, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v118 = __riscv_vor_vv_u8m1(v102, v116, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v119 = __riscv_vreinterpret_v_u8m1_i8m1(v117);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v120 = __riscv_vreinterpret_v_u8m1_i8m1(v118);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v121 = __riscv_vwmul_vv_i16m2(v119, v96, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v122 = __riscv_vwmacc_vv_i16m2(v121, v120, v100, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v123 = v84;
      vint32m1_t v124 = __riscv_vmv_v_x_i32m1(v123, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v125 = __riscv_vwredsum_vs_i16m2_i32m1(v122, v124, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v126 = __riscv_vmv_x_s_i32m1_i32(v125);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v84 = v126;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v127 = v25;
    float v128 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v128 + ((v15 * v16) * (float) v127 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v129 = v84;
    float v130 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v130 + ((v74 * v75) * (float) v129 + v77 * v79);
  }
  for (size_t v131 = v9; v131 < v7; v131 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v132 = v131 * 24;
    const uint8_t* v133 = v3 + v132;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v134 = v131 * 36;
    const uint8_t* v135 = v4 + v134;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v136 = (float)*(const _Float16 *)(v133);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v137 = (float)*(const _Float16 *)(v135);
    const uint8_t* v138 = v133 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v139 = (float)*(const _Float16 *)(v138);
    const uint8_t* v140 = v135 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v141 = (float)*(const _Float16 *)(v140);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v142 = v133 + 4;
    uint32_t v143 = (uint16_t)*(const uint16_t *)(v142);
    const uint8_t* v144 = v133 + 6;
    uint32_t v145 = (uint16_t)*(const uint16_t *)(v144);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v146;
    v146 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v147 = __riscv_vsetvl_e8m1(16);
    for (size_t v148 = 0; v148 < 16; v148 += v147) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v149 = 16 - v148;
      size_t v150 = __riscv_vsetvl_e8m1(v149);
      const uint8_t* v151 = v133 + 8;
      const uint8_t* v152 = v151 + v148;
      const uint8_t* v153 = (const uint8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v154 = __riscv_vle8_v_u8m1(v153, v150);
      const uint8_t* v155 = v135 + 4;
      const uint8_t* v156 = v155 + v148;
      const int8_t* v157 = (const int8_t*) v156;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v158 = __riscv_vle8_v_i8m1(v157, v150);
      const uint8_t* v159 = v135 + 20;
      const uint8_t* v160 = v159 + v148;
      const int8_t* v161 = (const int8_t*) v160;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v162 = __riscv_vle8_v_i8m1(v161, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v163 = __riscv_vand_vx_u8m1(v154, 0x0F, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v164 = __riscv_vsrl_vx_u8m1(v154, 0x04, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v165 = __riscv_vid_v_u16m2(v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v166 = __riscv_vadd_vx_u16m2(v165, v148, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v167 = __riscv_vmv_v_x_u16m2(v143, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v168 = __riscv_vsrl_vv_u16m2(v167, v166, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v169 = __riscv_vand_vx_u16m2(v168, 0x1, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v170 = __riscv_vsll_vx_u16m2(v169, 0x4, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v171 = __riscv_vncvt_x_x_w_u8m1(v170, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v172 = __riscv_vid_v_u16m2(v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v173 = __riscv_vadd_vx_u16m2(v172, v148, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v174 = __riscv_vmv_v_x_u16m2(v145, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v175 = __riscv_vsrl_vv_u16m2(v174, v173, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v176 = __riscv_vand_vx_u16m2(v175, 0x1, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v177 = __riscv_vsll_vx_u16m2(v176, 0x4, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v178 = __riscv_vncvt_x_x_w_u8m1(v177, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v179 = __riscv_vor_vv_u8m1(v163, v171, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v180 = __riscv_vor_vv_u8m1(v164, v178, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v181 = __riscv_vreinterpret_v_u8m1_i8m1(v179);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v182 = __riscv_vreinterpret_v_u8m1_i8m1(v180);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v183 = __riscv_vwmul_vv_i16m2(v181, v158, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v184 = __riscv_vwmacc_vv_i16m2(v183, v182, v162, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v185 = v146;
      vint32m1_t v186 = __riscv_vmv_v_x_i32m1(v185, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v187 = __riscv_vwredsum_vs_i16m2_i32m1(v184, v186, v150);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v188 = __riscv_vmv_x_s_i32m1_i32(v187);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v146 = v188;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v189 = v146;
    float v190 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v190 + ((v136 * v137) * (float) v189 + v139 * v141);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v191 = v6;
  v2[0] = v191;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 24;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 36;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    const uint8_t* v17 = v12 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    const uint8_t* v19 = v14 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v19);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v21 = v12 + 4;
    uint32_t v22 = (uint16_t)*(const uint16_t *)(v21);
    const uint8_t* v23 = v12 + 6;
    uint32_t v24 = (uint16_t)*(const uint16_t *)(v23);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v25;
    v25 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v26 = __riscv_vsetvl_e8m1(16);
    for (size_t v27 = 0; v27 < 16; v27 += v26) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v28 = 16 - v27;
      size_t v29 = __riscv_vsetvl_e8m1(v28);
      const uint8_t* v30 = v12 + 8;
      const uint8_t* v31 = v30 + v27;
      const uint8_t* v32 = (const uint8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v33 = __riscv_vle8_v_u8m1(v32, v29);
      const uint8_t* v34 = v14 + 4;
      const uint8_t* v35 = v34 + v27;
      const int8_t* v36 = (const int8_t*) v35;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, v29);
      const uint8_t* v38 = v14 + 20;
      const uint8_t* v39 = v38 + v27;
      const int8_t* v40 = (const int8_t*) v39;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v41 = __riscv_vle8_v_i8m1(v40, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v42 = __riscv_vand_vx_u8m1(v33, 0x0F, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v43 = __riscv_vsrl_vx_u8m1(v33, 0x04, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v44 = __riscv_vid_v_u16m2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v45 = __riscv_vadd_vx_u16m2(v44, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v46 = __riscv_vmv_v_x_u16m2(v22, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v47 = __riscv_vsrl_vv_u16m2(v46, v45, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v48 = __riscv_vand_vx_u16m2(v47, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v49 = __riscv_vsll_vx_u16m2(v48, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v50 = __riscv_vncvt_x_x_w_u8m1(v49, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v51 = __riscv_vid_v_u16m2(v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v52 = __riscv_vadd_vx_u16m2(v51, v27, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v53 = __riscv_vmv_v_x_u16m2(v24, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v54 = __riscv_vsrl_vv_u16m2(v53, v52, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v55 = __riscv_vand_vx_u16m2(v54, 0x1, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v56 = __riscv_vsll_vx_u16m2(v55, 0x4, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v57 = __riscv_vncvt_x_x_w_u8m1(v56, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v58 = __riscv_vor_vv_u8m1(v42, v50, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v59 = __riscv_vor_vv_u8m1(v43, v57, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v60 = __riscv_vreinterpret_v_u8m1_i8m1(v58);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v61 = __riscv_vreinterpret_v_u8m1_i8m1(v59);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v62 = __riscv_vwmul_vv_i16m2(v60, v37, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v63 = __riscv_vwmacc_vv_i16m2(v62, v61, v41, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v64 = v25;
      vint32m1_t v65 = __riscv_vmv_v_x_i32m1(v64, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v66 = __riscv_vwredsum_vs_i16m2_i32m1(v63, v65, v29);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v67 = __riscv_vmv_x_s_i32m1_i32(v66);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v25 = v67;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v68 = v10 + 1;
    size_t v69 = v68 * 24;
    const uint8_t* v70 = v3 + v69;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v71 = v10 + 1;
    size_t v72 = v71 * 36;
    const uint8_t* v73 = v4 + v72;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v74 = (float)*(const _Float16 *)(v70);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v73);
    const uint8_t* v76 = v70 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v77 = (float)*(const _Float16 *)(v76);
    const uint8_t* v78 = v73 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v79 = (float)*(const _Float16 *)(v78);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v80 = v70 + 4;
    uint32_t v81 = (uint16_t)*(const uint16_t *)(v80);
    const uint8_t* v82 = v70 + 6;
    uint32_t v83 = (uint16_t)*(const uint16_t *)(v82);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v84;
    v84 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v85 = __riscv_vsetvl_e8m1(16);
    for (size_t v86 = 0; v86 < 16; v86 += v85) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v87 = 16 - v86;
      size_t v88 = __riscv_vsetvl_e8m1(v87);
      const uint8_t* v89 = v70 + 8;
      const uint8_t* v90 = v89 + v86;
      const uint8_t* v91 = (const uint8_t*) v90;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v92 = __riscv_vle8_v_u8m1(v91, v88);
      const uint8_t* v93 = v73 + 4;
      const uint8_t* v94 = v93 + v86;
      const int8_t* v95 = (const int8_t*) v94;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v96 = __riscv_vle8_v_i8m1(v95, v88);
      const uint8_t* v97 = v73 + 20;
      const uint8_t* v98 = v97 + v86;
      const int8_t* v99 = (const int8_t*) v98;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v100 = __riscv_vle8_v_i8m1(v99, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v101 = __riscv_vand_vx_u8m1(v92, 0x0F, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v102 = __riscv_vsrl_vx_u8m1(v92, 0x04, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v103 = __riscv_vid_v_u16m2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v104 = __riscv_vadd_vx_u16m2(v103, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v105 = __riscv_vmv_v_x_u16m2(v81, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v106 = __riscv_vsrl_vv_u16m2(v105, v104, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v107 = __riscv_vand_vx_u16m2(v106, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v108 = __riscv_vsll_vx_u16m2(v107, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v109 = __riscv_vncvt_x_x_w_u8m1(v108, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v110 = __riscv_vid_v_u16m2(v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v111 = __riscv_vadd_vx_u16m2(v110, v86, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v112 = __riscv_vmv_v_x_u16m2(v83, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v113 = __riscv_vsrl_vv_u16m2(v112, v111, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v114 = __riscv_vand_vx_u16m2(v113, 0x1, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v115 = __riscv_vsll_vx_u16m2(v114, 0x4, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v116 = __riscv_vncvt_x_x_w_u8m1(v115, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v117 = __riscv_vor_vv_u8m1(v101, v109, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v118 = __riscv_vor_vv_u8m1(v102, v116, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v119 = __riscv_vreinterpret_v_u8m1_i8m1(v117);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v120 = __riscv_vreinterpret_v_u8m1_i8m1(v118);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v121 = __riscv_vwmul_vv_i16m2(v119, v96, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v122 = __riscv_vwmacc_vv_i16m2(v121, v120, v100, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v123 = v84;
      vint32m1_t v124 = __riscv_vmv_v_x_i32m1(v123, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v125 = __riscv_vwredsum_vs_i16m2_i32m1(v122, v124, v88);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v126 = __riscv_vmv_x_s_i32m1_i32(v125);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v84 = v126;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v127 = v10 + 2;
    size_t v128 = v127 * 24;
    const uint8_t* v129 = v3 + v128;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v130 = v10 + 2;
    size_t v131 = v130 * 36;
    const uint8_t* v132 = v4 + v131;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v134 = (float)*(const _Float16 *)(v132);
    const uint8_t* v135 = v129 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v136 = (float)*(const _Float16 *)(v135);
    const uint8_t* v137 = v132 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v138 = (float)*(const _Float16 *)(v137);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v139 = v129 + 4;
    uint32_t v140 = (uint16_t)*(const uint16_t *)(v139);
    const uint8_t* v141 = v129 + 6;
    uint32_t v142 = (uint16_t)*(const uint16_t *)(v141);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v143;
    v143 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v144 = __riscv_vsetvl_e8m1(16);
    for (size_t v145 = 0; v145 < 16; v145 += v144) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v146 = 16 - v145;
      size_t v147 = __riscv_vsetvl_e8m1(v146);
      const uint8_t* v148 = v129 + 8;
      const uint8_t* v149 = v148 + v145;
      const uint8_t* v150 = (const uint8_t*) v149;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v151 = __riscv_vle8_v_u8m1(v150, v147);
      const uint8_t* v152 = v132 + 4;
      const uint8_t* v153 = v152 + v145;
      const int8_t* v154 = (const int8_t*) v153;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v155 = __riscv_vle8_v_i8m1(v154, v147);
      const uint8_t* v156 = v132 + 20;
      const uint8_t* v157 = v156 + v145;
      const int8_t* v158 = (const int8_t*) v157;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v159 = __riscv_vle8_v_i8m1(v158, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v160 = __riscv_vand_vx_u8m1(v151, 0x0F, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v161 = __riscv_vsrl_vx_u8m1(v151, 0x04, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v162 = __riscv_vid_v_u16m2(v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v163 = __riscv_vadd_vx_u16m2(v162, v145, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v164 = __riscv_vmv_v_x_u16m2(v140, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v165 = __riscv_vsrl_vv_u16m2(v164, v163, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v166 = __riscv_vand_vx_u16m2(v165, 0x1, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v167 = __riscv_vsll_vx_u16m2(v166, 0x4, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v168 = __riscv_vncvt_x_x_w_u8m1(v167, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v169 = __riscv_vid_v_u16m2(v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v170 = __riscv_vadd_vx_u16m2(v169, v145, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v171 = __riscv_vmv_v_x_u16m2(v142, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v172 = __riscv_vsrl_vv_u16m2(v171, v170, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v173 = __riscv_vand_vx_u16m2(v172, 0x1, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v174 = __riscv_vsll_vx_u16m2(v173, 0x4, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v175 = __riscv_vncvt_x_x_w_u8m1(v174, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v176 = __riscv_vor_vv_u8m1(v160, v168, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v177 = __riscv_vor_vv_u8m1(v161, v175, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v178 = __riscv_vreinterpret_v_u8m1_i8m1(v176);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v179 = __riscv_vreinterpret_v_u8m1_i8m1(v177);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v180 = __riscv_vwmul_vv_i16m2(v178, v155, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v181 = __riscv_vwmacc_vv_i16m2(v180, v179, v159, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v182 = v143;
      vint32m1_t v183 = __riscv_vmv_v_x_i32m1(v182, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v184 = __riscv_vwredsum_vs_i16m2_i32m1(v181, v183, v147);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v185 = __riscv_vmv_x_s_i32m1_i32(v184);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v143 = v185;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v186 = v10 + 3;
    size_t v187 = v186 * 24;
    const uint8_t* v188 = v3 + v187;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v189 = v10 + 3;
    size_t v190 = v189 * 36;
    const uint8_t* v191 = v4 + v190;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v192 = (float)*(const _Float16 *)(v188);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v193 = (float)*(const _Float16 *)(v191);
    const uint8_t* v194 = v188 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v195 = (float)*(const _Float16 *)(v194);
    const uint8_t* v196 = v191 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v197 = (float)*(const _Float16 *)(v196);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v198 = v188 + 4;
    uint32_t v199 = (uint16_t)*(const uint16_t *)(v198);
    const uint8_t* v200 = v188 + 6;
    uint32_t v201 = (uint16_t)*(const uint16_t *)(v200);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v202;
    v202 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v203 = __riscv_vsetvl_e8m1(16);
    for (size_t v204 = 0; v204 < 16; v204 += v203) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v205 = 16 - v204;
      size_t v206 = __riscv_vsetvl_e8m1(v205);
      const uint8_t* v207 = v188 + 8;
      const uint8_t* v208 = v207 + v204;
      const uint8_t* v209 = (const uint8_t*) v208;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v210 = __riscv_vle8_v_u8m1(v209, v206);
      const uint8_t* v211 = v191 + 4;
      const uint8_t* v212 = v211 + v204;
      const int8_t* v213 = (const int8_t*) v212;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v214 = __riscv_vle8_v_i8m1(v213, v206);
      const uint8_t* v215 = v191 + 20;
      const uint8_t* v216 = v215 + v204;
      const int8_t* v217 = (const int8_t*) v216;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v218 = __riscv_vle8_v_i8m1(v217, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v219 = __riscv_vand_vx_u8m1(v210, 0x0F, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v220 = __riscv_vsrl_vx_u8m1(v210, 0x04, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v221 = __riscv_vid_v_u16m2(v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v222 = __riscv_vadd_vx_u16m2(v221, v204, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v223 = __riscv_vmv_v_x_u16m2(v199, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v224 = __riscv_vsrl_vv_u16m2(v223, v222, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v225 = __riscv_vand_vx_u16m2(v224, 0x1, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v226 = __riscv_vsll_vx_u16m2(v225, 0x4, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v227 = __riscv_vncvt_x_x_w_u8m1(v226, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v228 = __riscv_vid_v_u16m2(v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v229 = __riscv_vadd_vx_u16m2(v228, v204, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v230 = __riscv_vmv_v_x_u16m2(v201, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v231 = __riscv_vsrl_vv_u16m2(v230, v229, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v232 = __riscv_vand_vx_u16m2(v231, 0x1, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v233 = __riscv_vsll_vx_u16m2(v232, 0x4, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v234 = __riscv_vncvt_x_x_w_u8m1(v233, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v235 = __riscv_vor_vv_u8m1(v219, v227, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v236 = __riscv_vor_vv_u8m1(v220, v234, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v237 = __riscv_vreinterpret_v_u8m1_i8m1(v235);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v238 = __riscv_vreinterpret_v_u8m1_i8m1(v236);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v239 = __riscv_vwmul_vv_i16m2(v237, v214, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v240 = __riscv_vwmacc_vv_i16m2(v239, v238, v218, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v241 = v202;
      vint32m1_t v242 = __riscv_vmv_v_x_i32m1(v241, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v243 = __riscv_vwredsum_vs_i16m2_i32m1(v240, v242, v206);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v244 = __riscv_vmv_x_s_i32m1_i32(v243);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v202 = v244;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v245 = v25;
    float v246 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v246 + ((v15 * v16) * (float) v245 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v247 = v84;
    float v248 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v248 + ((v74 * v75) * (float) v247 + v77 * v79);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v249 = v143;
    float v250 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v250 + ((v133 * v134) * (float) v249 + v136 * v138);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v251 = v202;
    float v252 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v252 + ((v192 * v193) * (float) v251 + v195 * v197);
  }
  for (size_t v253 = v9; v253 < v7; v253 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v254 = v253 * 24;
    const uint8_t* v255 = v3 + v254;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v256 = v253 * 36;
    const uint8_t* v257 = v4 + v256;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v258 = (float)*(const _Float16 *)(v255);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v259 = (float)*(const _Float16 *)(v257);
    const uint8_t* v260 = v255 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v261 = (float)*(const _Float16 *)(v260);
    const uint8_t* v262 = v257 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v263 = (float)*(const _Float16 *)(v262);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v264 = v255 + 4;
    uint32_t v265 = (uint16_t)*(const uint16_t *)(v264);
    const uint8_t* v266 = v255 + 6;
    uint32_t v267 = (uint16_t)*(const uint16_t *)(v266);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v268;
    v268 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v269 = __riscv_vsetvl_e8m1(16);
    for (size_t v270 = 0; v270 < 16; v270 += v269) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v271 = 16 - v270;
      size_t v272 = __riscv_vsetvl_e8m1(v271);
      const uint8_t* v273 = v255 + 8;
      const uint8_t* v274 = v273 + v270;
      const uint8_t* v275 = (const uint8_t*) v274;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v276 = __riscv_vle8_v_u8m1(v275, v272);
      const uint8_t* v277 = v257 + 4;
      const uint8_t* v278 = v277 + v270;
      const int8_t* v279 = (const int8_t*) v278;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v280 = __riscv_vle8_v_i8m1(v279, v272);
      const uint8_t* v281 = v257 + 20;
      const uint8_t* v282 = v281 + v270;
      const int8_t* v283 = (const int8_t*) v282;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v284 = __riscv_vle8_v_i8m1(v283, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v285 = __riscv_vand_vx_u8m1(v276, 0x0F, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v286 = __riscv_vsrl_vx_u8m1(v276, 0x04, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v287 = __riscv_vid_v_u16m2(v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v288 = __riscv_vadd_vx_u16m2(v287, v270, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v289 = __riscv_vmv_v_x_u16m2(v265, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v290 = __riscv_vsrl_vv_u16m2(v289, v288, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v291 = __riscv_vand_vx_u16m2(v290, 0x1, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v292 = __riscv_vsll_vx_u16m2(v291, 0x4, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v293 = __riscv_vncvt_x_x_w_u8m1(v292, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v294 = __riscv_vid_v_u16m2(v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v295 = __riscv_vadd_vx_u16m2(v294, v270, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v296 = __riscv_vmv_v_x_u16m2(v267, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v297 = __riscv_vsrl_vv_u16m2(v296, v295, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v298 = __riscv_vand_vx_u16m2(v297, 0x1, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v299 = __riscv_vsll_vx_u16m2(v298, 0x4, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v300 = __riscv_vncvt_x_x_w_u8m1(v299, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v301 = __riscv_vor_vv_u8m1(v285, v293, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v302 = __riscv_vor_vv_u8m1(v286, v300, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v303 = __riscv_vreinterpret_v_u8m1_i8m1(v301);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v304 = __riscv_vreinterpret_v_u8m1_i8m1(v302);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v305 = __riscv_vwmul_vv_i16m2(v303, v280, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v306 = __riscv_vwmacc_vv_i16m2(v305, v304, v284, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v307 = v268;
      vint32m1_t v308 = __riscv_vmv_v_x_i32m1(v307, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v309 = __riscv_vwredsum_vs_i16m2_i32m1(v306, v308, v272);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v310 = __riscv_vmv_x_s_i32m1_i32(v309);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v268 = v310;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v311 = v268;
    float v312 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v312 + ((v258 * v259) * (float) v311 + v261 * v263);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v313 = v6;
  v2[0] = v313;
  return;
}


