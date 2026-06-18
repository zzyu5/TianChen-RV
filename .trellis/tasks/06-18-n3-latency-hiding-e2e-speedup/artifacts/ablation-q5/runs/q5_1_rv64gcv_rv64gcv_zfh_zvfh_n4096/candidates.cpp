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
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f1_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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
    const uint8_t* v25 = v10 + 8;
    const uint8_t* v26 = v25 + 0;
    const uint8_t* v27 = (const uint8_t*) v26;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v28 = __riscv_vle8_v_u8m1(v27, v24);
    const uint8_t* v29 = v12 + 4;
    const uint8_t* v30 = v29 + 0;
    const int8_t* v31 = (const int8_t*) v30;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v32 = __riscv_vle8_v_i8m1(v31, v24);
    const uint8_t* v33 = v12 + 20;
    const uint8_t* v34 = v33 + 0;
    const int8_t* v35 = (const int8_t*) v34;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v36 = __riscv_vle8_v_i8m1(v35, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v37 = __riscv_vand_vx_u8m1(v28, 0x0F, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v38 = __riscv_vsrl_vx_u8m1(v28, 0x04, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v39 = __riscv_vid_v_u16m2(v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v40 = __riscv_vadd_vx_u16m2(v39, 0, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v41 = __riscv_vmv_v_x_u16m2(v20, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v42 = __riscv_vsrl_vv_u16m2(v41, v40, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v43 = __riscv_vand_vx_u16m2(v42, 0x1, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v44 = __riscv_vsll_vx_u16m2(v43, 0x4, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v45 = __riscv_vncvt_x_x_w_u8m1(v44, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v46 = __riscv_vid_v_u16m2(v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v47 = __riscv_vadd_vx_u16m2(v46, 0, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v48 = __riscv_vmv_v_x_u16m2(v22, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v49 = __riscv_vsrl_vv_u16m2(v48, v47, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v50 = __riscv_vand_vx_u16m2(v49, 0x1, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v51 = __riscv_vsll_vx_u16m2(v50, 0x4, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v52 = __riscv_vncvt_x_x_w_u8m1(v51, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v53 = __riscv_vor_vv_u8m1(v37, v45, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v54 = __riscv_vor_vv_u8m1(v38, v52, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v55 = __riscv_vreinterpret_v_u8m1_i8m1(v53);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v56 = __riscv_vreinterpret_v_u8m1_i8m1(v54);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v57 = __riscv_vwmul_vv_i16m2(v55, v32, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v58 = __riscv_vwmacc_vv_i16m2(v57, v56, v36, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v59 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v60 = __riscv_vwredsum_vs_i16m2_i32m1(v58, v59, v24);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v61 = __riscv_vmv_x_s_i32m1_i32(v60);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v23 = v61;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v62 = v23;
    float v63 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v63 + ((v13 * v14) * (float) v62 + v16 * v18);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v64 = v6;
  v2[0] = v64;
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
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f2_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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
    const uint8_t* v27 = v12 + 8;
    const uint8_t* v28 = v27 + 0;
    const uint8_t* v29 = (const uint8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v30 = __riscv_vle8_v_u8m1(v29, v26);
    const uint8_t* v31 = v14 + 4;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v26);
    const uint8_t* v35 = v14 + 20;
    const uint8_t* v36 = v35 + 0;
    const int8_t* v37 = (const int8_t*) v36;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v38 = __riscv_vle8_v_i8m1(v37, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v39 = __riscv_vand_vx_u8m1(v30, 0x0F, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v40 = __riscv_vsrl_vx_u8m1(v30, 0x04, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v41 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v42 = __riscv_vadd_vx_u16m2(v41, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v43 = __riscv_vmv_v_x_u16m2(v22, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v44 = __riscv_vsrl_vv_u16m2(v43, v42, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v45 = __riscv_vand_vx_u16m2(v44, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v46 = __riscv_vsll_vx_u16m2(v45, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v47 = __riscv_vncvt_x_x_w_u8m1(v46, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v48 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v49 = __riscv_vadd_vx_u16m2(v48, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v50 = __riscv_vmv_v_x_u16m2(v24, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v51 = __riscv_vsrl_vv_u16m2(v50, v49, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v52 = __riscv_vand_vx_u16m2(v51, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v53 = __riscv_vsll_vx_u16m2(v52, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v54 = __riscv_vncvt_x_x_w_u8m1(v53, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v47, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v56 = __riscv_vor_vv_u8m1(v40, v54, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v57 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v56);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v59 = __riscv_vwmul_vv_i16m2(v57, v34, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v60 = __riscv_vwmacc_vv_i16m2(v59, v58, v38, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v61 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v62 = __riscv_vwredsum_vs_i16m2_i32m1(v60, v61, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v63 = __riscv_vmv_x_s_i32m1_i32(v62);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v63;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v64 = v10 + 1;
    size_t v65 = v64 * 24;
    const uint8_t* v66 = v3 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v67 = v10 + 1;
    size_t v68 = v67 * 36;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    const uint8_t* v72 = v66 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v72);
    const uint8_t* v74 = v69 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v74);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v76 = v66 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    const uint8_t* v78 = v66 + 6;
    uint32_t v79 = (uint16_t)*(const uint16_t *)(v78);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v80;
    v80 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v81 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v82 = v66 + 8;
    const uint8_t* v83 = v82 + 0;
    const uint8_t* v84 = (const uint8_t*) v83;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v85 = __riscv_vle8_v_u8m1(v84, v81);
    const uint8_t* v86 = v69 + 4;
    const uint8_t* v87 = v86 + 0;
    const int8_t* v88 = (const int8_t*) v87;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v89 = __riscv_vle8_v_i8m1(v88, v81);
    const uint8_t* v90 = v69 + 20;
    const uint8_t* v91 = v90 + 0;
    const int8_t* v92 = (const int8_t*) v91;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v92, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v94 = __riscv_vand_vx_u8m1(v85, 0x0F, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v95 = __riscv_vsrl_vx_u8m1(v85, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v96 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v97 = __riscv_vadd_vx_u16m2(v96, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v98 = __riscv_vmv_v_x_u16m2(v77, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v99 = __riscv_vsrl_vv_u16m2(v98, v97, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v100 = __riscv_vand_vx_u16m2(v99, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v101 = __riscv_vsll_vx_u16m2(v100, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v102 = __riscv_vncvt_x_x_w_u8m1(v101, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v103 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v104 = __riscv_vadd_vx_u16m2(v103, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v105 = __riscv_vmv_v_x_u16m2(v79, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v106 = __riscv_vsrl_vv_u16m2(v105, v104, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v107 = __riscv_vand_vx_u16m2(v106, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v108 = __riscv_vsll_vx_u16m2(v107, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v109 = __riscv_vncvt_x_x_w_u8m1(v108, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v110 = __riscv_vor_vv_u8m1(v94, v102, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v109, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v112 = __riscv_vreinterpret_v_u8m1_i8m1(v110);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v114 = __riscv_vwmul_vv_i16m2(v112, v89, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v115 = __riscv_vwmacc_vv_i16m2(v114, v113, v93, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v116 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v117 = __riscv_vwredsum_vs_i16m2_i32m1(v115, v116, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v118 = __riscv_vmv_x_s_i32m1_i32(v117);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v80 = v118;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v119 = v25;
    float v120 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v120 + ((v15 * v16) * (float) v119 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v121 = v80;
    float v122 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v122 + ((v70 * v71) * (float) v121 + v73 * v75);
  }
  for (size_t v123 = v9; v123 < v7; v123 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v124 = v123 * 24;
    const uint8_t* v125 = v3 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v126 = v123 * 36;
    const uint8_t* v127 = v4 + v126;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v128 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v129 = (float)*(const _Float16 *)(v127);
    const uint8_t* v130 = v125 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v131 = (float)*(const _Float16 *)(v130);
    const uint8_t* v132 = v127 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v133 = (float)*(const _Float16 *)(v132);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v134 = v125 + 4;
    uint32_t v135 = (uint16_t)*(const uint16_t *)(v134);
    const uint8_t* v136 = v125 + 6;
    uint32_t v137 = (uint16_t)*(const uint16_t *)(v136);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v138;
    v138 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v139 = __riscv_vsetvl_e8m1(16);
    for (size_t v140 = 0; v140 < 16; v140 += v139) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v141 = 16 - v140;
      size_t v142 = __riscv_vsetvl_e8m1(v141);
      const uint8_t* v143 = v125 + 8;
      const uint8_t* v144 = v143 + v140;
      const uint8_t* v145 = (const uint8_t*) v144;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v146 = __riscv_vle8_v_u8m1(v145, v142);
      const uint8_t* v147 = v127 + 4;
      const uint8_t* v148 = v147 + v140;
      const int8_t* v149 = (const int8_t*) v148;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v150 = __riscv_vle8_v_i8m1(v149, v142);
      const uint8_t* v151 = v127 + 20;
      const uint8_t* v152 = v151 + v140;
      const int8_t* v153 = (const int8_t*) v152;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v154 = __riscv_vle8_v_i8m1(v153, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v155 = __riscv_vand_vx_u8m1(v146, 0x0F, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v156 = __riscv_vsrl_vx_u8m1(v146, 0x04, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v157 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v158 = __riscv_vadd_vx_u16m2(v157, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v159 = __riscv_vmv_v_x_u16m2(v135, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v160 = __riscv_vsrl_vv_u16m2(v159, v158, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v161 = __riscv_vand_vx_u16m2(v160, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v162 = __riscv_vsll_vx_u16m2(v161, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v163 = __riscv_vncvt_x_x_w_u8m1(v162, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v164 = __riscv_vid_v_u16m2(v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v165 = __riscv_vadd_vx_u16m2(v164, v140, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v166 = __riscv_vmv_v_x_u16m2(v137, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v167 = __riscv_vsrl_vv_u16m2(v166, v165, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v168 = __riscv_vand_vx_u16m2(v167, 0x1, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v169 = __riscv_vsll_vx_u16m2(v168, 0x4, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v170 = __riscv_vncvt_x_x_w_u8m1(v169, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v171 = __riscv_vor_vv_u8m1(v155, v163, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v172 = __riscv_vor_vv_u8m1(v156, v170, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v173 = __riscv_vreinterpret_v_u8m1_i8m1(v171);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v174 = __riscv_vreinterpret_v_u8m1_i8m1(v172);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v175 = __riscv_vwmul_vv_i16m2(v173, v150, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v176 = __riscv_vwmacc_vv_i16m2(v175, v174, v154, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v177 = v138;
      vint32m1_t v178 = __riscv_vmv_v_x_i32m1(v177, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v179 = __riscv_vwredsum_vs_i16m2_i32m1(v176, v178, v142);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v180 = __riscv_vmv_x_s_i32m1_i32(v179);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v138 = v180;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v181 = v138;
    float v182 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v182 + ((v128 * v129) * (float) v181 + v131 * v133);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v183 = v6;
  v2[0] = v183;
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



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1_m1_f4_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
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
    const uint8_t* v27 = v12 + 8;
    const uint8_t* v28 = v27 + 0;
    const uint8_t* v29 = (const uint8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v30 = __riscv_vle8_v_u8m1(v29, v26);
    const uint8_t* v31 = v14 + 4;
    const uint8_t* v32 = v31 + 0;
    const int8_t* v33 = (const int8_t*) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v34 = __riscv_vle8_v_i8m1(v33, v26);
    const uint8_t* v35 = v14 + 20;
    const uint8_t* v36 = v35 + 0;
    const int8_t* v37 = (const int8_t*) v36;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v38 = __riscv_vle8_v_i8m1(v37, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v39 = __riscv_vand_vx_u8m1(v30, 0x0F, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v40 = __riscv_vsrl_vx_u8m1(v30, 0x04, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v41 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v42 = __riscv_vadd_vx_u16m2(v41, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v43 = __riscv_vmv_v_x_u16m2(v22, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v44 = __riscv_vsrl_vv_u16m2(v43, v42, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v45 = __riscv_vand_vx_u16m2(v44, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v46 = __riscv_vsll_vx_u16m2(v45, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v47 = __riscv_vncvt_x_x_w_u8m1(v46, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v48 = __riscv_vid_v_u16m2(v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v49 = __riscv_vadd_vx_u16m2(v48, 0, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v50 = __riscv_vmv_v_x_u16m2(v24, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v51 = __riscv_vsrl_vv_u16m2(v50, v49, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v52 = __riscv_vand_vx_u16m2(v51, 0x1, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v53 = __riscv_vsll_vx_u16m2(v52, 0x4, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v54 = __riscv_vncvt_x_x_w_u8m1(v53, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v55 = __riscv_vor_vv_u8m1(v39, v47, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v56 = __riscv_vor_vv_u8m1(v40, v54, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v57 = __riscv_vreinterpret_v_u8m1_i8m1(v55);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v58 = __riscv_vreinterpret_v_u8m1_i8m1(v56);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v59 = __riscv_vwmul_vv_i16m2(v57, v34, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v60 = __riscv_vwmacc_vv_i16m2(v59, v58, v38, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v61 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v62 = __riscv_vwredsum_vs_i16m2_i32m1(v60, v61, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v63 = __riscv_vmv_x_s_i32m1_i32(v62);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v25 = v63;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v64 = v10 + 1;
    size_t v65 = v64 * 24;
    const uint8_t* v66 = v3 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v67 = v10 + 1;
    size_t v68 = v67 * 36;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    const uint8_t* v72 = v66 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v73 = (float)*(const _Float16 *)(v72);
    const uint8_t* v74 = v69 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v74);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v76 = v66 + 4;
    uint32_t v77 = (uint16_t)*(const uint16_t *)(v76);
    const uint8_t* v78 = v66 + 6;
    uint32_t v79 = (uint16_t)*(const uint16_t *)(v78);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v80;
    v80 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v81 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v82 = v66 + 8;
    const uint8_t* v83 = v82 + 0;
    const uint8_t* v84 = (const uint8_t*) v83;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v85 = __riscv_vle8_v_u8m1(v84, v81);
    const uint8_t* v86 = v69 + 4;
    const uint8_t* v87 = v86 + 0;
    const int8_t* v88 = (const int8_t*) v87;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v89 = __riscv_vle8_v_i8m1(v88, v81);
    const uint8_t* v90 = v69 + 20;
    const uint8_t* v91 = v90 + 0;
    const int8_t* v92 = (const int8_t*) v91;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v93 = __riscv_vle8_v_i8m1(v92, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v94 = __riscv_vand_vx_u8m1(v85, 0x0F, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v95 = __riscv_vsrl_vx_u8m1(v85, 0x04, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v96 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v97 = __riscv_vadd_vx_u16m2(v96, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v98 = __riscv_vmv_v_x_u16m2(v77, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v99 = __riscv_vsrl_vv_u16m2(v98, v97, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v100 = __riscv_vand_vx_u16m2(v99, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v101 = __riscv_vsll_vx_u16m2(v100, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v102 = __riscv_vncvt_x_x_w_u8m1(v101, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v103 = __riscv_vid_v_u16m2(v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v104 = __riscv_vadd_vx_u16m2(v103, 0, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v105 = __riscv_vmv_v_x_u16m2(v79, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v106 = __riscv_vsrl_vv_u16m2(v105, v104, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v107 = __riscv_vand_vx_u16m2(v106, 0x1, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v108 = __riscv_vsll_vx_u16m2(v107, 0x4, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v109 = __riscv_vncvt_x_x_w_u8m1(v108, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v110 = __riscv_vor_vv_u8m1(v94, v102, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v111 = __riscv_vor_vv_u8m1(v95, v109, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v112 = __riscv_vreinterpret_v_u8m1_i8m1(v110);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v113 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v114 = __riscv_vwmul_vv_i16m2(v112, v89, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v115 = __riscv_vwmacc_vv_i16m2(v114, v113, v93, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v116 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v117 = __riscv_vwredsum_vs_i16m2_i32m1(v115, v116, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v118 = __riscv_vmv_x_s_i32m1_i32(v117);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v80 = v118;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v119 = v10 + 2;
    size_t v120 = v119 * 24;
    const uint8_t* v121 = v3 + v120;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v122 = v10 + 2;
    size_t v123 = v122 * 36;
    const uint8_t* v124 = v4 + v123;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v125 = (float)*(const _Float16 *)(v121);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v126 = (float)*(const _Float16 *)(v124);
    const uint8_t* v127 = v121 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v128 = (float)*(const _Float16 *)(v127);
    const uint8_t* v129 = v124 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v130 = (float)*(const _Float16 *)(v129);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v131 = v121 + 4;
    uint32_t v132 = (uint16_t)*(const uint16_t *)(v131);
    const uint8_t* v133 = v121 + 6;
    uint32_t v134 = (uint16_t)*(const uint16_t *)(v133);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v135;
    v135 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v136 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v137 = v121 + 8;
    const uint8_t* v138 = v137 + 0;
    const uint8_t* v139 = (const uint8_t*) v138;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v140 = __riscv_vle8_v_u8m1(v139, v136);
    const uint8_t* v141 = v124 + 4;
    const uint8_t* v142 = v141 + 0;
    const int8_t* v143 = (const int8_t*) v142;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v144 = __riscv_vle8_v_i8m1(v143, v136);
    const uint8_t* v145 = v124 + 20;
    const uint8_t* v146 = v145 + 0;
    const int8_t* v147 = (const int8_t*) v146;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v148 = __riscv_vle8_v_i8m1(v147, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v149 = __riscv_vand_vx_u8m1(v140, 0x0F, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v150 = __riscv_vsrl_vx_u8m1(v140, 0x04, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v151 = __riscv_vid_v_u16m2(v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v152 = __riscv_vadd_vx_u16m2(v151, 0, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v153 = __riscv_vmv_v_x_u16m2(v132, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v154 = __riscv_vsrl_vv_u16m2(v153, v152, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v155 = __riscv_vand_vx_u16m2(v154, 0x1, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v156 = __riscv_vsll_vx_u16m2(v155, 0x4, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v157 = __riscv_vncvt_x_x_w_u8m1(v156, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v158 = __riscv_vid_v_u16m2(v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v159 = __riscv_vadd_vx_u16m2(v158, 0, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v160 = __riscv_vmv_v_x_u16m2(v134, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v161 = __riscv_vsrl_vv_u16m2(v160, v159, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v162 = __riscv_vand_vx_u16m2(v161, 0x1, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v163 = __riscv_vsll_vx_u16m2(v162, 0x4, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v164 = __riscv_vncvt_x_x_w_u8m1(v163, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v165 = __riscv_vor_vv_u8m1(v149, v157, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v166 = __riscv_vor_vv_u8m1(v150, v164, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v167 = __riscv_vreinterpret_v_u8m1_i8m1(v165);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v168 = __riscv_vreinterpret_v_u8m1_i8m1(v166);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v169 = __riscv_vwmul_vv_i16m2(v167, v144, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v170 = __riscv_vwmacc_vv_i16m2(v169, v168, v148, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v171 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v172 = __riscv_vwredsum_vs_i16m2_i32m1(v170, v171, v136);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v173 = __riscv_vmv_x_s_i32m1_i32(v172);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v135 = v173;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v174 = v10 + 3;
    size_t v175 = v174 * 24;
    const uint8_t* v176 = v3 + v175;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v177 = v10 + 3;
    size_t v178 = v177 * 36;
    const uint8_t* v179 = v4 + v178;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v180 = (float)*(const _Float16 *)(v176);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v181 = (float)*(const _Float16 *)(v179);
    const uint8_t* v182 = v176 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v183 = (float)*(const _Float16 *)(v182);
    const uint8_t* v184 = v179 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v185 = (float)*(const _Float16 *)(v184);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v186 = v176 + 4;
    uint32_t v187 = (uint16_t)*(const uint16_t *)(v186);
    const uint8_t* v188 = v176 + 6;
    uint32_t v189 = (uint16_t)*(const uint16_t *)(v188);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v190;
    v190 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v191 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v192 = v176 + 8;
    const uint8_t* v193 = v192 + 0;
    const uint8_t* v194 = (const uint8_t*) v193;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v195 = __riscv_vle8_v_u8m1(v194, v191);
    const uint8_t* v196 = v179 + 4;
    const uint8_t* v197 = v196 + 0;
    const int8_t* v198 = (const int8_t*) v197;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v199 = __riscv_vle8_v_i8m1(v198, v191);
    const uint8_t* v200 = v179 + 20;
    const uint8_t* v201 = v200 + 0;
    const int8_t* v202 = (const int8_t*) v201;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v203 = __riscv_vle8_v_i8m1(v202, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v204 = __riscv_vand_vx_u8m1(v195, 0x0F, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v205 = __riscv_vsrl_vx_u8m1(v195, 0x04, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v206 = __riscv_vid_v_u16m2(v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v207 = __riscv_vadd_vx_u16m2(v206, 0, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v208 = __riscv_vmv_v_x_u16m2(v187, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v209 = __riscv_vsrl_vv_u16m2(v208, v207, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v210 = __riscv_vand_vx_u16m2(v209, 0x1, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v211 = __riscv_vsll_vx_u16m2(v210, 0x4, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v212 = __riscv_vncvt_x_x_w_u8m1(v211, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v213 = __riscv_vid_v_u16m2(v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v214 = __riscv_vadd_vx_u16m2(v213, 0, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v215 = __riscv_vmv_v_x_u16m2(v189, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v216 = __riscv_vsrl_vv_u16m2(v215, v214, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v217 = __riscv_vand_vx_u16m2(v216, 0x1, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v218 = __riscv_vsll_vx_u16m2(v217, 0x4, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v219 = __riscv_vncvt_x_x_w_u8m1(v218, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v220 = __riscv_vor_vv_u8m1(v204, v212, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v221 = __riscv_vor_vv_u8m1(v205, v219, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v222 = __riscv_vreinterpret_v_u8m1_i8m1(v220);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v223 = __riscv_vreinterpret_v_u8m1_i8m1(v221);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v224 = __riscv_vwmul_vv_i16m2(v222, v199, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v225 = __riscv_vwmacc_vv_i16m2(v224, v223, v203, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v226 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v227 = __riscv_vwredsum_vs_i16m2_i32m1(v225, v226, v191);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v228 = __riscv_vmv_x_s_i32m1_i32(v227);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v190 = v228;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v229 = v25;
    float v230 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v230 + ((v15 * v16) * (float) v229 + v18 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v231 = v80;
    float v232 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v232 + ((v70 * v71) * (float) v231 + v73 * v75);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v233 = v135;
    float v234 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v234 + ((v125 * v126) * (float) v233 + v128 * v130);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v235 = v190;
    float v236 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v236 + ((v180 * v181) * (float) v235 + v183 * v185);
  }
  for (size_t v237 = v9; v237 < v7; v237 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v238 = v237 * 24;
    const uint8_t* v239 = v3 + v238;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v240 = v237 * 36;
    const uint8_t* v241 = v4 + v240;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v242 = (float)*(const _Float16 *)(v239);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v243 = (float)*(const _Float16 *)(v241);
    const uint8_t* v244 = v239 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v245 = (float)*(const _Float16 *)(v244);
    const uint8_t* v246 = v241 + 2;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v247 = (float)*(const _Float16 *)(v246);
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v248 = v239 + 4;
    uint32_t v249 = (uint16_t)*(const uint16_t *)(v248);
    const uint8_t* v250 = v239 + 6;
    uint32_t v251 = (uint16_t)*(const uint16_t *)(v250);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v252;
    v252 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v253 = __riscv_vsetvl_e8m1(16);
    for (size_t v254 = 0; v254 < 16; v254 += v253) {
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v255 = 16 - v254;
      size_t v256 = __riscv_vsetvl_e8m1(v255);
      const uint8_t* v257 = v239 + 8;
      const uint8_t* v258 = v257 + v254;
      const uint8_t* v259 = (const uint8_t*) v258;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
      vuint8m1_t v260 = __riscv_vle8_v_u8m1(v259, v256);
      const uint8_t* v261 = v241 + 4;
      const uint8_t* v262 = v261 + v254;
      const int8_t* v263 = (const int8_t*) v262;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v264 = __riscv_vle8_v_i8m1(v263, v256);
      const uint8_t* v265 = v241 + 20;
      const uint8_t* v266 = v265 + v254;
      const int8_t* v267 = (const int8_t*) v266;
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v268 = __riscv_vle8_v_i8m1(v267, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
      vuint8m1_t v269 = __riscv_vand_vx_u8m1(v260, 0x0F, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
      vuint8m1_t v270 = __riscv_vsrl_vx_u8m1(v260, 0x04, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v271 = __riscv_vid_v_u16m2(v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v272 = __riscv_vadd_vx_u16m2(v271, v254, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v273 = __riscv_vmv_v_x_u16m2(v249, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v274 = __riscv_vsrl_vv_u16m2(v273, v272, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v275 = __riscv_vand_vx_u16m2(v274, 0x1, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v276 = __riscv_vsll_vx_u16m2(v275, 0x4, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v277 = __riscv_vncvt_x_x_w_u8m1(v276, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
      vuint16m2_t v278 = __riscv_vid_v_u16m2(v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
      vuint16m2_t v279 = __riscv_vadd_vx_u16m2(v278, v254, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
      vuint16m2_t v280 = __riscv_vmv_v_x_u16m2(v251, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
      vuint16m2_t v281 = __riscv_vsrl_vv_u16m2(v280, v279, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
      vuint16m2_t v282 = __riscv_vand_vx_u16m2(v281, 0x1, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
      vuint16m2_t v283 = __riscv_vsll_vx_u16m2(v282, 0x4, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
      vuint8m1_t v284 = __riscv_vncvt_x_x_w_u8m1(v283, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v285 = __riscv_vor_vv_u8m1(v269, v277, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
      vuint8m1_t v286 = __riscv_vor_vv_u8m1(v270, v284, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v287 = __riscv_vreinterpret_v_u8m1_i8m1(v285);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
      vint8m1_t v288 = __riscv_vreinterpret_v_u8m1_i8m1(v286);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v289 = __riscv_vwmul_vv_i16m2(v287, v264, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
      vint16m2_t v290 = __riscv_vwmacc_vv_i16m2(v289, v288, v268, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v291 = v252;
      vint32m1_t v292 = __riscv_vmv_v_x_i32m1(v291, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v293 = __riscv_vwredsum_vs_i16m2_i32m1(v290, v292, v256);
      // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v294 = __riscv_vmv_x_s_i32m1_i32(v293);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v252 = v294;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v295 = v252;
    float v296 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v296 + ((v242 * v243) * (float) v295 + v245 * v247);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q5_1_q8_1_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v297 = v6;
  v2[0] = v297;
  return;
}


