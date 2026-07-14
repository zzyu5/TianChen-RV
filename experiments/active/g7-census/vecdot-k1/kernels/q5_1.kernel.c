#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, const int32_t* v5) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
  size_t v6 = __riscv_vsetvl_e8m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v7;
  v7 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v8 = v1 / 32;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_x
    size_t v10 = v9 * 24;
    const uint8_t* v11 = v3 + v10;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_base_y
    size_t v12 = v9 * 36;
    const uint8_t* v13 = v4 + v12;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_field
    const uint8_t* v14 = v11 + 4;
    uint32_t v15 = (uint16_t)*(const uint16_t *)(v14);
    const uint8_t* v16 = v11 + 6;
    uint32_t v17 = (uint16_t)*(const uint16_t *)(v16);
    // weft_emitc.local_variable=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int32_t v18;
    v18 = 0;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v19 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v20 = v11 + 8;
    const uint8_t* v21 = v20 + 0;
    const uint8_t* v22 = (const uint8_t*) v21;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v23 = __riscv_vle8_v_u8m1(v22, v19);
    const uint8_t* v24 = v13 + 4;
    const uint8_t* v25 = v24 + 0;
    const int8_t* v26 = (const int8_t*) v25;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v27 = __riscv_vle8_v_i8m1(v26, v19);
    const uint8_t* v28 = v13 + 20;
    const uint8_t* v29 = v28 + 0;
    const int8_t* v30 = (const int8_t*) v29;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v31 = __riscv_vle8_v_i8m1(v30, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
    vuint8m1_t v32 = __riscv_vand_vx_u8m1(v23, 0x0F, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
    vuint8m1_t v33 = __riscv_vsrl_vx_u8m1(v23, 0x04, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v34 = __riscv_vid_v_u16m2(v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v35 = __riscv_vadd_vx_u16m2(v34, 0, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v36 = __riscv_vmv_v_x_u16m2(v15, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v37 = __riscv_vsrl_vv_u16m2(v36, v35, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v38 = __riscv_vand_vx_u16m2(v37, 0x1, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v39 = __riscv_vsll_vx_u16m2(v38, 0x4, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v40 = __riscv_vncvt_x_x_w_u8m1(v39, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
    vuint16m2_t v41 = __riscv_vid_v_u16m2(v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
    vuint16m2_t v42 = __riscv_vadd_vx_u16m2(v41, 0, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
    vuint16m2_t v43 = __riscv_vmv_v_x_u16m2(v17, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
    vuint16m2_t v44 = __riscv_vsrl_vv_u16m2(v43, v42, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
    vuint16m2_t v45 = __riscv_vand_vx_u16m2(v44, 0x1, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
    vuint16m2_t v46 = __riscv_vsll_vx_u16m2(v45, 0x4, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v47 = __riscv_vncvt_x_x_w_u8m1(v46, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v48 = __riscv_vor_vv_u8m1(v32, v40, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
    vuint8m1_t v49 = __riscv_vor_vv_u8m1(v33, v47, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v50 = __riscv_vreinterpret_v_u8m1_i8m1(v48);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v51 = __riscv_vreinterpret_v_u8m1_i8m1(v49);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v52 = __riscv_vwmul_vv_i16m2(v50, v27, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
    vint16m2_t v53 = __riscv_vwmacc_vv_i16m2(v52, v51, v31, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v54 = __riscv_vmv_v_x_i32m1(0, 1);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v55 = __riscv_vwredsum_vs_i16m2_i32m1(v53, v54, v19);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v56 = __riscv_vmv_x_s_i32m1_i32(v55);
    // weft_emitc.assign target=sumi source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v18 = v56;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v57 = (float)*(const _Float16 *)(v11);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v58 = (float)*(const _Float16 *)(v13);
    const uint8_t* v59 = v11 + 2;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v60 = (float)*(const _Float16 *)(v59);
    const uint8_t* v61 = v13 + 2;
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v62 = (float)*(const _Float16 *)(v61);
    // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v63 = v18;
    float v64 = v7;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v7 = v64 + ((v57 * v58) * (float) v63 + v60 * v62);
  }
  // weft_emitc.source_op=weft_rvv.typed_flat_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  float v65 = v7;
  v2[0] = v65;
  return;
}


