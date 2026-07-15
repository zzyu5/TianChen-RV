#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_quantize_row_q8_1_kernel_quantize_row_q8_1(size_t v1, const float* v2, uint8_t* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=x_block
    size_t v7 = v6 * 32;
    const float* v8 = v2 + v7;
    const float* v9 = (const float*) v8;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=y_block
    size_t v10 = v6 * 36;
    uint8_t* v11 = v3 + v10;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v12 = __riscv_vle32_v_f32m8(v9, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m8
    vfloat32m8_t v13 = __riscv_vfabs_v_f32m8(v12, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
    vfloat32m1_t v14 = __riscv_vfmv_v_f_f32m1(0.0f, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfredmax_vs_f32m8_f32m1
    vfloat32m1_t v15 = __riscv_vfredmax_vs_f32m8_f32m1(v13, v14, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f32m1_f32
    float v16 = __riscv_vfmv_f_s_f32m1_f32(v15);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=d
    float v17 = v16 / 127.0f;
    // weft_emitc.local_variable=id source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    float v18;
    v18 = 0.0f;
    bool v19 = v17 != 0.0f;
    if (v19) {
      float v20 = 1.0f / v17;
      v18 = v20;
    }
    float v21 = v18;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp16_d_store
    uint8_t* v22 = v11 + 0;
    _Float16* v23 = (_Float16*) v22;
    _Float16 v24 = (_Float16) v17;
    v23[0] = v24;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v25 = __riscv_vfmul_vf_f32m8(v12, v21, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfncvt_x_f_w_i16m4
    vint16m4_t v26 = __riscv_vfncvt_x_f_w_i16m4(v25, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_i8m2
    vint8m2_t v27 = __riscv_vncvt_x_x_w_i8m2(v26, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qs_store
    uint8_t* v28 = v11 + 4;
    int8_t* v29 = (int8_t*) v28;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v29, v27, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_sum
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
    vint16m1_t v30 = __riscv_vmv_v_x_i16m1(0, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m2_i16m1
    vint16m1_t v31 = __riscv_vwredsum_vs_i8m2_i16m1(v27, v30, 32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
    int v32 = __riscv_vmv_x_s_i16m1_i16(v31);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fp16_s_store
    float v33 = v32 * v17;
    uint8_t* v34 = v11 + 2;
    _Float16* v35 = (_Float16*) v34;
    _Float16 v36 = (_Float16) v33;
    v35[0] = v36;
  }
  return;
}


