#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_quantize_row_q8_0_kernel_quantize_row_q8_0(size_t v1, const float* v2, uint8_t* v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 32;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=x_block
    size_t v7 = v6 * 32;
    const float* v8 = v2 + v7;
    const float* v9 = (const float*) v8;
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=y_block
    size_t v10 = v6 * 34;
    uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
    vfloat32m8_t v12 = __riscv_vle32_v_f32m8(v9, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m8
    vfloat32m8_t v13 = __riscv_vfabs_v_f32m8(v12, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
    vfloat32m1_t v14 = __riscv_vfmv_v_f_f32m1(0.0f, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfredmax_vs_f32m8_f32m1
    vfloat32m1_t v15 = __riscv_vfredmax_vs_f32m8_f32m1(v13, v14, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f32m1_f32
    float v16 = __riscv_vfmv_f_s_f32m1_f32(v15);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=d
    float v17 = v16 / 127.0f;
    // tcrv_emitc.local_variable=id source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
    float v18;
    v18 = 0.0f;
    bool v19 = v17 != 0.0f;
    if (v19) {
      float v20 = 1.0f / v17;
      v18 = v20;
    }
    float v21 = v18;
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp16_d_store
    uint8_t* v22 = v11 + 0;
    _Float16* v23 = (_Float16*) v22;
    _Float16 v24 = (_Float16) v17;
    v23[0] = v24;
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
    vfloat32m8_t v25 = __riscv_vfmul_vf_f32m8(v12, v21, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfncvt_x_f_w_i16m4
    vint16m4_t v26 = __riscv_vfncvt_x_f_w_i16m4(v25, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_i8m2
    vint8m2_t v27 = __riscv_vncvt_x_x_w_i8m2(v26, 32);
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=qs_store
    uint8_t* v28 = v11 + 2;
    int8_t* v29 = (int8_t*) v28;
    // tcrv_emitc.source_op=tcrv_rvv.quantize_row_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v29, v27, 32);
  }
  return;
}


