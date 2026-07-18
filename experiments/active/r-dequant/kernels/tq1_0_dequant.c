#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_tq1_0_kernel_dequant_tq1_0(size_t v1, const uint8_t* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v5 = v1 / 256;
  for (size_t v6 = 0; v6 < v5; v6 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=xb
    size_t v7 = v6 * 54;
    const uint8_t* v8 = v2 + v7;
    size_t v9 = v6 * 256;
    float* v10 = v3 + v9;
    float* v11 = (float*) v10;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=d_scale
    const uint8_t* v12 = v8 + 52;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v12);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=tq1_0_decode
    const uint8_t* v14 = v8 + 0;
    const uint8_t* v15 = (const uint8_t*) v14;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v16 = __riscv_vle8_v_u8m1(v15, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v17 = __riscv_vmul_vx_u8m1(v16, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v18 = __riscv_vzext_vf2_u16m2(v17, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v19 = __riscv_vmul_vx_u16m2(v18, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v20 = __riscv_vsrl_vx_u16m2(v19, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v21 = __riscv_vreinterpret_v_u16m2_i16m2(v20);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v22 = __riscv_vsext_vf2_i32m4(v21, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v23 = __riscv_vsub_vx_i32m4(v22, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v24 = __riscv_vfcvt_f_x_v_f32m4(v23, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v25 = __riscv_vfmul_vf_f32m4(v24, v13, 16);
    float* v26 = v11 + 0;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v26, v25, 16);
    const uint8_t* v27 = v8 + 16;
    const uint8_t* v28 = (const uint8_t*) v27;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v29 = __riscv_vle8_v_u8m1(v28, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v30 = __riscv_vmul_vx_u8m1(v29, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v31 = __riscv_vzext_vf2_u16m2(v30, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v32 = __riscv_vmul_vx_u16m2(v31, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v33 = __riscv_vsrl_vx_u16m2(v32, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v34 = __riscv_vreinterpret_v_u16m2_i16m2(v33);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v35 = __riscv_vsext_vf2_i32m4(v34, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v36 = __riscv_vsub_vx_i32m4(v35, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v37 = __riscv_vfcvt_f_x_v_f32m4(v36, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v38 = __riscv_vfmul_vf_f32m4(v37, v13, 16);
    float* v39 = v11 + 16;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v39, v38, 16);
    const uint8_t* v40 = v8 + 0;
    const uint8_t* v41 = (const uint8_t*) v40;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v42 = __riscv_vle8_v_u8m1(v41, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v43 = __riscv_vmul_vx_u8m1(v42, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v44 = __riscv_vzext_vf2_u16m2(v43, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v45 = __riscv_vmul_vx_u16m2(v44, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v46 = __riscv_vsrl_vx_u16m2(v45, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v47 = __riscv_vreinterpret_v_u16m2_i16m2(v46);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v48 = __riscv_vsext_vf2_i32m4(v47, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v49 = __riscv_vsub_vx_i32m4(v48, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v50 = __riscv_vfcvt_f_x_v_f32m4(v49, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v51 = __riscv_vfmul_vf_f32m4(v50, v13, 16);
    float* v52 = v11 + 32;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v52, v51, 16);
    const uint8_t* v53 = v8 + 16;
    const uint8_t* v54 = (const uint8_t*) v53;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v55 = __riscv_vle8_v_u8m1(v54, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v56 = __riscv_vmul_vx_u8m1(v55, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v57 = __riscv_vzext_vf2_u16m2(v56, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v58 = __riscv_vmul_vx_u16m2(v57, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v59 = __riscv_vsrl_vx_u16m2(v58, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v60 = __riscv_vreinterpret_v_u16m2_i16m2(v59);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v61 = __riscv_vsext_vf2_i32m4(v60, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v62 = __riscv_vsub_vx_i32m4(v61, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v63 = __riscv_vfcvt_f_x_v_f32m4(v62, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v64 = __riscv_vfmul_vf_f32m4(v63, v13, 16);
    float* v65 = v11 + 48;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v65, v64, 16);
    const uint8_t* v66 = v8 + 0;
    const uint8_t* v67 = (const uint8_t*) v66;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v68 = __riscv_vle8_v_u8m1(v67, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v69 = __riscv_vmul_vx_u8m1(v68, 9, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v70 = __riscv_vzext_vf2_u16m2(v69, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v71 = __riscv_vmul_vx_u16m2(v70, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v72 = __riscv_vsrl_vx_u16m2(v71, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v73 = __riscv_vreinterpret_v_u16m2_i16m2(v72);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v74 = __riscv_vsext_vf2_i32m4(v73, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v75 = __riscv_vsub_vx_i32m4(v74, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v76 = __riscv_vfcvt_f_x_v_f32m4(v75, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v77 = __riscv_vfmul_vf_f32m4(v76, v13, 16);
    float* v78 = v11 + 64;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v78, v77, 16);
    const uint8_t* v79 = v8 + 16;
    const uint8_t* v80 = (const uint8_t*) v79;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v81 = __riscv_vle8_v_u8m1(v80, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v82 = __riscv_vmul_vx_u8m1(v81, 9, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v83 = __riscv_vzext_vf2_u16m2(v82, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v84 = __riscv_vmul_vx_u16m2(v83, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v85 = __riscv_vsrl_vx_u16m2(v84, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v86 = __riscv_vreinterpret_v_u16m2_i16m2(v85);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v87 = __riscv_vsext_vf2_i32m4(v86, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v88 = __riscv_vsub_vx_i32m4(v87, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v89 = __riscv_vfcvt_f_x_v_f32m4(v88, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v90 = __riscv_vfmul_vf_f32m4(v89, v13, 16);
    float* v91 = v11 + 80;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v91, v90, 16);
    const uint8_t* v92 = v8 + 0;
    const uint8_t* v93 = (const uint8_t*) v92;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v94 = __riscv_vle8_v_u8m1(v93, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v95 = __riscv_vmul_vx_u8m1(v94, 27, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v96 = __riscv_vzext_vf2_u16m2(v95, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v97 = __riscv_vmul_vx_u16m2(v96, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v98 = __riscv_vsrl_vx_u16m2(v97, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v99 = __riscv_vreinterpret_v_u16m2_i16m2(v98);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v100 = __riscv_vsext_vf2_i32m4(v99, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v101 = __riscv_vsub_vx_i32m4(v100, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v102 = __riscv_vfcvt_f_x_v_f32m4(v101, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v103 = __riscv_vfmul_vf_f32m4(v102, v13, 16);
    float* v104 = v11 + 96;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v104, v103, 16);
    const uint8_t* v105 = v8 + 16;
    const uint8_t* v106 = (const uint8_t*) v105;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v107 = __riscv_vle8_v_u8m1(v106, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v108 = __riscv_vmul_vx_u8m1(v107, 27, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v109 = __riscv_vzext_vf2_u16m2(v108, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v110 = __riscv_vmul_vx_u16m2(v109, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v111 = __riscv_vsrl_vx_u16m2(v110, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v112 = __riscv_vreinterpret_v_u16m2_i16m2(v111);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v113 = __riscv_vsext_vf2_i32m4(v112, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v114 = __riscv_vsub_vx_i32m4(v113, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v115 = __riscv_vfcvt_f_x_v_f32m4(v114, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v116 = __riscv_vfmul_vf_f32m4(v115, v13, 16);
    float* v117 = v11 + 112;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v117, v116, 16);
    const uint8_t* v118 = v8 + 0;
    const uint8_t* v119 = (const uint8_t*) v118;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v120 = __riscv_vle8_v_u8m1(v119, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v121 = __riscv_vmul_vx_u8m1(v120, 81, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v122 = __riscv_vzext_vf2_u16m2(v121, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v123 = __riscv_vmul_vx_u16m2(v122, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v124 = __riscv_vsrl_vx_u16m2(v123, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v125 = __riscv_vreinterpret_v_u16m2_i16m2(v124);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v126 = __riscv_vsext_vf2_i32m4(v125, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v127 = __riscv_vsub_vx_i32m4(v126, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v128 = __riscv_vfcvt_f_x_v_f32m4(v127, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v129 = __riscv_vfmul_vf_f32m4(v128, v13, 16);
    float* v130 = v11 + 128;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v130, v129, 16);
    const uint8_t* v131 = v8 + 16;
    const uint8_t* v132 = (const uint8_t*) v131;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v133 = __riscv_vle8_v_u8m1(v132, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v134 = __riscv_vmul_vx_u8m1(v133, 81, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v135 = __riscv_vzext_vf2_u16m2(v134, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v136 = __riscv_vmul_vx_u16m2(v135, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v137 = __riscv_vsrl_vx_u16m2(v136, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v138 = __riscv_vreinterpret_v_u16m2_i16m2(v137);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v139 = __riscv_vsext_vf2_i32m4(v138, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v140 = __riscv_vsub_vx_i32m4(v139, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v141 = __riscv_vfcvt_f_x_v_f32m4(v140, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v142 = __riscv_vfmul_vf_f32m4(v141, v13, 16);
    float* v143 = v11 + 144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v143, v142, 16);
    const uint8_t* v144 = v8 + 32;
    const uint8_t* v145 = (const uint8_t*) v144;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v146 = __riscv_vle8_v_u8m1(v145, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v147 = __riscv_vmul_vx_u8m1(v146, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v148 = __riscv_vzext_vf2_u16m2(v147, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v149 = __riscv_vmul_vx_u16m2(v148, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v150 = __riscv_vsrl_vx_u16m2(v149, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v151 = __riscv_vreinterpret_v_u16m2_i16m2(v150);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v152 = __riscv_vsext_vf2_i32m4(v151, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v153 = __riscv_vsub_vx_i32m4(v152, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v154 = __riscv_vfcvt_f_x_v_f32m4(v153, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v155 = __riscv_vfmul_vf_f32m4(v154, v13, 16);
    float* v156 = v11 + 160;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v156, v155, 16);
    const uint8_t* v157 = v8 + 32;
    const uint8_t* v158 = (const uint8_t*) v157;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v159 = __riscv_vle8_v_u8m1(v158, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v160 = __riscv_vmul_vx_u8m1(v159, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v161 = __riscv_vzext_vf2_u16m2(v160, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v162 = __riscv_vmul_vx_u16m2(v161, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v163 = __riscv_vsrl_vx_u16m2(v162, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v164 = __riscv_vreinterpret_v_u16m2_i16m2(v163);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v165 = __riscv_vsext_vf2_i32m4(v164, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v166 = __riscv_vsub_vx_i32m4(v165, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v167 = __riscv_vfcvt_f_x_v_f32m4(v166, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v168 = __riscv_vfmul_vf_f32m4(v167, v13, 16);
    float* v169 = v11 + 176;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v169, v168, 16);
    const uint8_t* v170 = v8 + 32;
    const uint8_t* v171 = (const uint8_t*) v170;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v172 = __riscv_vle8_v_u8m1(v171, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v173 = __riscv_vmul_vx_u8m1(v172, 9, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v174 = __riscv_vzext_vf2_u16m2(v173, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v175 = __riscv_vmul_vx_u16m2(v174, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v176 = __riscv_vsrl_vx_u16m2(v175, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v177 = __riscv_vreinterpret_v_u16m2_i16m2(v176);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v178 = __riscv_vsext_vf2_i32m4(v177, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v179 = __riscv_vsub_vx_i32m4(v178, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v180 = __riscv_vfcvt_f_x_v_f32m4(v179, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v181 = __riscv_vfmul_vf_f32m4(v180, v13, 16);
    float* v182 = v11 + 192;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v182, v181, 16);
    const uint8_t* v183 = v8 + 32;
    const uint8_t* v184 = (const uint8_t*) v183;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v185 = __riscv_vle8_v_u8m1(v184, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v186 = __riscv_vmul_vx_u8m1(v185, 27, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v187 = __riscv_vzext_vf2_u16m2(v186, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v188 = __riscv_vmul_vx_u16m2(v187, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v189 = __riscv_vsrl_vx_u16m2(v188, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v190 = __riscv_vreinterpret_v_u16m2_i16m2(v189);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v191 = __riscv_vsext_vf2_i32m4(v190, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v192 = __riscv_vsub_vx_i32m4(v191, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v193 = __riscv_vfcvt_f_x_v_f32m4(v192, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v194 = __riscv_vfmul_vf_f32m4(v193, v13, 16);
    float* v195 = v11 + 208;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v195, v194, 16);
    const uint8_t* v196 = v8 + 32;
    const uint8_t* v197 = (const uint8_t*) v196;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v198 = __riscv_vle8_v_u8m1(v197, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v199 = __riscv_vmul_vx_u8m1(v198, 81, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v200 = __riscv_vzext_vf2_u16m2(v199, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v201 = __riscv_vmul_vx_u16m2(v200, 3, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v202 = __riscv_vsrl_vx_u16m2(v201, 8, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v203 = __riscv_vreinterpret_v_u16m2_i16m2(v202);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v204 = __riscv_vsext_vf2_i32m4(v203, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v205 = __riscv_vsub_vx_i32m4(v204, 1, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v206 = __riscv_vfcvt_f_x_v_f32m4(v205, 16);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v207 = __riscv_vfmul_vf_f32m4(v206, v13, 16);
    float* v208 = v11 + 224;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v208, v207, 16);
    const uint8_t* v209 = v8 + 48;
    const uint8_t* v210 = (const uint8_t*) v209;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v211 = __riscv_vle8_v_u8m1(v210, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v212 = __riscv_vmul_vx_u8m1(v211, 1, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v213 = __riscv_vzext_vf2_u16m2(v212, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v214 = __riscv_vmul_vx_u16m2(v213, 3, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v215 = __riscv_vsrl_vx_u16m2(v214, 8, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v216 = __riscv_vreinterpret_v_u16m2_i16m2(v215);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v217 = __riscv_vsext_vf2_i32m4(v216, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v218 = __riscv_vsub_vx_i32m4(v217, 1, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v219 = __riscv_vfcvt_f_x_v_f32m4(v218, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v220 = __riscv_vfmul_vf_f32m4(v219, v13, 4);
    float* v221 = v11 + 240;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v221, v220, 4);
    const uint8_t* v222 = v8 + 48;
    const uint8_t* v223 = (const uint8_t*) v222;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v224 = __riscv_vle8_v_u8m1(v223, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v225 = __riscv_vmul_vx_u8m1(v224, 3, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v226 = __riscv_vzext_vf2_u16m2(v225, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v227 = __riscv_vmul_vx_u16m2(v226, 3, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v228 = __riscv_vsrl_vx_u16m2(v227, 8, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v229 = __riscv_vreinterpret_v_u16m2_i16m2(v228);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v230 = __riscv_vsext_vf2_i32m4(v229, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v231 = __riscv_vsub_vx_i32m4(v230, 1, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v232 = __riscv_vfcvt_f_x_v_f32m4(v231, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v233 = __riscv_vfmul_vf_f32m4(v232, v13, 4);
    float* v234 = v11 + 244;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v234, v233, 4);
    const uint8_t* v235 = v8 + 48;
    const uint8_t* v236 = (const uint8_t*) v235;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v237 = __riscv_vle8_v_u8m1(v236, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v238 = __riscv_vmul_vx_u8m1(v237, 9, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v239 = __riscv_vzext_vf2_u16m2(v238, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v240 = __riscv_vmul_vx_u16m2(v239, 3, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v241 = __riscv_vsrl_vx_u16m2(v240, 8, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v242 = __riscv_vreinterpret_v_u16m2_i16m2(v241);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v243 = __riscv_vsext_vf2_i32m4(v242, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v244 = __riscv_vsub_vx_i32m4(v243, 1, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v245 = __riscv_vfcvt_f_x_v_f32m4(v244, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v246 = __riscv_vfmul_vf_f32m4(v245, v13, 4);
    float* v247 = v11 + 248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v247, v246, 4);
    const uint8_t* v248 = v8 + 48;
    const uint8_t* v249 = (const uint8_t*) v248;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v250 = __riscv_vle8_v_u8m1(v249, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v251 = __riscv_vmul_vx_u8m1(v250, 27, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vzext_vf2_u16m2
    vuint16m2_t v252 = __riscv_vzext_vf2_u16m2(v251, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmul_vx_u16m2
    vuint16m2_t v253 = __riscv_vmul_vx_u16m2(v252, 3, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v254 = __riscv_vsrl_vx_u16m2(v253, 8, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u16m2_i16m2
    vint16m2_t v255 = __riscv_vreinterpret_v_u16m2_i16m2(v254);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsext_vf2_i32m4
    vint32m4_t v256 = __riscv_vsext_vf2_i32m4(v255, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i32m4
    vint32m4_t v257 = __riscv_vsub_vx_i32m4(v256, 1, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
    vfloat32m4_t v258 = __riscv_vfcvt_f_x_v_f32m4(v257, 4);
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m4
    vfloat32m4_t v259 = __riscv_vfmul_vf_f32m4(v258, v13, 4);
    float* v260 = v11 + 252;
    // weft_emitc.source_op=weft_rvv.typed_dequantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
    __riscv_vse32_v_f32m4(v260, v259, 4);
  }
  return;
}


