#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(size_t v1, const float* v2, uint8_t* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v5 = v1 / 256;
  // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=vlmax
  // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvlmax_e32m8
  size_t v6 = __riscv_vsetvlmax_e32m8();
  for (size_t v7 = 0; v7 < v5; v7 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=x_block
    size_t v8 = v7 * 256;
    const float* v9 = v2 + v8;
    const float* v10 = (const float*) v9;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=y_block
    size_t v11 = v7 * 292;
    uint8_t* v12 = v3 + v11;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=minmax_init
    vfloat32m8_t v13;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v14 = __riscv_vfmv_v_f_f32m8(-__builtin_inff(), v6);
    v13 = v14;
    vfloat32m8_t v15;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m8
    vfloat32m8_t v16 = __riscv_vfmv_v_f_f32m8(__builtin_inff(), v6);
    v15 = v16;
    for (size_t v17 = 0; v17 < 256; v17 += v6) {
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
      size_t v18 = 256 - v17;
      size_t v19 = __riscv_vsetvl_e32m8(v18);
      const float* v20 = v10 + v17;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
      vfloat32m8_t v21 = __riscv_vle32_v_f32m8(v20, v19);
      vfloat32m8_t v22 = v13;
      vfloat32m8_t v23 = v15;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmax_vv_f32m8
      vfloat32m8_t v24 = __riscv_vfmax_vv_f32m8(v22, v21, v19);
      v13 = v24;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmin_vv_f32m8
      vfloat32m8_t v25 = __riscv_vfmin_vv_f32m8(v23, v21, v19);
      v15 = v25;
    }
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=minmax_reduce
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_s_f_f32m1
    vfloat32m1_t v26 = __riscv_vfmv_s_f_f32m1(-__builtin_inff(), 1);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_s_f_f32m1
    vfloat32m1_t v27 = __riscv_vfmv_s_f_f32m1(__builtin_inff(), 1);
    vfloat32m8_t v28 = v13;
    vfloat32m8_t v29 = v15;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfredmax_vs_f32m8_f32m1
    vfloat32m1_t v30 = __riscv_vfredmax_vs_f32m8_f32m1(v28, v26, v6);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfredmin_vs_f32m8_f32m1
    vfloat32m1_t v31 = __riscv_vfredmin_vs_f32m8_f32m1(v29, v27, v6);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f32m1_f32
    float v32 = __riscv_vfmv_f_s_f32m1_f32(v30);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f32m1_f32
    float v33 = __riscv_vfmv_f_s_f32m1_f32(v31);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=amax
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fabsf
    float v34 = fabsf(v32);
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fabsf
    float v35 = fabsf(v33);
    bool v36 = v34 > v35;
    float v37;
    v37 = v35;
    float v38;
    v38 = v33;
    if (v36) {
      v37 = v34;
      v38 = v32;
    }
    float v39 = v37;
    float v40 = v38;
    // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=zero_block_guard
    bool v41 = v39 == 0.0f;
    if (v41) {
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=zero_block
      uint8_t* v42 = v12 + 0;
      float* v43 = (float*) v42;
      v43[0] = 0.0f;
      uint8_t* v44 = v12 + 4;
      int8_t* v45 = (int8_t*) v44;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=memset
      memset(v45, 0, 256);
      uint8_t* v46 = v12 + 260;
      int16_t* v47 = (int16_t*) v46;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=memset
      memset(v47, 0, 32);
    } else {
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=iscale
      float v48 = -127.f / v40;
      float v49 = 1.0f / v48;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=float_d_store
      uint8_t* v50 = v12 + 0;
      float* v51 = (float*) v50;
      v51[0] = v49;
      // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
      vint16m1_t v52 = __riscv_vmv_v_x_i16m1(0, 1);
      for (size_t v53 = 0; v53 < 256; v53 += v6) {
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m8
        size_t v54 = 256 - v53;
        size_t v55 = __riscv_vsetvl_e32m8(v54);
        const float* v56 = v10 + v53;
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m8
        vfloat32m8_t v57 = __riscv_vle32_v_f32m8(v56, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m8
        vfloat32m8_t v58 = __riscv_vfmul_vf_f32m8(v57, v48, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_x_f_v_i32m8_rm
        vint32m8_t v59 = __riscv_vfcvt_x_f_v_i32m8_rm(v58, __RISCV_FRM_RNE, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vnclip_wx_i16m4
        vint16m4_t v60 = __riscv_vnclip_wx_i16m4(v59, 0, __RISCV_VXRM_RNE, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vnclip_wx_i8m2
        vint8m2_t v61 = __riscv_vnclip_wx_i8m2(v60, 0, __RISCV_VXRM_RNE, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qs_store
        uint8_t* v62 = v12 + 4;
        int8_t* v63 = (int8_t*) v62;
        int8_t* v64 = v63 + v53;
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
        __riscv_vse8_v_i8m2(v64, v61, v55);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=bsums_first
        size_t v65 = v53 / 16;
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m2_i8m1
        vint8m1_t v66 = __riscv_vget_v_i8m2_i8m1(v61, 0);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
        vint16m1_t v67 = __riscv_vwredsum_vs_i8m1_i16m1(v66, v52, 16);
        // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
        int16_t v68 = __riscv_vmv_x_s_i16m1_i16(v67);
        uint8_t* v69 = v12 + 260;
        int16_t* v70 = (int16_t*) v69;
        int16_t* v71 = v70 + v65;
        v71[0] = v68;
        vint8m2_t v72;
        v72 = v61;
        for (size_t v73 = 16; v73 < v55; v73 += 16) {
          vint8m2_t v74 = v72;
          // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vslidedown_vx_i8m2
          vint8m2_t v75 = __riscv_vslidedown_vx_i8m2(v74, 16, v55);
          v72 = v75;
          size_t v76 = v53 + v73;
          size_t v77 = v76 / 16;
          // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vget_v_i8m2_i8m1
          vint8m1_t v78 = __riscv_vget_v_i8m2_i8m1(v75, 0);
          // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
          vint16m1_t v79 = __riscv_vwredsum_vs_i8m1_i16m1(v78, v52, 16);
          // weft_emitc.source_op=weft_rvv.typed_quantize_row_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
          int16_t v80 = __riscv_vmv_x_s_i16m1_i16(v79);
          uint8_t* v81 = v12 + 260;
          int16_t* v82 = (int16_t*) v81;
          int16_t* v83 = v82 + v77;
          v83[0] = v80;
        }
      }
    }
  }
  return;
}


