#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void k_gemv_NARROW(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v8 = v5 / 16;
  for (size_t v9 = 0; v9 < v8; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v10 = v9 * v7;
    size_t v11 = v10 * 544;
    const uint8_t* v12 = v3 + v11;
    vfloat32m2_t v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v14 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v13 = v14;
    vfloat32m2_t v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v16 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v15 = v16;
    for (size_t v17 = 0; v17 < v7; v17 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v18 = v17 * 544;
      const uint8_t* v19 = v12 + v18;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v20 = v17 * 34;
      const uint8_t* v21 = v4 + v20;
      vint32m2_t v22;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v23 = __riscv_vmv_v_x_i32m2(0, 8);
      v22 = v23;
      vint32m2_t v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v25 = __riscv_vmv_v_x_i32m2(0, 8);
      v24 = v25;
      for (size_t v26 = 0; v26 < 32; v26 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_quant_addr
        size_t v27 = v26 * 16;
        size_t v28 = 32 + v27;
        size_t v29 = v28 + 8;
        const uint8_t* v30 = v19 + v28;
        const int8_t* v31 = (const int8_t*) v30;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v32 = __riscv_vle8_v_i8mf2(v31, 8);
        const uint8_t* v33 = v19 + v29;
        const int8_t* v34 = (const int8_t*) v33;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v35 = __riscv_vle8_v_i8mf2(v34, 8);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
        size_t v36 = 2 + v26;
        const uint8_t* v37 = v21 + v36;
        const int8_t* v38 = (const int8_t*) v37;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v39 = *(const int8_t *)(v38);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v40 = __riscv_vwmul_vx_i16m1(v32, v39, 8);
        vint32m2_t v41 = v22;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v42 = __riscv_vwadd_wv_i32m2(v41, v40, 8);
        v22 = v42;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v43 = __riscv_vwmul_vx_i16m1(v35, v39, 8);
        vint32m2_t v44 = v24;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v45 = __riscv_vwadd_wv_i32m2(v44, v43, 8);
        v24 = v45;
      }
      vint32m2_t v46 = v22;
      vint32m2_t v47 = v24;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v48 = (const _Float16*) v19;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v49 = __riscv_vle16_v_f16m1(v48, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v50 = v19 + 16;
      const _Float16* v51 = (const _Float16*) v50;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v52 = __riscv_vle16_v_f16m1(v51, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v53 = (const _Float16*) v21;
      _Float16 v54 = *(const _Float16 *)(v53);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v55 = __riscv_vfwmul_vf_f32m2(v49, v54, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v56 = __riscv_vfcvt_f_x_v_f32m2(v46, 8);
      vfloat32m2_t v57 = v13;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v58 = __riscv_vfmacc_vv_f32m2(v57, v56, v55, 8);
      v13 = v58;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v59 = __riscv_vfwmul_vf_f32m2(v52, v54, 8);
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v60 = __riscv_vfcvt_f_x_v_f32m2(v47, 8);
      vfloat32m2_t v61 = v15;
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v62 = __riscv_vfmacc_vv_f32m2(v61, v60, v59, 8);
      v15 = v62;
    }
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v63 = v9 * 16;
    float* v64 = v2 + v63;
    vfloat32m2_t v65 = v13;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v64, v65, 8);
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v66 = v9 * 16;
    size_t v67 = v66 + 8;
    float* v68 = v2 + v67;
    vfloat32m2_t v69 = v15;
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q8_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v68, v69, 8);
  }
  vint32m1_t v70 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


