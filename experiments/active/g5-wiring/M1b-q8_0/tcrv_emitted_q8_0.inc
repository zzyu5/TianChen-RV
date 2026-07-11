/* tcrv_emitted_q8_0.inc  (G5-M1b: deploy our EMITTED vl=8 q8_0 repack kernels)
 * Front-door CONSTRUCTED q8_0 16x1 REPACK GEVM (decode) + GEMM (prefill), VLEN128
 * vl=8 two-8-lane-halves form (the correctness-carrier that intercepts BEFORE the
 * broken upstream vl=16 body -- see [GAP-Q8_0-VLEN128-KERNEL], G5-M1 casefile).
 * Emit: tcrv-opt --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc
 *       | mlir-translate --mlir-to-cpp  (VERBATIM, LLVM 20.1.8), HEAD 92c27389.
 * Source MLIR:
 *   GEVM: test/Conversion/RVV/rvv-emit-identity-quant-contraction-q8-0-repack-vlen128.mlir
 *   GEMM: test/Conversion/RVV/rvv-emit-quant-contraction-q8-0-repack-gemm-prefill-vlen128.mlir
 * ABI (as read by the deployed arch/riscv branch):
 *   GEVM tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_...(v1=n, v2=s, v3=nc[col count],
 *        v4=vx[block_q8_0x16 stride544], v5=bx[unused], v6=vy[plain q8_0 stride34], v7,v8 unused)
 *   GEMM tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_...(v1=nr, v2=bs[row stride], v3=n, v4=s,
 *        v5=nc, v6=vx[block_q8_0x16 stride544], v7=vy[block_q8_0x4 stride136])
 */
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v10 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v3 / 16;
  for (size_t v12 = 0; v12 < v11; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
    size_t v13 = v12 * v10;
    size_t v14 = v13 * 544;
    const uint8_t* v15 = v4 + v14;
    vfloat32m2_t v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v17 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v16 = v17;
    vfloat32m2_t v18;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v19 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
    v18 = v19;
    for (size_t v20 = 0; v20 < v10; v20 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
      size_t v21 = v20 * 544;
      const uint8_t* v22 = v15 + v21;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
      size_t v23 = v20 * 34;
      const uint8_t* v24 = v6 + v23;
      vint32m2_t v25;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v26 = __riscv_vmv_v_x_i32m2(0, 8);
      v25 = v26;
      vint32m2_t v27;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
      vint32m2_t v28 = __riscv_vmv_v_x_i32m2(0, 8);
      v27 = v28;
      for (size_t v29 = 0; v29 < 32; v29 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_quant_addr
        size_t v30 = v29 * 16;
        size_t v31 = 32 + v30;
        size_t v32 = v31 + 8;
        const uint8_t* v33 = v22 + v31;
        const int8_t* v34 = (const int8_t*) v33;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v35 = __riscv_vle8_v_i8mf2(v34, 8);
        const uint8_t* v36 = v22 + v32;
        const int8_t* v37 = (const int8_t*) v36;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
        vint8mf2_t v38 = __riscv_vle8_v_i8mf2(v37, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
        size_t v39 = 2 + v29;
        const uint8_t* v40 = v24 + v39;
        const int8_t* v41 = (const int8_t*) v40;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
        int32_t v42 = *(const int8_t *)(v41);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v43 = __riscv_vwmul_vx_i16m1(v35, v42, 8);
        vint32m2_t v44 = v25;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v45 = __riscv_vwadd_wv_i32m2(v44, v43, 8);
        v25 = v45;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
        vint16m1_t v46 = __riscv_vwmul_vx_i16m1(v38, v42, 8);
        vint32m2_t v47 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
        vint32m2_t v48 = __riscv_vwadd_wv_i32m2(v47, v46, 8);
        v27 = v48;
      }
      vint32m2_t v49 = v25;
      vint32m2_t v50 = v27;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const _Float16* v51 = (const _Float16*) v22;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v52 = __riscv_vle16_v_f16m1(v51, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
      const uint8_t* v53 = v22 + 16;
      const _Float16* v54 = (const _Float16*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
      vfloat16m1_t v55 = __riscv_vle16_v_f16m1(v54, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
      const _Float16* v56 = (const _Float16*) v24;
      _Float16 v57 = *(const _Float16 *)(v56);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v58 = __riscv_vfwmul_vf_f32m2(v52, v57, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v59 = __riscv_vfcvt_f_x_v_f32m2(v49, 8);
      vfloat32m2_t v60 = v16;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v61 = __riscv_vfmacc_vv_f32m2(v60, v59, v58, 8);
      v16 = v61;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
      vfloat32m2_t v62 = __riscv_vfwmul_vf_f32m2(v55, v57, 8);
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
      vfloat32m2_t v63 = __riscv_vfcvt_f_x_v_f32m2(v50, 8);
      vfloat32m2_t v64 = v18;
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
      vfloat32m2_t v65 = __riscv_vfmacc_vv_f32m2(v64, v63, v62, 8);
      v18 = v65;
    }
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v66 = v12 * 16;
    float* v67 = v2 + v66;
    vfloat32m2_t v68 = v16;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v67, v68, 8);
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
    size_t v69 = v12 * 16;
    size_t v70 = v69 + 8;
    float* v71 = v2 + v70;
    vfloat32m2_t v72 = v18;
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemv_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v71, v72, 8);
  }
  return;
}



extern "C" void tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v3);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v9 = v3 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v1 / 4;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v5 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v7 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 544;
      const uint8_t* v19 = v6 + v18;
      for (size_t v20 = 0; v20 < 2; v20 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 8;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v22 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v24 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v25 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        vfloat32m2_t v26;
        v26 = v22;
        vfloat32m2_t v27;
        v27 = v23;
        vfloat32m2_t v28;
        v28 = v24;
        vfloat32m2_t v29;
        v29 = v25;
        for (size_t v30 = 0; v30 < v9; v30 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v31 = v30 * 544;
          const uint8_t* v32 = v19 + v31;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 136;
          const uint8_t* v34 = v15 + v33;
          vint32m2_t v35;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v36 = __riscv_vmv_v_x_i32m2(0, 8);
          v35 = v36;
          vint32m2_t v37;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v38 = __riscv_vmv_v_x_i32m2(0, 8);
          v37 = v38;
          vint32m2_t v39;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v40 = __riscv_vmv_v_x_i32m2(0, 8);
          v39 = v40;
          vint32m2_t v41;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v42 = __riscv_vmv_v_x_i32m2(0, 8);
          v41 = v42;
          for (size_t v43 = 0; v43 < 32; v43 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v44 = v43 * 16;
            size_t v45 = 32 + v44;
            size_t v46 = v45 + v21;
            const uint8_t* v47 = v32 + v46;
            const int8_t* v48 = (const int8_t*) v47;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
            vint8mf2_t v49 = __riscv_vle8_v_i8mf2(v48, 8);
            size_t v50 = v43 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
            size_t v51 = v50 + 0;
            size_t v52 = 8 + v51;
            const uint8_t* v53 = v34 + v52;
            const int8_t* v54 = (const int8_t*) v53;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v55 = *(const int8_t *)(v54);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v56 = __riscv_vwmul_vx_i16m1(v49, v55, 8);
            vint32m2_t v57 = v35;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v58 = __riscv_vwadd_wv_i32m2(v57, v56, 8);
            v35 = v58;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
            size_t v59 = v50 + 1;
            size_t v60 = 8 + v59;
            const uint8_t* v61 = v34 + v60;
            const int8_t* v62 = (const int8_t*) v61;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v63 = *(const int8_t *)(v62);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v64 = __riscv_vwmul_vx_i16m1(v49, v63, 8);
            vint32m2_t v65 = v37;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v66 = __riscv_vwadd_wv_i32m2(v65, v64, 8);
            v37 = v66;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
            size_t v67 = v50 + 2;
            size_t v68 = 8 + v67;
            const uint8_t* v69 = v34 + v68;
            const int8_t* v70 = (const int8_t*) v69;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v71 = *(const int8_t *)(v70);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v72 = __riscv_vwmul_vx_i16m1(v49, v71, 8);
            vint32m2_t v73 = v39;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v74 = __riscv_vwadd_wv_i32m2(v73, v72, 8);
            v39 = v74;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr
            size_t v75 = v50 + 3;
            size_t v76 = 8 + v75;
            const uint8_t* v77 = v34 + v76;
            const int8_t* v78 = (const int8_t*) v77;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v79 = *(const int8_t *)(v78);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v80 = __riscv_vwmul_vx_i16m1(v49, v79, 8);
            vint32m2_t v81 = v41;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v82 = __riscv_vwadd_wv_i32m2(v81, v80, 8);
            v41 = v82;
          }
          vint32m2_t v83 = v35;
          vint32m2_t v84 = v37;
          vint32m2_t v85 = v39;
          vint32m2_t v86 = v41;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v87 = v21 * 2;
          const uint8_t* v88 = v32 + v87;
          const _Float16* v89 = (const _Float16*) v88;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v90 = __riscv_vle16_v_f16m1(v89, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v91 = v34 + 0;
          const _Float16* v92 = (const _Float16*) v91;
          _Float16 v93 = *(const _Float16 *)(v92);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v94 = __riscv_vfwmul_vf_f32m2(v90, v93, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v95 = __riscv_vfcvt_f_x_v_f32m2(v83, 8);
          vfloat32m2_t v96 = v26;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v97 = __riscv_vfmacc_vv_f32m2(v96, v95, v94, 8);
          v26 = v97;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v98 = v34 + 2;
          const _Float16* v99 = (const _Float16*) v98;
          _Float16 v100 = *(const _Float16 *)(v99);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v101 = __riscv_vfwmul_vf_f32m2(v90, v100, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v102 = __riscv_vfcvt_f_x_v_f32m2(v84, 8);
          vfloat32m2_t v103 = v27;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v104 = __riscv_vfmacc_vv_f32m2(v103, v102, v101, 8);
          v27 = v104;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v105 = v34 + 4;
          const _Float16* v106 = (const _Float16*) v105;
          _Float16 v107 = *(const _Float16 *)(v106);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v108 = __riscv_vfwmul_vf_f32m2(v90, v107, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v109 = __riscv_vfcvt_f_x_v_f32m2(v85, 8);
          vfloat32m2_t v110 = v28;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v111 = __riscv_vfmacc_vv_f32m2(v110, v109, v108, 8);
          v28 = v111;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v112 = v34 + 6;
          const _Float16* v113 = (const _Float16*) v112;
          _Float16 v114 = *(const _Float16 *)(v113);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v115 = __riscv_vfwmul_vf_f32m2(v90, v114, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v116 = __riscv_vfcvt_f_x_v_f32m2(v86, 8);
          vfloat32m2_t v117 = v29;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v118 = __riscv_vfmacc_vv_f32m2(v117, v116, v115, 8);
          v29 = v118;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v119 = v12 * 4;
        size_t v120 = v119 + 0;
        size_t v121 = v120 * v2;
        size_t v122 = v16 * 16;
        size_t v123 = v121 + v122;
        size_t v124 = v123 + v21;
        float* v125 = v4 + v124;
        vfloat32m2_t v126 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v125, v126, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v127 = v12 * 4;
        size_t v128 = v127 + 1;
        size_t v129 = v128 * v2;
        size_t v130 = v16 * 16;
        size_t v131 = v129 + v130;
        size_t v132 = v131 + v21;
        float* v133 = v4 + v132;
        vfloat32m2_t v134 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v133, v134, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v135 = v12 * 4;
        size_t v136 = v135 + 2;
        size_t v137 = v136 * v2;
        size_t v138 = v16 * 16;
        size_t v139 = v137 + v138;
        size_t v140 = v139 + v21;
        float* v141 = v4 + v140;
        vfloat32m2_t v142 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v141, v142, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v143 = v12 * 4;
        size_t v144 = v143 + 3;
        size_t v145 = v144 * v2;
        size_t v146 = v16 * 16;
        size_t v147 = v145 + v146;
        size_t v148 = v147 + v21;
        float* v149 = v4 + v148;
        vfloat32m2_t v150 = v29;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v149, v150, 8);
      }
    }
  }
  return;
}


