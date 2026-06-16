#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_gemm_m6_gemm_m6(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7, size_t v8, size_t v9) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v10 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v11 = v1 / 32;
  for (size_t v12 = 0; v12 < v6; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_row_base
    size_t v13 = v12 * v8;
    const uint8_t* v14 = v3 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_row_base
    size_t v15 = v12 * v9;
    float* v16 = v2 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=full_strip_span
    size_t v17 = v7 / 6;
    size_t v18 = v17 * 6;
    for (size_t v19 = 0; v19 < v18; v19 += 6) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=strip_base_y
      size_t v20 = v19 * v5;
      const uint8_t* v21 = v4 + v20;
      // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface
      float v22[6];
      for (size_t v23 = 0; v23 < 6; v23 += 1) {
        v22[v23] = 0.0f;
      }
      for (size_t v24 = 0; v24 < v11; v24 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
        size_t v25 = v24 * 18;
        const uint8_t* v26 = v14 + v25;
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
        float v27 = (float)*(const _Float16 *)(v26);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
        size_t v28 = __riscv_vsetvl_e8m1(16);
        const uint8_t* v29 = v26 + 2;
        const int8_t* v30 = (const int8_t*) v29;
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
        vint8m1_t v31 = __riscv_vle8_v_i8m1(v30, v28);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
        vint8m1_t v32 = __riscv_vxor_vx_i8m1(v31, 0x88, v28);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
        vint8m1_t v33 = __riscv_vsll_vx_i8m1(v32, 4, v28);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v34 = __riscv_vsra_vx_i8m1(v33, 4, v28);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v35 = __riscv_vsra_vx_i8m1(v32, 4, v28);
        for (size_t v36 = 0; v36 < 6; v36 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=column_base_y
          size_t v37 = v36 * v5;
          const uint8_t* v38 = v21 + v37;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
          size_t v39 = v24 * 34;
          const uint8_t* v40 = v38 + v39;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
          float v41 = (float)*(const _Float16 *)(v40);
          const uint8_t* v42 = v40 + 2;
          const int8_t* v43 = (const int8_t*) v42;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
          vint8m1_t v44 = __riscv_vle8_v_i8m1(v43, v28);
          const uint8_t* v45 = v40 + 18;
          const int8_t* v46 = (const int8_t*) v45;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
          vint8m1_t v47 = __riscv_vle8_v_i8m1(v46, v28);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
          vint16m2_t v48 = __riscv_vwmul_vv_i16m2(v34, v44, v28);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
          vint16m2_t v49 = __riscv_vwmacc_vv_i16m2(v48, v35, v47, v28);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
          vint32m1_t v50 = __riscv_vmv_v_x_i32m1(0, 1);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
          vint32m1_t v51 = __riscv_vwredsum_vs_i16m2_i32m1(v49, v50, v28);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
          int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
          float v53 = v22[v36];
          // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface
          v22[v36] = v53 + ((float) v52 * v27) * v41;
        }
      }
      for (size_t v54 = 0; v54 < 6; v54 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
        size_t v55 = v19 + v54;
        float v56 = v22[v54];
        v16[v55] = v56;
      }
    }
    size_t v57 = v7 - v18;
    bool v58 = v18 < v7;
    if (v58) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=strip_base_y
      size_t v59 = v18 * v5;
      const uint8_t* v60 = v4 + v59;
      // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface
      float v61[6];
      for (size_t v62 = 0; v62 < v57; v62 += 1) {
        v61[v62] = 0.0f;
      }
      for (size_t v63 = 0; v63 < v11; v63 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
        size_t v64 = v63 * 18;
        const uint8_t* v65 = v14 + v64;
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
        float v66 = (float)*(const _Float16 *)(v65);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
        size_t v67 = __riscv_vsetvl_e8m1(16);
        const uint8_t* v68 = v65 + 2;
        const int8_t* v69 = (const int8_t*) v68;
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
        vint8m1_t v70 = __riscv_vle8_v_i8m1(v69, v67);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vxor_vx_i8m1
        vint8m1_t v71 = __riscv_vxor_vx_i8m1(v70, 0x88, v67);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
        vint8m1_t v72 = __riscv_vsll_vx_i8m1(v71, 4, v67);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v73 = __riscv_vsra_vx_i8m1(v72, 4, v67);
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
        vint8m1_t v74 = __riscv_vsra_vx_i8m1(v71, 4, v67);
        for (size_t v75 = 0; v75 < v57; v75 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=column_base_y
          size_t v76 = v75 * v5;
          const uint8_t* v77 = v60 + v76;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
          size_t v78 = v63 * 34;
          const uint8_t* v79 = v77 + v78;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
          float v80 = (float)*(const _Float16 *)(v79);
          const uint8_t* v81 = v79 + 2;
          const int8_t* v82 = (const int8_t*) v81;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
          vint8m1_t v83 = __riscv_vle8_v_i8m1(v82, v67);
          const uint8_t* v84 = v79 + 18;
          const int8_t* v85 = (const int8_t*) v84;
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
          vint8m1_t v86 = __riscv_vle8_v_i8m1(v85, v67);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
          vint16m2_t v87 = __riscv_vwmul_vv_i16m2(v73, v83, v67);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vv_i16m2
          vint16m2_t v88 = __riscv_vwmacc_vv_i16m2(v87, v74, v86, v67);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
          vint32m1_t v89 = __riscv_vmv_v_x_i32m1(0, 1);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
          vint32m1_t v90 = __riscv_vwredsum_vs_i16m2_i32m1(v88, v89, v67);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
          int32_t v91 = __riscv_vmv_x_s_i32m1_i32(v90);
          // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
          float v92 = v61[v75];
          // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface
          v61[v75] = v92 + ((float) v91 * v66) * v80;
        }
      }
      for (size_t v93 = 0; v93 < v57; v93 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.q4_0_q8_0_gemm role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
        size_t v94 = v18 + v93;
        float v95 = v61[v93];
        v16[v94] = v95;
      }
    }
  }
  vint32m1_t v96 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


