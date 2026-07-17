#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_mf4_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 34;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v16 = __riscv_vsetvl_e32m1(32);
    for (size_t v17 = 0; v17 < 32; v17 += v16) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v18 = 32 - v17;
      size_t v19 = __riscv_vsetvl_e32m1(v18);
      const uint8_t* v20 = v10 + 2;
      const uint8_t* v21 = v20 + v17;
      const int8_t* v22 = (const int8_t*) v21;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v23 = __riscv_vle8_v_i8mf4(v22, v19);
      const uint8_t* v24 = v12 + 2;
      const uint8_t* v25 = v24 + v17;
      const int8_t* v26 = (const int8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v27 = __riscv_vle8_v_i8mf4(v26, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v28 = __riscv_vwmul_vv_i16mf2(v23, v27, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v29 = v15;
      vint32m1_t v30 = __riscv_vmv_v_x_i32m1(v29, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v31 = __riscv_vwredsum_vs_i16mf2_i32m1(v28, v30, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v32 = __riscv_vmv_x_s_i32m1_i32(v31);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v15 = v32;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v33 = v15;
    float v34 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v34 + (float) v33 * (v13 * v14);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v35 = v6;
  v2[0] = v35;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_mf4_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v18 = __riscv_vsetvl_e32m1(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e32m1(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v25 = __riscv_vle8_v_i8mf4(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v30 = __riscv_vwmul_vv_i16mf2(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16mf2_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v44 = __riscv_vsetvl_e32m1(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e32m1(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v51 = __riscv_vle8_v_i8mf4(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v56 = __riscv_vwmul_vv_i16mf2(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16mf2_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v61 = v17;
    float v62 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v62 + (float) v61 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v63 = v43;
    float v64 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v64 + (float) v63 * (v41 * v42);
  }
  for (size_t v65 = v9; v65 < v7; v65 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v65 * 34;
    const uint8_t* v67 = v3 + v66;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v68 = v65 * 34;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v67);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v72;
    v72 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v73 = __riscv_vsetvl_e32m1(32);
    for (size_t v74 = 0; v74 < 32; v74 += v73) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v75 = 32 - v74;
      size_t v76 = __riscv_vsetvl_e32m1(v75);
      const uint8_t* v77 = v67 + 2;
      const uint8_t* v78 = v77 + v74;
      const int8_t* v79 = (const int8_t*) v78;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v80 = __riscv_vle8_v_i8mf4(v79, v76);
      const uint8_t* v81 = v69 + 2;
      const uint8_t* v82 = v81 + v74;
      const int8_t* v83 = (const int8_t*) v82;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v84 = __riscv_vle8_v_i8mf4(v83, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v85 = __riscv_vwmul_vv_i16mf2(v80, v84, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v86 = v72;
      vint32m1_t v87 = __riscv_vmv_v_x_i32m1(v86, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v88 = __riscv_vwredsum_vs_i16mf2_i32m1(v85, v87, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v89 = __riscv_vmv_x_s_i32m1_i32(v88);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v72 = v89;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v90 = v72;
    float v91 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v91 + (float) v90 * (v70 * v71);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v92 = v6;
  v2[0] = v92;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_mf4_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v18 = __riscv_vsetvl_e32m1(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e32m1(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v25 = __riscv_vle8_v_i8mf4(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v29 = __riscv_vle8_v_i8mf4(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v30 = __riscv_vwmul_vv_i16mf2(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16mf2_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v44 = __riscv_vsetvl_e32m1(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e32m1(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v51 = __riscv_vle8_v_i8mf4(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v55 = __riscv_vle8_v_i8mf4(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v56 = __riscv_vwmul_vv_i16mf2(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16mf2_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v61 = v10 + 2;
    size_t v62 = v61 * 34;
    const uint8_t* v63 = v3 + v62;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v64 = v10 + 2;
    size_t v65 = v64 * 34;
    const uint8_t* v66 = v4 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v67 = (float)*(const _Float16 *)(v63);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v68 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v69;
    v69 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v70 = __riscv_vsetvl_e32m1(32);
    for (size_t v71 = 0; v71 < 32; v71 += v70) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v72 = 32 - v71;
      size_t v73 = __riscv_vsetvl_e32m1(v72);
      const uint8_t* v74 = v63 + 2;
      const uint8_t* v75 = v74 + v71;
      const int8_t* v76 = (const int8_t*) v75;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v77 = __riscv_vle8_v_i8mf4(v76, v73);
      const uint8_t* v78 = v66 + 2;
      const uint8_t* v79 = v78 + v71;
      const int8_t* v80 = (const int8_t*) v79;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v81 = __riscv_vle8_v_i8mf4(v80, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v82 = __riscv_vwmul_vv_i16mf2(v77, v81, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v83 = v69;
      vint32m1_t v84 = __riscv_vmv_v_x_i32m1(v83, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v85 = __riscv_vwredsum_vs_i16mf2_i32m1(v82, v84, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v86 = __riscv_vmv_x_s_i32m1_i32(v85);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v69 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v87 = v10 + 3;
    size_t v88 = v87 * 34;
    const uint8_t* v89 = v3 + v88;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v90 = v10 + 3;
    size_t v91 = v90 * 34;
    const uint8_t* v92 = v4 + v91;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v93 = (float)*(const _Float16 *)(v89);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v94 = (float)*(const _Float16 *)(v92);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v95;
    v95 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v96 = __riscv_vsetvl_e32m1(32);
    for (size_t v97 = 0; v97 < 32; v97 += v96) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v98 = 32 - v97;
      size_t v99 = __riscv_vsetvl_e32m1(v98);
      const uint8_t* v100 = v89 + 2;
      const uint8_t* v101 = v100 + v97;
      const int8_t* v102 = (const int8_t*) v101;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v103 = __riscv_vle8_v_i8mf4(v102, v99);
      const uint8_t* v104 = v92 + 2;
      const uint8_t* v105 = v104 + v97;
      const int8_t* v106 = (const int8_t*) v105;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v107 = __riscv_vle8_v_i8mf4(v106, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v108 = __riscv_vwmul_vv_i16mf2(v103, v107, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v109 = v95;
      vint32m1_t v110 = __riscv_vmv_v_x_i32m1(v109, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v111 = __riscv_vwredsum_vs_i16mf2_i32m1(v108, v110, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v112 = __riscv_vmv_x_s_i32m1_i32(v111);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v95 = v112;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v113 = v17;
    float v114 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v114 + (float) v113 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v115 = v43;
    float v116 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v116 + (float) v115 * (v41 * v42);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v117 = v69;
    float v118 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v118 + (float) v117 * (v67 * v68);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v119 = v95;
    float v120 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v120 + (float) v119 * (v93 * v94);
  }
  for (size_t v121 = v9; v121 < v7; v121 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v122 = v121 * 34;
    const uint8_t* v123 = v3 + v122;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v124 = v121 * 34;
    const uint8_t* v125 = v4 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v126 = (float)*(const _Float16 *)(v123);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v127 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v128;
    v128 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
    size_t v129 = __riscv_vsetvl_e32m1(32);
    for (size_t v130 = 0; v130 < 32; v130 += v129) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
      size_t v131 = 32 - v130;
      size_t v132 = __riscv_vsetvl_e32m1(v131);
      const uint8_t* v133 = v123 + 2;
      const uint8_t* v134 = v133 + v130;
      const int8_t* v135 = (const int8_t*) v134;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v136 = __riscv_vle8_v_i8mf4(v135, v132);
      const uint8_t* v137 = v125 + 2;
      const uint8_t* v138 = v137 + v130;
      const int8_t* v139 = (const int8_t*) v138;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf4
      vint8mf4_t v140 = __riscv_vle8_v_i8mf4(v139, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16mf2
      vint16mf2_t v141 = __riscv_vwmul_vv_i16mf2(v136, v140, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v142 = v128;
      vint32m1_t v143 = __riscv_vmv_v_x_i32m1(v142, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16mf2_i32m1
      vint32m1_t v144 = __riscv_vwredsum_vs_i16mf2_i32m1(v141, v143, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v145 = __riscv_vmv_x_s_i32m1_i32(v144);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v128 = v145;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v146 = v128;
    float v147 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v147 + (float) v146 * (v126 * v127);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v148 = v6;
  v2[0] = v148;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m1_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 34;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v16 = __riscv_vsetvl_e8m1(32);
    for (size_t v17 = 0; v17 < 32; v17 += v16) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v18 = 32 - v17;
      size_t v19 = __riscv_vsetvl_e8m1(v18);
      const uint8_t* v20 = v10 + 2;
      const uint8_t* v21 = v20 + v17;
      const int8_t* v22 = (const int8_t*) v21;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v23 = __riscv_vle8_v_i8m1(v22, v19);
      const uint8_t* v24 = v12 + 2;
      const uint8_t* v25 = v24 + v17;
      const int8_t* v26 = (const int8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v27 = __riscv_vle8_v_i8m1(v26, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v28 = __riscv_vwmul_vv_i16m2(v23, v27, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v29 = v15;
      vint32m1_t v30 = __riscv_vmv_v_x_i32m1(v29, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v31 = __riscv_vwredsum_vs_i16m2_i32m1(v28, v30, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v32 = __riscv_vmv_x_s_i32m1_i32(v31);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v15 = v32;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v33 = v15;
    float v34 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v34 + (float) v33 * (v13 * v14);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v35 = v6;
  v2[0] = v35;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m1_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v18 = __riscv_vsetvl_e8m1(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e8m1(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v25 = __riscv_vle8_v_i8m1(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v30 = __riscv_vwmul_vv_i16m2(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16m2_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v44 = __riscv_vsetvl_e8m1(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e8m1(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v51 = __riscv_vle8_v_i8m1(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v55 = __riscv_vle8_v_i8m1(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v56 = __riscv_vwmul_vv_i16m2(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16m2_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v61 = v17;
    float v62 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v62 + (float) v61 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v63 = v43;
    float v64 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v64 + (float) v63 * (v41 * v42);
  }
  for (size_t v65 = v9; v65 < v7; v65 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v65 * 34;
    const uint8_t* v67 = v3 + v66;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v68 = v65 * 34;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v67);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v72;
    v72 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v73 = __riscv_vsetvl_e8m1(32);
    for (size_t v74 = 0; v74 < 32; v74 += v73) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v75 = 32 - v74;
      size_t v76 = __riscv_vsetvl_e8m1(v75);
      const uint8_t* v77 = v67 + 2;
      const uint8_t* v78 = v77 + v74;
      const int8_t* v79 = (const int8_t*) v78;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v80 = __riscv_vle8_v_i8m1(v79, v76);
      const uint8_t* v81 = v69 + 2;
      const uint8_t* v82 = v81 + v74;
      const int8_t* v83 = (const int8_t*) v82;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v84 = __riscv_vle8_v_i8m1(v83, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v85 = __riscv_vwmul_vv_i16m2(v80, v84, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v86 = v72;
      vint32m1_t v87 = __riscv_vmv_v_x_i32m1(v86, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v88 = __riscv_vwredsum_vs_i16m2_i32m1(v85, v87, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v89 = __riscv_vmv_x_s_i32m1_i32(v88);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v72 = v89;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v90 = v72;
    float v91 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v91 + (float) v90 * (v70 * v71);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v92 = v6;
  v2[0] = v92;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m1_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v18 = __riscv_vsetvl_e8m1(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e8m1(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v25 = __riscv_vle8_v_i8m1(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v30 = __riscv_vwmul_vv_i16m2(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16m2_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v44 = __riscv_vsetvl_e8m1(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e8m1(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v51 = __riscv_vle8_v_i8m1(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v55 = __riscv_vle8_v_i8m1(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v56 = __riscv_vwmul_vv_i16m2(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16m2_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v61 = v10 + 2;
    size_t v62 = v61 * 34;
    const uint8_t* v63 = v3 + v62;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v64 = v10 + 2;
    size_t v65 = v64 * 34;
    const uint8_t* v66 = v4 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v67 = (float)*(const _Float16 *)(v63);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v68 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v69;
    v69 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v70 = __riscv_vsetvl_e8m1(32);
    for (size_t v71 = 0; v71 < 32; v71 += v70) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v72 = 32 - v71;
      size_t v73 = __riscv_vsetvl_e8m1(v72);
      const uint8_t* v74 = v63 + 2;
      const uint8_t* v75 = v74 + v71;
      const int8_t* v76 = (const int8_t*) v75;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v77 = __riscv_vle8_v_i8m1(v76, v73);
      const uint8_t* v78 = v66 + 2;
      const uint8_t* v79 = v78 + v71;
      const int8_t* v80 = (const int8_t*) v79;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v81 = __riscv_vle8_v_i8m1(v80, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v82 = __riscv_vwmul_vv_i16m2(v77, v81, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v83 = v69;
      vint32m1_t v84 = __riscv_vmv_v_x_i32m1(v83, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v85 = __riscv_vwredsum_vs_i16m2_i32m1(v82, v84, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v86 = __riscv_vmv_x_s_i32m1_i32(v85);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v69 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v87 = v10 + 3;
    size_t v88 = v87 * 34;
    const uint8_t* v89 = v3 + v88;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v90 = v10 + 3;
    size_t v91 = v90 * 34;
    const uint8_t* v92 = v4 + v91;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v93 = (float)*(const _Float16 *)(v89);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v94 = (float)*(const _Float16 *)(v92);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v95;
    v95 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v96 = __riscv_vsetvl_e8m1(32);
    for (size_t v97 = 0; v97 < 32; v97 += v96) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v98 = 32 - v97;
      size_t v99 = __riscv_vsetvl_e8m1(v98);
      const uint8_t* v100 = v89 + 2;
      const uint8_t* v101 = v100 + v97;
      const int8_t* v102 = (const int8_t*) v101;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v103 = __riscv_vle8_v_i8m1(v102, v99);
      const uint8_t* v104 = v92 + 2;
      const uint8_t* v105 = v104 + v97;
      const int8_t* v106 = (const int8_t*) v105;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v107 = __riscv_vle8_v_i8m1(v106, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v108 = __riscv_vwmul_vv_i16m2(v103, v107, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v109 = v95;
      vint32m1_t v110 = __riscv_vmv_v_x_i32m1(v109, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v111 = __riscv_vwredsum_vs_i16m2_i32m1(v108, v110, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v112 = __riscv_vmv_x_s_i32m1_i32(v111);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v95 = v112;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v113 = v17;
    float v114 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v114 + (float) v113 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v115 = v43;
    float v116 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v116 + (float) v115 * (v41 * v42);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v117 = v69;
    float v118 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v118 + (float) v117 * (v67 * v68);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v119 = v95;
    float v120 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v120 + (float) v119 * (v93 * v94);
  }
  for (size_t v121 = v9; v121 < v7; v121 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v122 = v121 * 34;
    const uint8_t* v123 = v3 + v122;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v124 = v121 * 34;
    const uint8_t* v125 = v4 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v126 = (float)*(const _Float16 *)(v123);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v127 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v128;
    v128 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v129 = __riscv_vsetvl_e8m1(32);
    for (size_t v130 = 0; v130 < 32; v130 += v129) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v131 = 32 - v130;
      size_t v132 = __riscv_vsetvl_e8m1(v131);
      const uint8_t* v133 = v123 + 2;
      const uint8_t* v134 = v133 + v130;
      const int8_t* v135 = (const int8_t*) v134;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v136 = __riscv_vle8_v_i8m1(v135, v132);
      const uint8_t* v137 = v125 + 2;
      const uint8_t* v138 = v137 + v130;
      const int8_t* v139 = (const int8_t*) v138;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v140 = __riscv_vle8_v_i8m1(v139, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v141 = __riscv_vwmul_vv_i16m2(v136, v140, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v142 = v128;
      vint32m1_t v143 = __riscv_vmv_v_x_i32m1(v142, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v144 = __riscv_vwredsum_vs_i16m2_i32m1(v141, v143, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v145 = __riscv_vmv_x_s_i32m1_i32(v144);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v128 = v145;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v146 = v128;
    float v147 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v147 + (float) v146 * (v126 * v127);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v148 = v6;
  v2[0] = v148;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f1_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 34;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v16 = __riscv_vsetvl_e8m2(32);
    for (size_t v17 = 0; v17 < 32; v17 += v16) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v18 = 32 - v17;
      size_t v19 = __riscv_vsetvl_e8m2(v18);
      const uint8_t* v20 = v10 + 2;
      const uint8_t* v21 = v20 + v17;
      const int8_t* v22 = (const int8_t*) v21;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v23 = __riscv_vle8_v_i8m2(v22, v19);
      const uint8_t* v24 = v12 + 2;
      const uint8_t* v25 = v24 + v17;
      const int8_t* v26 = (const int8_t*) v25;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v27 = __riscv_vle8_v_i8m2(v26, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v28 = __riscv_vwmul_vv_i16m4(v23, v27, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v29 = v15;
      vint32m1_t v30 = __riscv_vmv_v_x_i32m1(v29, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v31 = __riscv_vwredsum_vs_i16m4_i32m1(v28, v30, v19);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v32 = __riscv_vmv_x_s_i32m1_i32(v31);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v15 = v32;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v33 = v15;
    float v34 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v34 + (float) v33 * (v13 * v14);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v35 = v6;
  v2[0] = v35;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f1_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 34;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v11 = v8 * 34;
    const uint8_t* v12 = v4 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v13 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v14 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v15;
    v15 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v16 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v17 = v10 + 2;
    const uint8_t* v18 = v17 + 0;
    const int8_t* v19 = (const int8_t*) v18;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v20 = __riscv_vle8_v_i8m2(v19, v16);
    const uint8_t* v21 = v12 + 2;
    const uint8_t* v22 = v21 + 0;
    const int8_t* v23 = (const int8_t*) v22;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v24 = __riscv_vle8_v_i8m2(v23, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v25 = __riscv_vwmul_vv_i16m4(v20, v24, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v26 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v27 = __riscv_vwredsum_vs_i16m4_i32m1(v25, v26, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v28 = __riscv_vmv_x_s_i32m1_i32(v27);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v15 = v28;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v29 = v15;
    float v30 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v30 + (float) v29 * (v13 * v14);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v31 = v6;
  v2[0] = v31;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f2_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e8m2(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v25 = __riscv_vle8_v_i8m2(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v29 = __riscv_vle8_v_i8m2(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v30 = __riscv_vwmul_vv_i16m4(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16m4_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v44 = __riscv_vsetvl_e8m2(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e8m2(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v51 = __riscv_vle8_v_i8m2(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v55 = __riscv_vle8_v_i8m2(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v56 = __riscv_vwmul_vv_i16m4(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16m4_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v61 = v17;
    float v62 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v62 + (float) v61 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v63 = v43;
    float v64 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v64 + (float) v63 * (v41 * v42);
  }
  for (size_t v65 = v9; v65 < v7; v65 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v66 = v65 * 34;
    const uint8_t* v67 = v3 + v66;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v68 = v65 * 34;
    const uint8_t* v69 = v4 + v68;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v70 = (float)*(const _Float16 *)(v67);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v71 = (float)*(const _Float16 *)(v69);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v72;
    v72 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v73 = __riscv_vsetvl_e8m2(32);
    for (size_t v74 = 0; v74 < 32; v74 += v73) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v75 = 32 - v74;
      size_t v76 = __riscv_vsetvl_e8m2(v75);
      const uint8_t* v77 = v67 + 2;
      const uint8_t* v78 = v77 + v74;
      const int8_t* v79 = (const int8_t*) v78;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v80 = __riscv_vle8_v_i8m2(v79, v76);
      const uint8_t* v81 = v69 + 2;
      const uint8_t* v82 = v81 + v74;
      const int8_t* v83 = (const int8_t*) v82;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v84 = __riscv_vle8_v_i8m2(v83, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v85 = __riscv_vwmul_vv_i16m4(v80, v84, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v86 = v72;
      vint32m1_t v87 = __riscv_vmv_v_x_i32m1(v86, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v88 = __riscv_vwredsum_vs_i16m4_i32m1(v85, v87, v76);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v89 = __riscv_vmv_x_s_i32m1_i32(v88);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v72 = v89;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v90 = v72;
    float v91 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v91 + (float) v90 * (v70 * v71);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v92 = v6;
  v2[0] = v92;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f2_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 2;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = v19 + 0;
    const int8_t* v21 = (const int8_t*) v20;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v22 = __riscv_vle8_v_i8m2(v21, v18);
    const uint8_t* v23 = v14 + 2;
    const uint8_t* v24 = v23 + 0;
    const int8_t* v25 = (const int8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v26 = __riscv_vle8_v_i8m2(v25, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v27 = __riscv_vwmul_vv_i16m4(v22, v26, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v28 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v29 = __riscv_vwredsum_vs_i16m4_i32m1(v27, v28, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v30 = __riscv_vmv_x_s_i32m1_i32(v29);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v17 = v30;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v31 = v10 + 1;
    size_t v32 = v31 * 34;
    const uint8_t* v33 = v3 + v32;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v34 = v10 + 1;
    size_t v35 = v34 * 34;
    const uint8_t* v36 = v4 + v35;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v37 = (float)*(const _Float16 *)(v33);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v36);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v39;
    v39 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v40 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v41 = v33 + 2;
    const uint8_t* v42 = v41 + 0;
    const int8_t* v43 = (const int8_t*) v42;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v44 = __riscv_vle8_v_i8m2(v43, v40);
    const uint8_t* v45 = v36 + 2;
    const uint8_t* v46 = v45 + 0;
    const int8_t* v47 = (const int8_t*) v46;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v48 = __riscv_vle8_v_i8m2(v47, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v49 = __riscv_vwmul_vv_i16m4(v44, v48, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v50 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v51 = __riscv_vwredsum_vs_i16m4_i32m1(v49, v50, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v39 = v52;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v53 = v17;
    float v54 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v54 + (float) v53 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v55 = v39;
    float v56 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v56 + (float) v55 * (v37 * v38);
  }
  for (size_t v57 = v9; v57 < v7; v57 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v58 = v57 * 34;
    const uint8_t* v59 = v3 + v58;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v60 = v57 * 34;
    const uint8_t* v61 = v4 + v60;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v62 = (float)*(const _Float16 *)(v59);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v63 = (float)*(const _Float16 *)(v61);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v64;
    v64 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v65 = __riscv_vsetvl_e8m2(32);
    for (size_t v66 = 0; v66 < 32; v66 += v65) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v67 = 32 - v66;
      size_t v68 = __riscv_vsetvl_e8m2(v67);
      const uint8_t* v69 = v59 + 2;
      const uint8_t* v70 = v69 + v66;
      const int8_t* v71 = (const int8_t*) v70;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v72 = __riscv_vle8_v_i8m2(v71, v68);
      const uint8_t* v73 = v61 + 2;
      const uint8_t* v74 = v73 + v66;
      const int8_t* v75 = (const int8_t*) v74;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v76 = __riscv_vle8_v_i8m2(v75, v68);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v77 = __riscv_vwmul_vv_i16m4(v72, v76, v68);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v78 = v64;
      vint32m1_t v79 = __riscv_vmv_v_x_i32m1(v78, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v80 = __riscv_vwredsum_vs_i16m4_i32m1(v77, v79, v68);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v81 = __riscv_vmv_x_s_i32m1_i32(v80);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v64 = v81;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v82 = v64;
    float v83 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v83 + (float) v82 * (v62 * v63);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v84 = v6;
  v2[0] = v84;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f4_robust(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    for (size_t v19 = 0; v19 < 32; v19 += v18) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v20 = 32 - v19;
      size_t v21 = __riscv_vsetvl_e8m2(v20);
      const uint8_t* v22 = v12 + 2;
      const uint8_t* v23 = v22 + v19;
      const int8_t* v24 = (const int8_t*) v23;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v25 = __riscv_vle8_v_i8m2(v24, v21);
      const uint8_t* v26 = v14 + 2;
      const uint8_t* v27 = v26 + v19;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v29 = __riscv_vle8_v_i8m2(v28, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v30 = __riscv_vwmul_vv_i16m4(v25, v29, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v31 = v17;
      vint32m1_t v32 = __riscv_vmv_v_x_i32m1(v31, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v33 = __riscv_vwredsum_vs_i16m4_i32m1(v30, v32, v21);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v34 = __riscv_vmv_x_s_i32m1_i32(v33);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v17 = v34;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v35 = v10 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v3 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v38 = v10 + 1;
    size_t v39 = v38 * 34;
    const uint8_t* v40 = v4 + v39;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v41 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v42 = (float)*(const _Float16 *)(v40);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v43;
    v43 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v44 = __riscv_vsetvl_e8m2(32);
    for (size_t v45 = 0; v45 < 32; v45 += v44) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v46 = 32 - v45;
      size_t v47 = __riscv_vsetvl_e8m2(v46);
      const uint8_t* v48 = v37 + 2;
      const uint8_t* v49 = v48 + v45;
      const int8_t* v50 = (const int8_t*) v49;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v51 = __riscv_vle8_v_i8m2(v50, v47);
      const uint8_t* v52 = v40 + 2;
      const uint8_t* v53 = v52 + v45;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v55 = __riscv_vle8_v_i8m2(v54, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v56 = __riscv_vwmul_vv_i16m4(v51, v55, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v57 = v43;
      vint32m1_t v58 = __riscv_vmv_v_x_i32m1(v57, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v59 = __riscv_vwredsum_vs_i16m4_i32m1(v56, v58, v47);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v60 = __riscv_vmv_x_s_i32m1_i32(v59);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v43 = v60;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v61 = v10 + 2;
    size_t v62 = v61 * 34;
    const uint8_t* v63 = v3 + v62;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v64 = v10 + 2;
    size_t v65 = v64 * 34;
    const uint8_t* v66 = v4 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v67 = (float)*(const _Float16 *)(v63);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v68 = (float)*(const _Float16 *)(v66);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v69;
    v69 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v70 = __riscv_vsetvl_e8m2(32);
    for (size_t v71 = 0; v71 < 32; v71 += v70) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v72 = 32 - v71;
      size_t v73 = __riscv_vsetvl_e8m2(v72);
      const uint8_t* v74 = v63 + 2;
      const uint8_t* v75 = v74 + v71;
      const int8_t* v76 = (const int8_t*) v75;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v77 = __riscv_vle8_v_i8m2(v76, v73);
      const uint8_t* v78 = v66 + 2;
      const uint8_t* v79 = v78 + v71;
      const int8_t* v80 = (const int8_t*) v79;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v81 = __riscv_vle8_v_i8m2(v80, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v82 = __riscv_vwmul_vv_i16m4(v77, v81, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v83 = v69;
      vint32m1_t v84 = __riscv_vmv_v_x_i32m1(v83, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v85 = __riscv_vwredsum_vs_i16m4_i32m1(v82, v84, v73);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v86 = __riscv_vmv_x_s_i32m1_i32(v85);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v69 = v86;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v87 = v10 + 3;
    size_t v88 = v87 * 34;
    const uint8_t* v89 = v3 + v88;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v90 = v10 + 3;
    size_t v91 = v90 * 34;
    const uint8_t* v92 = v4 + v91;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v93 = (float)*(const _Float16 *)(v89);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v94 = (float)*(const _Float16 *)(v92);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v95;
    v95 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v96 = __riscv_vsetvl_e8m2(32);
    for (size_t v97 = 0; v97 < 32; v97 += v96) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v98 = 32 - v97;
      size_t v99 = __riscv_vsetvl_e8m2(v98);
      const uint8_t* v100 = v89 + 2;
      const uint8_t* v101 = v100 + v97;
      const int8_t* v102 = (const int8_t*) v101;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v103 = __riscv_vle8_v_i8m2(v102, v99);
      const uint8_t* v104 = v92 + 2;
      const uint8_t* v105 = v104 + v97;
      const int8_t* v106 = (const int8_t*) v105;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v107 = __riscv_vle8_v_i8m2(v106, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v108 = __riscv_vwmul_vv_i16m4(v103, v107, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v109 = v95;
      vint32m1_t v110 = __riscv_vmv_v_x_i32m1(v109, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v111 = __riscv_vwredsum_vs_i16m4_i32m1(v108, v110, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v112 = __riscv_vmv_x_s_i32m1_i32(v111);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v95 = v112;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v113 = v17;
    float v114 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v114 + (float) v113 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v115 = v43;
    float v116 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v116 + (float) v115 * (v41 * v42);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v117 = v69;
    float v118 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v118 + (float) v117 * (v67 * v68);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v119 = v95;
    float v120 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v120 + (float) v119 * (v93 * v94);
  }
  for (size_t v121 = v9; v121 < v7; v121 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v122 = v121 * 34;
    const uint8_t* v123 = v3 + v122;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v124 = v121 * 34;
    const uint8_t* v125 = v4 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v126 = (float)*(const _Float16 *)(v123);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v127 = (float)*(const _Float16 *)(v125);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v128;
    v128 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v129 = __riscv_vsetvl_e8m2(32);
    for (size_t v130 = 0; v130 < 32; v130 += v129) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v131 = 32 - v130;
      size_t v132 = __riscv_vsetvl_e8m2(v131);
      const uint8_t* v133 = v123 + 2;
      const uint8_t* v134 = v133 + v130;
      const int8_t* v135 = (const int8_t*) v134;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v136 = __riscv_vle8_v_i8m2(v135, v132);
      const uint8_t* v137 = v125 + 2;
      const uint8_t* v138 = v137 + v130;
      const int8_t* v139 = (const int8_t*) v138;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v140 = __riscv_vle8_v_i8m2(v139, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v141 = __riscv_vwmul_vv_i16m4(v136, v140, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v142 = v128;
      vint32m1_t v143 = __riscv_vmv_v_x_i32m1(v142, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v144 = __riscv_vwredsum_vs_i16m4_i32m1(v141, v143, v132);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v145 = __riscv_vmv_x_s_i32m1_i32(v144);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v128 = v145;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v146 = v128;
    float v147 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v147 + (float) v146 * (v126 * v127);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v148 = v6;
  v2[0] = v148;
  return;
}



#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0_m2_f4_elided(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v7 = v1 / 32;
  size_t v8 = v7 % 4;
  size_t v9 = v7 - v8;
  for (size_t v10 = 0; v10 < v9; v10 += 4) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v11 = v10 * 34;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v10 * 34;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v15 = (float)*(const _Float16 *)(v12);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v16 = (float)*(const _Float16 *)(v14);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v17;
    v17 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v18 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v19 = v12 + 2;
    const uint8_t* v20 = v19 + 0;
    const int8_t* v21 = (const int8_t*) v20;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v22 = __riscv_vle8_v_i8m2(v21, v18);
    const uint8_t* v23 = v14 + 2;
    const uint8_t* v24 = v23 + 0;
    const int8_t* v25 = (const int8_t*) v24;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v26 = __riscv_vle8_v_i8m2(v25, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v27 = __riscv_vwmul_vv_i16m4(v22, v26, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v28 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v29 = __riscv_vwredsum_vs_i16m4_i32m1(v27, v28, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v30 = __riscv_vmv_x_s_i32m1_i32(v29);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v17 = v30;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v31 = v10 + 1;
    size_t v32 = v31 * 34;
    const uint8_t* v33 = v3 + v32;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v34 = v10 + 1;
    size_t v35 = v34 * 34;
    const uint8_t* v36 = v4 + v35;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v37 = (float)*(const _Float16 *)(v33);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v36);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v39;
    v39 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v40 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v41 = v33 + 2;
    const uint8_t* v42 = v41 + 0;
    const int8_t* v43 = (const int8_t*) v42;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v44 = __riscv_vle8_v_i8m2(v43, v40);
    const uint8_t* v45 = v36 + 2;
    const uint8_t* v46 = v45 + 0;
    const int8_t* v47 = (const int8_t*) v46;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v48 = __riscv_vle8_v_i8m2(v47, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v49 = __riscv_vwmul_vv_i16m4(v44, v48, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v50 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v51 = __riscv_vwredsum_vs_i16m4_i32m1(v49, v50, v40);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v39 = v52;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v53 = v10 + 2;
    size_t v54 = v53 * 34;
    const uint8_t* v55 = v3 + v54;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v56 = v10 + 2;
    size_t v57 = v56 * 34;
    const uint8_t* v58 = v4 + v57;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v59 = (float)*(const _Float16 *)(v55);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v60 = (float)*(const _Float16 *)(v58);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v61;
    v61 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v62 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v63 = v55 + 2;
    const uint8_t* v64 = v63 + 0;
    const int8_t* v65 = (const int8_t*) v64;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v66 = __riscv_vle8_v_i8m2(v65, v62);
    const uint8_t* v67 = v58 + 2;
    const uint8_t* v68 = v67 + 0;
    const int8_t* v69 = (const int8_t*) v68;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v70 = __riscv_vle8_v_i8m2(v69, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v71 = __riscv_vwmul_vv_i16m4(v66, v70, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v72 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v73 = __riscv_vwredsum_vs_i16m4_i32m1(v71, v72, v62);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v74 = __riscv_vmv_x_s_i32m1_i32(v73);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v61 = v74;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v75 = v10 + 3;
    size_t v76 = v75 * 34;
    const uint8_t* v77 = v3 + v76;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v78 = v10 + 3;
    size_t v79 = v78 * 34;
    const uint8_t* v80 = v4 + v79;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v81 = (float)*(const _Float16 *)(v77);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v82 = (float)*(const _Float16 *)(v80);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v83;
    v83 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v84 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v85 = v77 + 2;
    const uint8_t* v86 = v85 + 0;
    const int8_t* v87 = (const int8_t*) v86;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v88 = __riscv_vle8_v_i8m2(v87, v84);
    const uint8_t* v89 = v80 + 2;
    const uint8_t* v90 = v89 + 0;
    const int8_t* v91 = (const int8_t*) v90;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
    vint8m2_t v92 = __riscv_vle8_v_i8m2(v91, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
    vint16m4_t v93 = __riscv_vwmul_vv_i16m4(v88, v92, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v94 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
    vint32m1_t v95 = __riscv_vwredsum_vs_i16m4_i32m1(v93, v94, v84);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v96 = __riscv_vmv_x_s_i32m1_i32(v95);
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v83 = v96;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v97 = v17;
    float v98 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v98 + (float) v97 * (v15 * v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v99 = v39;
    float v100 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v100 + (float) v99 * (v37 * v38);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v101 = v61;
    float v102 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v102 + (float) v101 * (v59 * v60);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v103 = v83;
    float v104 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v104 + (float) v103 * (v81 * v82);
  }
  for (size_t v105 = v9; v105 < v7; v105 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v106 = v105 * 34;
    const uint8_t* v107 = v3 + v106;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v108 = v105 * 34;
    const uint8_t* v109 = v4 + v108;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v110 = (float)*(const _Float16 *)(v107);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v111 = (float)*(const _Float16 *)(v109);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v112;
    v112 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v113 = __riscv_vsetvl_e8m2(32);
    for (size_t v114 = 0; v114 < 32; v114 += v113) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
      size_t v115 = 32 - v114;
      size_t v116 = __riscv_vsetvl_e8m2(v115);
      const uint8_t* v117 = v107 + 2;
      const uint8_t* v118 = v117 + v114;
      const int8_t* v119 = (const int8_t*) v118;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v120 = __riscv_vle8_v_i8m2(v119, v116);
      const uint8_t* v121 = v109 + 2;
      const uint8_t* v122 = v121 + v114;
      const int8_t* v123 = (const int8_t*) v122;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m2
      vint8m2_t v124 = __riscv_vle8_v_i8m2(v123, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m4
      vint16m4_t v125 = __riscv_vwmul_vv_i16m4(v120, v124, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v126 = v112;
      vint32m1_t v127 = __riscv_vmv_v_x_i32m1(v126, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m4_i32m1
      vint32m1_t v128 = __riscv_vwredsum_vs_i16m4_i32m1(v125, v127, v116);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v129 = __riscv_vmv_x_s_i32m1_i32(v128);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v112 = v129;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v130 = v112;
    float v131 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v131 + (float) v130 * (v110 * v111);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v132 = v6;
  v2[0] = v132;
  return;
}


