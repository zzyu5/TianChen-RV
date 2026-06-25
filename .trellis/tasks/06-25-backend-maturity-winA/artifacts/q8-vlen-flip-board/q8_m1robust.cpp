#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q8_kernel_m1robust(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v9 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v10;
  v10 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v11 = v1 / 32;
  size_t v12 = v11 % 2;
  size_t v13 = v11 - v12;
  for (size_t v14 = 0; v14 < v13; v14 += 2) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v15 = v14 * 34;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v17 = v14 * 34;
    const uint8_t* v18 = v6 + v17;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v19 = (float)*(const _Float16 *)(v16);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v20 = (float)*(const _Float16 *)(v18);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v21;
    v21 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v22 = __riscv_vsetvl_e8m1(32);
    for (size_t v23 = 0; v23 < 32; v23 += v22) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v24 = 32 - v23;
      size_t v25 = __riscv_vsetvl_e8m1(v24);
      const uint8_t* v26 = v16 + 2;
      const uint8_t* v27 = v26 + v23;
      const int8_t* v28 = (const int8_t*) v27;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v25);
      const uint8_t* v30 = v18 + 2;
      const uint8_t* v31 = v30 + v23;
      const int8_t* v32 = (const int8_t*) v31;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v33 = __riscv_vle8_v_i8m1(v32, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v34 = __riscv_vwmul_vv_i16m2(v29, v33, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v35 = v21;
      vint32m1_t v36 = __riscv_vmv_v_x_i32m1(v35, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v37 = __riscv_vwredsum_vs_i16m2_i32m1(v34, v36, v25);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v38 = __riscv_vmv_x_s_i32m1_i32(v37);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v21 = v38;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v39 = v14 + 1;
    size_t v40 = v39 * 34;
    const uint8_t* v41 = v4 + v40;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v42 = v14 + 1;
    size_t v43 = v42 * 34;
    const uint8_t* v44 = v6 + v43;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v45 = (float)*(const _Float16 *)(v41);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v46 = (float)*(const _Float16 *)(v44);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v47;
    v47 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v48 = __riscv_vsetvl_e8m1(32);
    for (size_t v49 = 0; v49 < 32; v49 += v48) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v50 = 32 - v49;
      size_t v51 = __riscv_vsetvl_e8m1(v50);
      const uint8_t* v52 = v41 + 2;
      const uint8_t* v53 = v52 + v49;
      const int8_t* v54 = (const int8_t*) v53;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v55 = __riscv_vle8_v_i8m1(v54, v51);
      const uint8_t* v56 = v44 + 2;
      const uint8_t* v57 = v56 + v49;
      const int8_t* v58 = (const int8_t*) v57;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v59 = __riscv_vle8_v_i8m1(v58, v51);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v60 = __riscv_vwmul_vv_i16m2(v55, v59, v51);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v61 = v47;
      vint32m1_t v62 = __riscv_vmv_v_x_i32m1(v61, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v63 = __riscv_vwredsum_vs_i16m2_i32m1(v60, v62, v51);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v64 = __riscv_vmv_x_s_i32m1_i32(v63);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v47 = v64;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v65 = v21;
    float v66 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v66 + (float) v65 * (v19 * v20);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v67 = v47;
    float v68 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v68 + (float) v67 * (v45 * v46);
  }
  for (size_t v69 = v13; v69 < v11; v69 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v70 = v69 * 34;
    const uint8_t* v71 = v4 + v70;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v72 = v69 * 34;
    const uint8_t* v73 = v6 + v72;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v74 = (float)*(const _Float16 *)(v71);
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v75 = (float)*(const _Float16 *)(v73);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v76;
    v76 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v77 = __riscv_vsetvl_e8m1(32);
    for (size_t v78 = 0; v78 < 32; v78 += v77) {
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
      size_t v79 = 32 - v78;
      size_t v80 = __riscv_vsetvl_e8m1(v79);
      const uint8_t* v81 = v71 + 2;
      const uint8_t* v82 = v81 + v78;
      const int8_t* v83 = (const int8_t*) v82;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v84 = __riscv_vle8_v_i8m1(v83, v80);
      const uint8_t* v85 = v73 + 2;
      const uint8_t* v86 = v85 + v78;
      const int8_t* v87 = (const int8_t*) v86;
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
      vint8m1_t v88 = __riscv_vle8_v_i8m1(v87, v80);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
      vint16m2_t v89 = __riscv_vwmul_vv_i16m2(v84, v88, v80);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
      int32_t v90 = v76;
      vint32m1_t v91 = __riscv_vmv_v_x_i32m1(v90, 1);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
      vint32m1_t v92 = __riscv_vwredsum_vs_i16m2_i32m1(v89, v91, v80);
      // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
      int32_t v93 = __riscv_vmv_x_s_i32m1_i32(v92);
      // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
      v76 = v93;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    int32_t v94 = v76;
    float v95 = v10;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v10 = v95 + (float) v94 * (v74 * v75);
  }
  // tcrv_emitc.source_op=tcrv_rvv.q8_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v96 = v10;
  v2[0] = v96;
  return;
}


