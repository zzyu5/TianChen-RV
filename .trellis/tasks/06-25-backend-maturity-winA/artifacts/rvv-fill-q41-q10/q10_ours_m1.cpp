#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 128;
  for (size_t v8 = 0; v8 < v7; v8 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v9 = v8 * 18;
    const uint8_t* v10 = v3 + v9;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v11 = (float)*(const _Float16 *)(v10);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    float v12;
    v12 = 0.0f;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v13 = v8 * 4;
    size_t v14 = v13 + 0;
    size_t v15 = v14 * 34;
    const uint8_t* v16 = v4 + v15;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v17 = (float)*(const _Float16 *)(v16);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v18 = __riscv_vsetvl_e8m1(32);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_addr
    const uint8_t* v19 = v10 + 2;
    const uint8_t* v20 = (const uint8_t*) v19;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlm_v_b8
    vbool8_t v21 = __riscv_vlm_v_b8(v20, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_addr
    const uint8_t* v22 = v16 + 2;
    const int8_t* v23 = (const int8_t*) v22;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v24 = __riscv_vle8_v_i8m1(v23, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v25 = __riscv_vneg_v_i8m1(v24, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v26 = __riscv_vmerge_vvm_i8m1(v25, v24, v21, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
    vint16m1_t v27 = __riscv_vmv_v_x_i16m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
    vint16m1_t v28 = __riscv_vwredsum_vs_i8m1_i16m1(v26, v27, v18);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
    int16_t v29 = __riscv_vmv_x_s_i16m1_i16(v28);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v30;
    int32_t v31 = (int32_t) v29;
    v30 = v31;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v32 = v30;
    float v33 = v12;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v12 = v33 + v17 * (float) v32;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v34 = v8 * 4;
    size_t v35 = v34 + 1;
    size_t v36 = v35 * 34;
    const uint8_t* v37 = v4 + v36;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v38 = (float)*(const _Float16 *)(v37);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v39 = __riscv_vsetvl_e8m1(32);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_addr
    const uint8_t* v40 = v10 + 6;
    const uint8_t* v41 = (const uint8_t*) v40;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlm_v_b8
    vbool8_t v42 = __riscv_vlm_v_b8(v41, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_addr
    const uint8_t* v43 = v37 + 2;
    const int8_t* v44 = (const int8_t*) v43;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v45 = __riscv_vle8_v_i8m1(v44, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v46 = __riscv_vneg_v_i8m1(v45, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v47 = __riscv_vmerge_vvm_i8m1(v46, v45, v42, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
    vint16m1_t v48 = __riscv_vmv_v_x_i16m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
    vint16m1_t v49 = __riscv_vwredsum_vs_i8m1_i16m1(v47, v48, v39);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
    int16_t v50 = __riscv_vmv_x_s_i16m1_i16(v49);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v51;
    int32_t v52 = (int32_t) v50;
    v51 = v52;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v53 = v51;
    float v54 = v12;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v12 = v54 + v38 * (float) v53;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v55 = v8 * 4;
    size_t v56 = v55 + 2;
    size_t v57 = v56 * 34;
    const uint8_t* v58 = v4 + v57;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v59 = (float)*(const _Float16 *)(v58);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v60 = __riscv_vsetvl_e8m1(32);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_addr
    const uint8_t* v61 = v10 + 10;
    const uint8_t* v62 = (const uint8_t*) v61;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlm_v_b8
    vbool8_t v63 = __riscv_vlm_v_b8(v62, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_addr
    const uint8_t* v64 = v58 + 2;
    const int8_t* v65 = (const int8_t*) v64;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v66 = __riscv_vle8_v_i8m1(v65, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v67 = __riscv_vneg_v_i8m1(v66, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v68 = __riscv_vmerge_vvm_i8m1(v67, v66, v63, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
    vint16m1_t v69 = __riscv_vmv_v_x_i16m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
    vint16m1_t v70 = __riscv_vwredsum_vs_i8m1_i16m1(v68, v69, v60);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
    int16_t v71 = __riscv_vmv_x_s_i16m1_i16(v70);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v72;
    int32_t v73 = (int32_t) v71;
    v72 = v73;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v74 = v72;
    float v75 = v12;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v12 = v75 + v59 * (float) v74;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v76 = v8 * 4;
    size_t v77 = v76 + 3;
    size_t v78 = v77 * 34;
    const uint8_t* v79 = v4 + v78;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v80 = (float)*(const _Float16 *)(v79);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v81 = __riscv_vsetvl_e8m1(32);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_addr
    const uint8_t* v82 = v10 + 14;
    const uint8_t* v83 = (const uint8_t*) v82;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vlm_v_b8
    vbool8_t v84 = __riscv_vlm_v_b8(v83, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_block_addr
    const uint8_t* v85 = v79 + 2;
    const int8_t* v86 = (const int8_t*) v85;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v87 = __riscv_vle8_v_i8m1(v86, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i8m1
    vint8m1_t v88 = __riscv_vneg_v_i8m1(v87, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i8m1
    vint8m1_t v89 = __riscv_vmerge_vvm_i8m1(v88, v87, v84, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
    vint16m1_t v90 = __riscv_vmv_v_x_i16m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i8m1_i16m1
    vint16m1_t v91 = __riscv_vwredsum_vs_i8m1_i16m1(v89, v90, v81);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i16m1_i16
    int16_t v92 = __riscv_vmv_x_s_i16m1_i16(v91);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v93;
    int32_t v94 = (int32_t) v92;
    v93 = v94;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v95 = v93;
    float v96 = v12;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v12 = v96 + v80 * (float) v95;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v97 = v12;
    float v98 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v98 + v11 * v97;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v99 = v6;
  v2[0] = v99;
  return;
}


