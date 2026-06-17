#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q1_0_q8_0_kernel_ggml_vec_dot_q1_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  static const uint8_t tcrv_q1_0_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v6;
  v6 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 128;
  // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=kmask_table_load
  vuint8m1_t v8 = __riscv_vle8_v_u8m1(tcrv_q1_0_kmask, 8);
  for (size_t v9 = 0; v9 < v7; v9 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_x
    size_t v10 = v9 * 18;
    const uint8_t* v11 = v3 + v10;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v12 = (float)*(const _Float16 *)(v11);
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    float v13;
    v13 = 0.0f;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v14 = v9 * 4;
    size_t v15 = v14 + 0;
    size_t v16 = v15 * 34;
    const uint8_t* v17 = v4 + v16;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v18 = (float)*(const _Float16 *)(v17);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v19;
    v19 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v20 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v21 = v11 + 2;
    const uint8_t* v22 = (const uint8_t*) v21;
    const uint8_t v23 = v22[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v24 = __riscv_vmv_v_x_u8m1(v23, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v25 = __riscv_vand_vv_u8m1(v24, v8, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v26 = __riscv_vmsne_vx_u8m1_b8(v25, 0, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v27 = v17 + 2;
    const int8_t* v28 = (const int8_t*) v27;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v30 = __riscv_vwcvt_x_x_v_i16m2(v29, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v31 = __riscv_vneg_v_i16m2(v30, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v32 = __riscv_vmerge_vvm_i16m2(v31, v30, v26, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v33 = v19;
    vint32m1_t v34 = __riscv_vmv_v_x_i32m1(v33, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v35 = __riscv_vwredsum_vs_i16m2_i32m1(v32, v34, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v36 = __riscv_vmv_x_s_i32m1_i32(v35);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v19 = v36;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v37 = v11 + 3;
    const uint8_t* v38 = (const uint8_t*) v37;
    const uint8_t v39 = v38[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v40 = __riscv_vmv_v_x_u8m1(v39, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v41 = __riscv_vand_vv_u8m1(v40, v8, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v42 = __riscv_vmsne_vx_u8m1_b8(v41, 0, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v43 = v17 + 10;
    const int8_t* v44 = (const int8_t*) v43;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v45 = __riscv_vle8_v_i8m1(v44, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v46 = __riscv_vwcvt_x_x_v_i16m2(v45, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v47 = __riscv_vneg_v_i16m2(v46, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v48 = __riscv_vmerge_vvm_i16m2(v47, v46, v42, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v49 = v19;
    vint32m1_t v50 = __riscv_vmv_v_x_i32m1(v49, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v51 = __riscv_vwredsum_vs_i16m2_i32m1(v48, v50, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v52 = __riscv_vmv_x_s_i32m1_i32(v51);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v19 = v52;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v53 = v11 + 4;
    const uint8_t* v54 = (const uint8_t*) v53;
    const uint8_t v55 = v54[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v56 = __riscv_vmv_v_x_u8m1(v55, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v57 = __riscv_vand_vv_u8m1(v56, v8, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v58 = __riscv_vmsne_vx_u8m1_b8(v57, 0, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v59 = v17 + 18;
    const int8_t* v60 = (const int8_t*) v59;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v61 = __riscv_vle8_v_i8m1(v60, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v62 = __riscv_vwcvt_x_x_v_i16m2(v61, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v63 = __riscv_vneg_v_i16m2(v62, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v64 = __riscv_vmerge_vvm_i16m2(v63, v62, v58, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v65 = v19;
    vint32m1_t v66 = __riscv_vmv_v_x_i32m1(v65, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v67 = __riscv_vwredsum_vs_i16m2_i32m1(v64, v66, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v68 = __riscv_vmv_x_s_i32m1_i32(v67);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v19 = v68;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v69 = v11 + 5;
    const uint8_t* v70 = (const uint8_t*) v69;
    const uint8_t v71 = v70[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v72 = __riscv_vmv_v_x_u8m1(v71, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v73 = __riscv_vand_vv_u8m1(v72, v8, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v74 = __riscv_vmsne_vx_u8m1_b8(v73, 0, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v75 = v17 + 26;
    const int8_t* v76 = (const int8_t*) v75;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v77 = __riscv_vle8_v_i8m1(v76, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v78 = __riscv_vwcvt_x_x_v_i16m2(v77, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v79 = __riscv_vneg_v_i16m2(v78, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v80 = __riscv_vmerge_vvm_i16m2(v79, v78, v74, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v81 = v19;
    vint32m1_t v82 = __riscv_vmv_v_x_i32m1(v81, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v83 = __riscv_vwredsum_vs_i16m2_i32m1(v80, v82, v20);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v84 = __riscv_vmv_x_s_i32m1_i32(v83);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v19 = v84;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v85 = v19;
    float v86 = v13;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v86 + v18 * (float) v85;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v87 = v9 * 4;
    size_t v88 = v87 + 1;
    size_t v89 = v88 * 34;
    const uint8_t* v90 = v4 + v89;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v91 = (float)*(const _Float16 *)(v90);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v92;
    v92 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v93 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v94 = v11 + 6;
    const uint8_t* v95 = (const uint8_t*) v94;
    const uint8_t v96 = v95[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v97 = __riscv_vmv_v_x_u8m1(v96, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v98 = __riscv_vand_vv_u8m1(v97, v8, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v99 = __riscv_vmsne_vx_u8m1_b8(v98, 0, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v100 = v90 + 2;
    const int8_t* v101 = (const int8_t*) v100;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v102 = __riscv_vle8_v_i8m1(v101, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v103 = __riscv_vwcvt_x_x_v_i16m2(v102, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v104 = __riscv_vneg_v_i16m2(v103, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v105 = __riscv_vmerge_vvm_i16m2(v104, v103, v99, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v106 = v92;
    vint32m1_t v107 = __riscv_vmv_v_x_i32m1(v106, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v108 = __riscv_vwredsum_vs_i16m2_i32m1(v105, v107, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v109 = __riscv_vmv_x_s_i32m1_i32(v108);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v92 = v109;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v110 = v11 + 7;
    const uint8_t* v111 = (const uint8_t*) v110;
    const uint8_t v112 = v111[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v113 = __riscv_vmv_v_x_u8m1(v112, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v114 = __riscv_vand_vv_u8m1(v113, v8, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v115 = __riscv_vmsne_vx_u8m1_b8(v114, 0, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v116 = v90 + 10;
    const int8_t* v117 = (const int8_t*) v116;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v118 = __riscv_vle8_v_i8m1(v117, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v119 = __riscv_vwcvt_x_x_v_i16m2(v118, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v120 = __riscv_vneg_v_i16m2(v119, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v121 = __riscv_vmerge_vvm_i16m2(v120, v119, v115, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v122 = v92;
    vint32m1_t v123 = __riscv_vmv_v_x_i32m1(v122, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v124 = __riscv_vwredsum_vs_i16m2_i32m1(v121, v123, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v125 = __riscv_vmv_x_s_i32m1_i32(v124);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v92 = v125;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v126 = v11 + 8;
    const uint8_t* v127 = (const uint8_t*) v126;
    const uint8_t v128 = v127[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v129 = __riscv_vmv_v_x_u8m1(v128, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v130 = __riscv_vand_vv_u8m1(v129, v8, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v131 = __riscv_vmsne_vx_u8m1_b8(v130, 0, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v132 = v90 + 18;
    const int8_t* v133 = (const int8_t*) v132;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v134 = __riscv_vle8_v_i8m1(v133, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v135 = __riscv_vwcvt_x_x_v_i16m2(v134, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v136 = __riscv_vneg_v_i16m2(v135, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v137 = __riscv_vmerge_vvm_i16m2(v136, v135, v131, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v138 = v92;
    vint32m1_t v139 = __riscv_vmv_v_x_i32m1(v138, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v140 = __riscv_vwredsum_vs_i16m2_i32m1(v137, v139, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v141 = __riscv_vmv_x_s_i32m1_i32(v140);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v92 = v141;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v142 = v11 + 9;
    const uint8_t* v143 = (const uint8_t*) v142;
    const uint8_t v144 = v143[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v145 = __riscv_vmv_v_x_u8m1(v144, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v146 = __riscv_vand_vv_u8m1(v145, v8, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v147 = __riscv_vmsne_vx_u8m1_b8(v146, 0, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v148 = v90 + 26;
    const int8_t* v149 = (const int8_t*) v148;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v150 = __riscv_vle8_v_i8m1(v149, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v151 = __riscv_vwcvt_x_x_v_i16m2(v150, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v152 = __riscv_vneg_v_i16m2(v151, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v153 = __riscv_vmerge_vvm_i16m2(v152, v151, v147, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v154 = v92;
    vint32m1_t v155 = __riscv_vmv_v_x_i32m1(v154, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v156 = __riscv_vwredsum_vs_i16m2_i32m1(v153, v155, v93);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v157 = __riscv_vmv_x_s_i32m1_i32(v156);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v92 = v157;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v158 = v92;
    float v159 = v13;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v159 + v91 * (float) v158;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v160 = v9 * 4;
    size_t v161 = v160 + 2;
    size_t v162 = v161 * 34;
    const uint8_t* v163 = v4 + v162;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v164 = (float)*(const _Float16 *)(v163);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v165;
    v165 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v166 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v167 = v11 + 10;
    const uint8_t* v168 = (const uint8_t*) v167;
    const uint8_t v169 = v168[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v170 = __riscv_vmv_v_x_u8m1(v169, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v171 = __riscv_vand_vv_u8m1(v170, v8, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v172 = __riscv_vmsne_vx_u8m1_b8(v171, 0, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v173 = v163 + 2;
    const int8_t* v174 = (const int8_t*) v173;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v175 = __riscv_vle8_v_i8m1(v174, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v176 = __riscv_vwcvt_x_x_v_i16m2(v175, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v177 = __riscv_vneg_v_i16m2(v176, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v178 = __riscv_vmerge_vvm_i16m2(v177, v176, v172, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v179 = v165;
    vint32m1_t v180 = __riscv_vmv_v_x_i32m1(v179, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v181 = __riscv_vwredsum_vs_i16m2_i32m1(v178, v180, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v182 = __riscv_vmv_x_s_i32m1_i32(v181);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v165 = v182;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v183 = v11 + 11;
    const uint8_t* v184 = (const uint8_t*) v183;
    const uint8_t v185 = v184[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v186 = __riscv_vmv_v_x_u8m1(v185, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v187 = __riscv_vand_vv_u8m1(v186, v8, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v188 = __riscv_vmsne_vx_u8m1_b8(v187, 0, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v189 = v163 + 10;
    const int8_t* v190 = (const int8_t*) v189;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v191 = __riscv_vle8_v_i8m1(v190, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v192 = __riscv_vwcvt_x_x_v_i16m2(v191, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v193 = __riscv_vneg_v_i16m2(v192, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v194 = __riscv_vmerge_vvm_i16m2(v193, v192, v188, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v195 = v165;
    vint32m1_t v196 = __riscv_vmv_v_x_i32m1(v195, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v197 = __riscv_vwredsum_vs_i16m2_i32m1(v194, v196, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v198 = __riscv_vmv_x_s_i32m1_i32(v197);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v165 = v198;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v199 = v11 + 12;
    const uint8_t* v200 = (const uint8_t*) v199;
    const uint8_t v201 = v200[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v202 = __riscv_vmv_v_x_u8m1(v201, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v203 = __riscv_vand_vv_u8m1(v202, v8, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v204 = __riscv_vmsne_vx_u8m1_b8(v203, 0, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v205 = v163 + 18;
    const int8_t* v206 = (const int8_t*) v205;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v207 = __riscv_vle8_v_i8m1(v206, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v208 = __riscv_vwcvt_x_x_v_i16m2(v207, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v209 = __riscv_vneg_v_i16m2(v208, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v210 = __riscv_vmerge_vvm_i16m2(v209, v208, v204, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v211 = v165;
    vint32m1_t v212 = __riscv_vmv_v_x_i32m1(v211, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v213 = __riscv_vwredsum_vs_i16m2_i32m1(v210, v212, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v214 = __riscv_vmv_x_s_i32m1_i32(v213);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v165 = v214;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v215 = v11 + 13;
    const uint8_t* v216 = (const uint8_t*) v215;
    const uint8_t v217 = v216[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v218 = __riscv_vmv_v_x_u8m1(v217, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v219 = __riscv_vand_vv_u8m1(v218, v8, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v220 = __riscv_vmsne_vx_u8m1_b8(v219, 0, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v221 = v163 + 26;
    const int8_t* v222 = (const int8_t*) v221;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v223 = __riscv_vle8_v_i8m1(v222, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v224 = __riscv_vwcvt_x_x_v_i16m2(v223, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v225 = __riscv_vneg_v_i16m2(v224, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v226 = __riscv_vmerge_vvm_i16m2(v225, v224, v220, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v227 = v165;
    vint32m1_t v228 = __riscv_vmv_v_x_i32m1(v227, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v229 = __riscv_vwredsum_vs_i16m2_i32m1(v226, v228, v166);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v230 = __riscv_vmv_x_s_i32m1_i32(v229);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v165 = v230;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v231 = v165;
    float v232 = v13;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v232 + v164 * (float) v231;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_base_y
    size_t v233 = v9 * 4;
    size_t v234 = v233 + 3;
    size_t v235 = v234 * 34;
    const uint8_t* v236 = v4 + v235;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v237 = (float)*(const _Float16 *)(v236);
    // tcrv_emitc.local_variable=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int32_t v238;
    v238 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v239 = __riscv_vsetvl_e8m1(8);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v240 = v11 + 14;
    const uint8_t* v241 = (const uint8_t*) v240;
    const uint8_t v242 = v241[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v243 = __riscv_vmv_v_x_u8m1(v242, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v244 = __riscv_vand_vv_u8m1(v243, v8, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v245 = __riscv_vmsne_vx_u8m1_b8(v244, 0, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v246 = v236 + 2;
    const int8_t* v247 = (const int8_t*) v246;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v248 = __riscv_vle8_v_i8m1(v247, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v249 = __riscv_vwcvt_x_x_v_i16m2(v248, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v250 = __riscv_vneg_v_i16m2(v249, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v251 = __riscv_vmerge_vvm_i16m2(v250, v249, v245, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v252 = v238;
    vint32m1_t v253 = __riscv_vmv_v_x_i32m1(v252, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v254 = __riscv_vwredsum_vs_i16m2_i32m1(v251, v253, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v255 = __riscv_vmv_x_s_i32m1_i32(v254);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v238 = v255;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v256 = v11 + 15;
    const uint8_t* v257 = (const uint8_t*) v256;
    const uint8_t v258 = v257[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v259 = __riscv_vmv_v_x_u8m1(v258, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v260 = __riscv_vand_vv_u8m1(v259, v8, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v261 = __riscv_vmsne_vx_u8m1_b8(v260, 0, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v262 = v236 + 10;
    const int8_t* v263 = (const int8_t*) v262;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v264 = __riscv_vle8_v_i8m1(v263, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v265 = __riscv_vwcvt_x_x_v_i16m2(v264, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v266 = __riscv_vneg_v_i16m2(v265, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v267 = __riscv_vmerge_vvm_i16m2(v266, v265, v261, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v268 = v238;
    vint32m1_t v269 = __riscv_vmv_v_x_i32m1(v268, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v270 = __riscv_vwredsum_vs_i16m2_i32m1(v267, v269, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v271 = __riscv_vmv_x_s_i32m1_i32(v270);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v238 = v271;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v272 = v11 + 16;
    const uint8_t* v273 = (const uint8_t*) v272;
    const uint8_t v274 = v273[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v275 = __riscv_vmv_v_x_u8m1(v274, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v276 = __riscv_vand_vv_u8m1(v275, v8, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v277 = __riscv_vmsne_vx_u8m1_b8(v276, 0, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v278 = v236 + 18;
    const int8_t* v279 = (const int8_t*) v278;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v280 = __riscv_vle8_v_i8m1(v279, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v281 = __riscv_vwcvt_x_x_v_i16m2(v280, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v282 = __riscv_vneg_v_i16m2(v281, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v283 = __riscv_vmerge_vvm_i16m2(v282, v281, v277, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v284 = v238;
    vint32m1_t v285 = __riscv_vmv_v_x_i32m1(v284, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v286 = __riscv_vwredsum_vs_i16m2_i32m1(v283, v285, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v287 = __riscv_vmv_x_s_i32m1_i32(v286);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v238 = v287;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=bits_byte_read
    const uint8_t* v288 = v11 + 17;
    const uint8_t* v289 = (const uint8_t*) v288;
    const uint8_t v290 = v289[0];
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u8m1
    vuint8m1_t v291 = __riscv_vmv_v_x_u8m1(v290, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vv_u8m1
    vuint8m1_t v292 = __riscv_vand_vv_u8m1(v291, v8, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m1_b8
    vbool8_t v293 = __riscv_vmsne_vx_u8m1_b8(v292, 0, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=q8_group_addr
    const uint8_t* v294 = v236 + 26;
    const int8_t* v295 = (const int8_t*) v294;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v296 = __riscv_vle8_v_i8m1(v295, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwcvt_x_x_v_i16m2
    vint16m2_t v297 = __riscv_vwcvt_x_x_v_i16m2(v296, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vneg_v_i16m2
    vint16m2_t v298 = __riscv_vneg_v_i16m2(v297, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i16m2
    vint16m2_t v299 = __riscv_vmerge_vvm_i16m2(v298, v297, v293, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    int32_t v300 = v238;
    vint32m1_t v301 = __riscv_vmv_v_x_i32m1(v300, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v302 = __riscv_vwredsum_vs_i16m2_i32m1(v299, v301, v239);
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int32_t v303 = __riscv_vmv_x_s_i32m1_i32(v302);
    // tcrv_emitc.assign target=sumi_block source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v238 = v303;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate_sub
    int32_t v304 = v238;
    float v305 = v13;
    // tcrv_emitc.assign target=sumi source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v13 = v305 + v237 * (float) v304;
    // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fp32_accumulate
    float v306 = v13;
    float v307 = v6;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v307 + v12 * v306;
  }
  // tcrv_emitc.source_op=tcrv_rvv.q1_0_q8_0_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  float v308 = v6;
  v2[0] = v308;
  return;
}


