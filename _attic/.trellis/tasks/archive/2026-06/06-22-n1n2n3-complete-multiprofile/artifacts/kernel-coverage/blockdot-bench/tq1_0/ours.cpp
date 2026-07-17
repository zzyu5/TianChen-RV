#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v9;
  v9 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v6; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 54;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_base3_trit
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v15 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v16 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v17 = __riscv_vle8_v_u8m2(v16, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2
    vuint8m2_t v18 = __riscv_vmul_vx_u8m2(v17, 1, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4
    vuint16m4_t v19 = __riscv_vwmulu_vx_u16m4(v18, 3, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4
    vuint16m4_t v20 = __riscv_vsrl_vx_u16m4(v19, 8, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2
    vuint8m2_t v21 = __riscv_vncvt_x_x_w_u8m2(v20, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v22 = __riscv_vreinterpret_v_u8m2_i8m2(v21);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v23 = __riscv_vadd_vx_i8m2(v22, -1, v15);
    int8_t* v24 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v24, v23, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v25 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v26 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v27 = __riscv_vle8_v_u8m2(v26, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2
    vuint8m2_t v28 = __riscv_vmul_vx_u8m2(v27, 3, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4
    vuint16m4_t v29 = __riscv_vwmulu_vx_u16m4(v28, 3, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4
    vuint16m4_t v30 = __riscv_vsrl_vx_u16m4(v29, 8, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2
    vuint8m2_t v31 = __riscv_vncvt_x_x_w_u8m2(v30, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v32 = __riscv_vreinterpret_v_u8m2_i8m2(v31);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v33 = __riscv_vadd_vx_i8m2(v32, -1, v25);
    int8_t* v34 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v34, v33, v25);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v35 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v36 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v37 = __riscv_vle8_v_u8m2(v36, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2
    vuint8m2_t v38 = __riscv_vmul_vx_u8m2(v37, 9, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4
    vuint16m4_t v39 = __riscv_vwmulu_vx_u16m4(v38, 3, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4
    vuint16m4_t v40 = __riscv_vsrl_vx_u16m4(v39, 8, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2
    vuint8m2_t v41 = __riscv_vncvt_x_x_w_u8m2(v40, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v42 = __riscv_vreinterpret_v_u8m2_i8m2(v41);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v43 = __riscv_vadd_vx_i8m2(v42, -1, v35);
    int8_t* v44 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v44, v43, v35);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v45 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v46 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v47 = __riscv_vle8_v_u8m2(v46, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2
    vuint8m2_t v48 = __riscv_vmul_vx_u8m2(v47, 27, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4
    vuint16m4_t v49 = __riscv_vwmulu_vx_u16m4(v48, 3, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4
    vuint16m4_t v50 = __riscv_vsrl_vx_u16m4(v49, 8, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2
    vuint8m2_t v51 = __riscv_vncvt_x_x_w_u8m2(v50, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v52 = __riscv_vreinterpret_v_u8m2_i8m2(v51);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v53 = __riscv_vadd_vx_i8m2(v52, -1, v45);
    int8_t* v54 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v54, v53, v45);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v55 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v56 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v57 = __riscv_vle8_v_u8m2(v56, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m2
    vuint8m2_t v58 = __riscv_vmul_vx_u8m2(v57, 81, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m4
    vuint16m4_t v59 = __riscv_vwmulu_vx_u16m4(v58, 3, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m4
    vuint16m4_t v60 = __riscv_vsrl_vx_u16m4(v59, 8, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m2
    vuint8m2_t v61 = __riscv_vncvt_x_x_w_u8m2(v60, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v62 = __riscv_vreinterpret_v_u8m2_i8m2(v61);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v63 = __riscv_vadd_vx_i8m2(v62, -1, v55);
    int8_t* v64 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v64, v63, v55);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v65 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v66 = v12 + 32;
    const uint8_t* v67 = (const uint8_t*) v66;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v68 = __riscv_vle8_v_u8m1(v67, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v69 = __riscv_vmul_vx_u8m1(v68, 1, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v70 = __riscv_vwmulu_vx_u16m2(v69, 3, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v71 = __riscv_vsrl_vx_u16m2(v70, 8, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v72 = __riscv_vncvt_x_x_w_u8m1(v71, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v73 = __riscv_vreinterpret_v_u8m1_i8m1(v72);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v74 = __riscv_vadd_vx_i8m1(v73, -1, v65);
    int8_t* v75 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v75, v74, v65);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v76 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v77 = v12 + 32;
    const uint8_t* v78 = (const uint8_t*) v77;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v79 = __riscv_vle8_v_u8m1(v78, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v80 = __riscv_vmul_vx_u8m1(v79, 3, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v81 = __riscv_vwmulu_vx_u16m2(v80, 3, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v82 = __riscv_vsrl_vx_u16m2(v81, 8, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v83 = __riscv_vncvt_x_x_w_u8m1(v82, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v84 = __riscv_vreinterpret_v_u8m1_i8m1(v83);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v85 = __riscv_vadd_vx_i8m1(v84, -1, v76);
    int8_t* v86 = &v7[176];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v86, v85, v76);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v87 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v88 = v12 + 32;
    const uint8_t* v89 = (const uint8_t*) v88;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v90 = __riscv_vle8_v_u8m1(v89, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v91 = __riscv_vmul_vx_u8m1(v90, 9, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v92 = __riscv_vwmulu_vx_u16m2(v91, 3, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v93 = __riscv_vsrl_vx_u16m2(v92, 8, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v94 = __riscv_vncvt_x_x_w_u8m1(v93, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v95 = __riscv_vreinterpret_v_u8m1_i8m1(v94);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v96 = __riscv_vadd_vx_i8m1(v95, -1, v87);
    int8_t* v97 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v97, v96, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v98 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v99 = v12 + 32;
    const uint8_t* v100 = (const uint8_t*) v99;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v101 = __riscv_vle8_v_u8m1(v100, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v102 = __riscv_vmul_vx_u8m1(v101, 27, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v103 = __riscv_vwmulu_vx_u16m2(v102, 3, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v104 = __riscv_vsrl_vx_u16m2(v103, 8, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v105 = __riscv_vncvt_x_x_w_u8m1(v104, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v106 = __riscv_vreinterpret_v_u8m1_i8m1(v105);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v107 = __riscv_vadd_vx_i8m1(v106, -1, v98);
    int8_t* v108 = &v7[208];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v108, v107, v98);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v109 = __riscv_vsetvl_e8m1(16);
    const uint8_t* v110 = v12 + 32;
    const uint8_t* v111 = (const uint8_t*) v110;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v112 = __riscv_vle8_v_u8m1(v111, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v113 = __riscv_vmul_vx_u8m1(v112, 81, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v114 = __riscv_vwmulu_vx_u16m2(v113, 3, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v115 = __riscv_vsrl_vx_u16m2(v114, 8, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v116 = __riscv_vncvt_x_x_w_u8m1(v115, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v117 = __riscv_vreinterpret_v_u8m1_i8m1(v116);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v118 = __riscv_vadd_vx_i8m1(v117, -1, v109);
    int8_t* v119 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v119, v118, v109);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v120 = __riscv_vsetvl_e8m1(4);
    const uint8_t* v121 = v12 + 48;
    const uint8_t* v122 = (const uint8_t*) v121;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v123 = __riscv_vle8_v_u8m1(v122, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v124 = __riscv_vmul_vx_u8m1(v123, 1, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v125 = __riscv_vwmulu_vx_u16m2(v124, 3, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v126 = __riscv_vsrl_vx_u16m2(v125, 8, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v127 = __riscv_vncvt_x_x_w_u8m1(v126, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v128 = __riscv_vreinterpret_v_u8m1_i8m1(v127);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v129 = __riscv_vadd_vx_i8m1(v128, -1, v120);
    int8_t* v130 = &v7[240];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v130, v129, v120);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v131 = __riscv_vsetvl_e8m1(4);
    const uint8_t* v132 = v12 + 48;
    const uint8_t* v133 = (const uint8_t*) v132;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v134 = __riscv_vle8_v_u8m1(v133, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v135 = __riscv_vmul_vx_u8m1(v134, 3, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v136 = __riscv_vwmulu_vx_u16m2(v135, 3, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v137 = __riscv_vsrl_vx_u16m2(v136, 8, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v138 = __riscv_vncvt_x_x_w_u8m1(v137, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v139 = __riscv_vreinterpret_v_u8m1_i8m1(v138);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v140 = __riscv_vadd_vx_i8m1(v139, -1, v131);
    int8_t* v141 = &v7[244];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v141, v140, v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v142 = __riscv_vsetvl_e8m1(4);
    const uint8_t* v143 = v12 + 48;
    const uint8_t* v144 = (const uint8_t*) v143;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v145 = __riscv_vle8_v_u8m1(v144, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v146 = __riscv_vmul_vx_u8m1(v145, 9, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v147 = __riscv_vwmulu_vx_u16m2(v146, 3, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v148 = __riscv_vsrl_vx_u16m2(v147, 8, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v149 = __riscv_vncvt_x_x_w_u8m1(v148, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v150 = __riscv_vreinterpret_v_u8m1_i8m1(v149);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v151 = __riscv_vadd_vx_i8m1(v150, -1, v142);
    int8_t* v152 = &v7[248];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v152, v151, v142);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v153 = __riscv_vsetvl_e8m1(4);
    const uint8_t* v154 = v12 + 48;
    const uint8_t* v155 = (const uint8_t*) v154;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v156 = __riscv_vle8_v_u8m1(v155, v153);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmul_vx_u8m1
    vuint8m1_t v157 = __riscv_vmul_vx_u8m1(v156, 27, v153);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmulu_vx_u16m2
    vuint16m2_t v158 = __riscv_vwmulu_vx_u16m2(v157, 3, v153);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u16m2
    vuint16m2_t v159 = __riscv_vsrl_vx_u16m2(v158, 8, v153);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
    vuint8m1_t v160 = __riscv_vncvt_x_x_w_u8m1(v159, v153);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
    vint8m1_t v161 = __riscv_vreinterpret_v_u8m1_i8m1(v160);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m1
    vint8m1_t v162 = __riscv_vadd_vx_i8m1(v161, -1, v153);
    int8_t* v163 = &v7[252];
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m1
    __riscv_vse8_v_i8m1(v163, v162, v153);
    const uint8_t* v164 = v14 + 4;
    const int8_t* v165 = (const int8_t*) v164;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v166;
    v166 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v167 = __riscv_vsetvl_e8m1(16);
    size_t v168 = 0 * 16;
    const int8_t* v169 = v165 + v168;
    const int8_t* v170 = v8 + v168;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v171 = __riscv_vle8_v_i8m1(v169, v167);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v172 = __riscv_vle8_v_i8m1(v170, v167);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v173 = __riscv_vwmul_vv_i16m2(v171, v172, v167);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v174 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v175 = __riscv_vwredsum_vs_i16m2_i32m1(v173, v174, v167);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v176 = __riscv_vmv_x_s_i32m1_i32(v175);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v177 = v166;
    int v178 = v177 + v176;
    v166 = v178;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v179 = __riscv_vsetvl_e8m1(16);
    size_t v180 = 1 * 16;
    const int8_t* v181 = v165 + v180;
    const int8_t* v182 = v8 + v180;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v183 = __riscv_vle8_v_i8m1(v181, v179);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v184 = __riscv_vle8_v_i8m1(v182, v179);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v185 = __riscv_vwmul_vv_i16m2(v183, v184, v179);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v186 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v187 = __riscv_vwredsum_vs_i16m2_i32m1(v185, v186, v179);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v188 = __riscv_vmv_x_s_i32m1_i32(v187);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v189 = v166;
    int v190 = v189 + v188;
    v166 = v190;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v191 = __riscv_vsetvl_e8m1(16);
    size_t v192 = 2 * 16;
    const int8_t* v193 = v165 + v192;
    const int8_t* v194 = v8 + v192;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v195 = __riscv_vle8_v_i8m1(v193, v191);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v196 = __riscv_vle8_v_i8m1(v194, v191);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v197 = __riscv_vwmul_vv_i16m2(v195, v196, v191);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v198 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v199 = __riscv_vwredsum_vs_i16m2_i32m1(v197, v198, v191);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v200 = __riscv_vmv_x_s_i32m1_i32(v199);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v201 = v166;
    int v202 = v201 + v200;
    v166 = v202;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v203 = __riscv_vsetvl_e8m1(16);
    size_t v204 = 3 * 16;
    const int8_t* v205 = v165 + v204;
    const int8_t* v206 = v8 + v204;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v207 = __riscv_vle8_v_i8m1(v205, v203);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v208 = __riscv_vle8_v_i8m1(v206, v203);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v209 = __riscv_vwmul_vv_i16m2(v207, v208, v203);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v210 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v211 = __riscv_vwredsum_vs_i16m2_i32m1(v209, v210, v203);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v212 = __riscv_vmv_x_s_i32m1_i32(v211);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v213 = v166;
    int v214 = v213 + v212;
    v166 = v214;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v215 = __riscv_vsetvl_e8m1(16);
    size_t v216 = 4 * 16;
    const int8_t* v217 = v165 + v216;
    const int8_t* v218 = v8 + v216;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v219 = __riscv_vle8_v_i8m1(v217, v215);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v220 = __riscv_vle8_v_i8m1(v218, v215);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v221 = __riscv_vwmul_vv_i16m2(v219, v220, v215);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v222 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v223 = __riscv_vwredsum_vs_i16m2_i32m1(v221, v222, v215);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v224 = __riscv_vmv_x_s_i32m1_i32(v223);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v225 = v166;
    int v226 = v225 + v224;
    v166 = v226;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v227 = __riscv_vsetvl_e8m1(16);
    size_t v228 = 5 * 16;
    const int8_t* v229 = v165 + v228;
    const int8_t* v230 = v8 + v228;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v231 = __riscv_vle8_v_i8m1(v229, v227);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v232 = __riscv_vle8_v_i8m1(v230, v227);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v233 = __riscv_vwmul_vv_i16m2(v231, v232, v227);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v234 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v235 = __riscv_vwredsum_vs_i16m2_i32m1(v233, v234, v227);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v236 = __riscv_vmv_x_s_i32m1_i32(v235);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v237 = v166;
    int v238 = v237 + v236;
    v166 = v238;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v239 = __riscv_vsetvl_e8m1(16);
    size_t v240 = 6 * 16;
    const int8_t* v241 = v165 + v240;
    const int8_t* v242 = v8 + v240;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v243 = __riscv_vle8_v_i8m1(v241, v239);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v244 = __riscv_vle8_v_i8m1(v242, v239);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v245 = __riscv_vwmul_vv_i16m2(v243, v244, v239);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v246 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v247 = __riscv_vwredsum_vs_i16m2_i32m1(v245, v246, v239);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v248 = __riscv_vmv_x_s_i32m1_i32(v247);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v249 = v166;
    int v250 = v249 + v248;
    v166 = v250;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v251 = __riscv_vsetvl_e8m1(16);
    size_t v252 = 7 * 16;
    const int8_t* v253 = v165 + v252;
    const int8_t* v254 = v8 + v252;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v255 = __riscv_vle8_v_i8m1(v253, v251);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v256 = __riscv_vle8_v_i8m1(v254, v251);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v257 = __riscv_vwmul_vv_i16m2(v255, v256, v251);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v258 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v259 = __riscv_vwredsum_vs_i16m2_i32m1(v257, v258, v251);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v260 = __riscv_vmv_x_s_i32m1_i32(v259);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v261 = v166;
    int v262 = v261 + v260;
    v166 = v262;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v263 = __riscv_vsetvl_e8m1(16);
    size_t v264 = 8 * 16;
    const int8_t* v265 = v165 + v264;
    const int8_t* v266 = v8 + v264;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v267 = __riscv_vle8_v_i8m1(v265, v263);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v268 = __riscv_vle8_v_i8m1(v266, v263);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v269 = __riscv_vwmul_vv_i16m2(v267, v268, v263);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v270 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v271 = __riscv_vwredsum_vs_i16m2_i32m1(v269, v270, v263);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v272 = __riscv_vmv_x_s_i32m1_i32(v271);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v273 = v166;
    int v274 = v273 + v272;
    v166 = v274;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v275 = __riscv_vsetvl_e8m1(16);
    size_t v276 = 9 * 16;
    const int8_t* v277 = v165 + v276;
    const int8_t* v278 = v8 + v276;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v279 = __riscv_vle8_v_i8m1(v277, v275);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v280 = __riscv_vle8_v_i8m1(v278, v275);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v281 = __riscv_vwmul_vv_i16m2(v279, v280, v275);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v282 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v283 = __riscv_vwredsum_vs_i16m2_i32m1(v281, v282, v275);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v284 = __riscv_vmv_x_s_i32m1_i32(v283);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v285 = v166;
    int v286 = v285 + v284;
    v166 = v286;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v287 = __riscv_vsetvl_e8m1(16);
    size_t v288 = 10 * 16;
    const int8_t* v289 = v165 + v288;
    const int8_t* v290 = v8 + v288;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v291 = __riscv_vle8_v_i8m1(v289, v287);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v292 = __riscv_vle8_v_i8m1(v290, v287);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v293 = __riscv_vwmul_vv_i16m2(v291, v292, v287);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v294 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v295 = __riscv_vwredsum_vs_i16m2_i32m1(v293, v294, v287);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v296 = __riscv_vmv_x_s_i32m1_i32(v295);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v297 = v166;
    int v298 = v297 + v296;
    v166 = v298;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v299 = __riscv_vsetvl_e8m1(16);
    size_t v300 = 11 * 16;
    const int8_t* v301 = v165 + v300;
    const int8_t* v302 = v8 + v300;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v303 = __riscv_vle8_v_i8m1(v301, v299);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v304 = __riscv_vle8_v_i8m1(v302, v299);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v305 = __riscv_vwmul_vv_i16m2(v303, v304, v299);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v306 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v307 = __riscv_vwredsum_vs_i16m2_i32m1(v305, v306, v299);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v308 = __riscv_vmv_x_s_i32m1_i32(v307);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v309 = v166;
    int v310 = v309 + v308;
    v166 = v310;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v311 = __riscv_vsetvl_e8m1(16);
    size_t v312 = 12 * 16;
    const int8_t* v313 = v165 + v312;
    const int8_t* v314 = v8 + v312;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v315 = __riscv_vle8_v_i8m1(v313, v311);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v316 = __riscv_vle8_v_i8m1(v314, v311);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v317 = __riscv_vwmul_vv_i16m2(v315, v316, v311);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v318 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v319 = __riscv_vwredsum_vs_i16m2_i32m1(v317, v318, v311);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v320 = __riscv_vmv_x_s_i32m1_i32(v319);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v321 = v166;
    int v322 = v321 + v320;
    v166 = v322;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v323 = __riscv_vsetvl_e8m1(16);
    size_t v324 = 13 * 16;
    const int8_t* v325 = v165 + v324;
    const int8_t* v326 = v8 + v324;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v327 = __riscv_vle8_v_i8m1(v325, v323);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v328 = __riscv_vle8_v_i8m1(v326, v323);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v329 = __riscv_vwmul_vv_i16m2(v327, v328, v323);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v330 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v331 = __riscv_vwredsum_vs_i16m2_i32m1(v329, v330, v323);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v332 = __riscv_vmv_x_s_i32m1_i32(v331);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v333 = v166;
    int v334 = v333 + v332;
    v166 = v334;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v335 = __riscv_vsetvl_e8m1(16);
    size_t v336 = 14 * 16;
    const int8_t* v337 = v165 + v336;
    const int8_t* v338 = v8 + v336;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v339 = __riscv_vle8_v_i8m1(v337, v335);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v340 = __riscv_vle8_v_i8m1(v338, v335);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v341 = __riscv_vwmul_vv_i16m2(v339, v340, v335);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v342 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v343 = __riscv_vwredsum_vs_i16m2_i32m1(v341, v342, v335);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v344 = __riscv_vmv_x_s_i32m1_i32(v343);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v345 = v166;
    int v346 = v345 + v344;
    v166 = v346;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v347 = __riscv_vsetvl_e8m1(16);
    size_t v348 = 15 * 16;
    const int8_t* v349 = v165 + v348;
    const int8_t* v350 = v8 + v348;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v351 = __riscv_vle8_v_i8m1(v349, v347);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v352 = __riscv_vle8_v_i8m1(v350, v347);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v353 = __riscv_vwmul_vv_i16m2(v351, v352, v347);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v354 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v355 = __riscv_vwredsum_vs_i16m2_i32m1(v353, v354, v347);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v356 = __riscv_vmv_x_s_i32m1_i32(v355);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v357 = v166;
    int v358 = v357 + v356;
    v166 = v358;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    const uint8_t* v359 = v12 + 52;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v360 = (float)*(const _Float16 *)(v359);
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v361 = (const float*) v14;
    const float v362 = v361[0];
    float v363 = v360 * v362;
    // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold
    int v364 = v166;
    float v365 = v9;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v365 + (float) v364 * v363;
  }
  float v366 = v9;
  // tcrv_emitc.source_op=tcrv_rvv.tq1_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v366;
  return;
}


