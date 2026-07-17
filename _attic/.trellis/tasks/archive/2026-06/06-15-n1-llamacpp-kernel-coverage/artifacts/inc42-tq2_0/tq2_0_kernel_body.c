#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v9;
  v9 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v6; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 66;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_2bit_ternary
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v15 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v16 = (const uint8_t*) v12;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v17 = __riscv_vle8_v_u8m2(v16, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v18 = __riscv_vand_vx_u8m2(v17, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v19 = __riscv_vreinterpret_v_u8m2_i8m2(v18);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v20 = __riscv_vadd_vx_i8m2(v19, -1, v15);
    int8_t* v21 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v21, v20, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v22 = __riscv_vsrl_vx_u8m2(v17, 2, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v23 = __riscv_vand_vx_u8m2(v22, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v24 = __riscv_vreinterpret_v_u8m2_i8m2(v23);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v25 = __riscv_vadd_vx_i8m2(v24, -1, v15);
    int8_t* v26 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v26, v25, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v27 = __riscv_vsrl_vx_u8m2(v17, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v28 = __riscv_vand_vx_u8m2(v27, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v29 = __riscv_vreinterpret_v_u8m2_i8m2(v28);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v30 = __riscv_vadd_vx_i8m2(v29, -1, v15);
    int8_t* v31 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v31, v30, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v32 = __riscv_vsrl_vx_u8m2(v17, 6, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v33 = __riscv_vand_vx_u8m2(v32, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v34 = __riscv_vreinterpret_v_u8m2_i8m2(v33);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v35 = __riscv_vadd_vx_i8m2(v34, -1, v15);
    int8_t* v36 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v36, v35, v15);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v37 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v38 = v12 + 32;
    const uint8_t* v39 = (const uint8_t*) v38;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v40 = __riscv_vle8_v_u8m2(v39, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v41 = __riscv_vand_vx_u8m2(v40, 0x03, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v42 = __riscv_vreinterpret_v_u8m2_i8m2(v41);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v43 = __riscv_vadd_vx_i8m2(v42, -1, v37);
    int8_t* v44 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v44, v43, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v45 = __riscv_vsrl_vx_u8m2(v40, 2, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v46 = __riscv_vand_vx_u8m2(v45, 0x03, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v47 = __riscv_vreinterpret_v_u8m2_i8m2(v46);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v48 = __riscv_vadd_vx_i8m2(v47, -1, v37);
    int8_t* v49 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v49, v48, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v50 = __riscv_vsrl_vx_u8m2(v40, 4, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v51 = __riscv_vand_vx_u8m2(v50, 0x03, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v52 = __riscv_vreinterpret_v_u8m2_i8m2(v51);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v53 = __riscv_vadd_vx_i8m2(v52, -1, v37);
    int8_t* v54 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v54, v53, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v55 = __riscv_vsrl_vx_u8m2(v40, 6, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v56 = __riscv_vand_vx_u8m2(v55, 0x03, v37);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v57 = __riscv_vreinterpret_v_u8m2_i8m2(v56);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_i8m2
    vint8m2_t v58 = __riscv_vadd_vx_i8m2(v57, -1, v37);
    int8_t* v59 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v59, v58, v37);
    const uint8_t* v60 = v14 + 4;
    const int8_t* v61 = (const int8_t*) v60;
    // tcrv_emitc.local_variable=sumi source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v62;
    v62 = 0;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v63 = __riscv_vsetvl_e8m1(16);
    size_t v64 = 0 * 16;
    const int8_t* v65 = v61 + v64;
    const int8_t* v66 = v8 + v64;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v67 = __riscv_vle8_v_i8m1(v65, v63);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v68 = __riscv_vle8_v_i8m1(v66, v63);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v69 = __riscv_vwmul_vv_i16m2(v67, v68, v63);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v70 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v71 = __riscv_vwredsum_vs_i16m2_i32m1(v69, v70, v63);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v72 = __riscv_vmv_x_s_i32m1_i32(v71);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v73 = v62;
    int v74 = v73 + v72;
    v62 = v74;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v75 = __riscv_vsetvl_e8m1(16);
    size_t v76 = 1 * 16;
    const int8_t* v77 = v61 + v76;
    const int8_t* v78 = v8 + v76;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v79 = __riscv_vle8_v_i8m1(v77, v75);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v80 = __riscv_vle8_v_i8m1(v78, v75);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v81 = __riscv_vwmul_vv_i16m2(v79, v80, v75);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v82 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v83 = __riscv_vwredsum_vs_i16m2_i32m1(v81, v82, v75);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v84 = __riscv_vmv_x_s_i32m1_i32(v83);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v85 = v62;
    int v86 = v85 + v84;
    v62 = v86;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v87 = __riscv_vsetvl_e8m1(16);
    size_t v88 = 2 * 16;
    const int8_t* v89 = v61 + v88;
    const int8_t* v90 = v8 + v88;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v91 = __riscv_vle8_v_i8m1(v89, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v92 = __riscv_vle8_v_i8m1(v90, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v93 = __riscv_vwmul_vv_i16m2(v91, v92, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v94 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v95 = __riscv_vwredsum_vs_i16m2_i32m1(v93, v94, v87);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v96 = __riscv_vmv_x_s_i32m1_i32(v95);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v97 = v62;
    int v98 = v97 + v96;
    v62 = v98;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v99 = __riscv_vsetvl_e8m1(16);
    size_t v100 = 3 * 16;
    const int8_t* v101 = v61 + v100;
    const int8_t* v102 = v8 + v100;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v103 = __riscv_vle8_v_i8m1(v101, v99);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v104 = __riscv_vle8_v_i8m1(v102, v99);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v105 = __riscv_vwmul_vv_i16m2(v103, v104, v99);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v106 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v107 = __riscv_vwredsum_vs_i16m2_i32m1(v105, v106, v99);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v108 = __riscv_vmv_x_s_i32m1_i32(v107);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v109 = v62;
    int v110 = v109 + v108;
    v62 = v110;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v111 = __riscv_vsetvl_e8m1(16);
    size_t v112 = 4 * 16;
    const int8_t* v113 = v61 + v112;
    const int8_t* v114 = v8 + v112;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v115 = __riscv_vle8_v_i8m1(v113, v111);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v116 = __riscv_vle8_v_i8m1(v114, v111);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v117 = __riscv_vwmul_vv_i16m2(v115, v116, v111);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v118 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v119 = __riscv_vwredsum_vs_i16m2_i32m1(v117, v118, v111);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v120 = __riscv_vmv_x_s_i32m1_i32(v119);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v121 = v62;
    int v122 = v121 + v120;
    v62 = v122;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v123 = __riscv_vsetvl_e8m1(16);
    size_t v124 = 5 * 16;
    const int8_t* v125 = v61 + v124;
    const int8_t* v126 = v8 + v124;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v127 = __riscv_vle8_v_i8m1(v125, v123);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v128 = __riscv_vle8_v_i8m1(v126, v123);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v129 = __riscv_vwmul_vv_i16m2(v127, v128, v123);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v130 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v131 = __riscv_vwredsum_vs_i16m2_i32m1(v129, v130, v123);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v132 = __riscv_vmv_x_s_i32m1_i32(v131);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v133 = v62;
    int v134 = v133 + v132;
    v62 = v134;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v135 = __riscv_vsetvl_e8m1(16);
    size_t v136 = 6 * 16;
    const int8_t* v137 = v61 + v136;
    const int8_t* v138 = v8 + v136;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v139 = __riscv_vle8_v_i8m1(v137, v135);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v140 = __riscv_vle8_v_i8m1(v138, v135);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v141 = __riscv_vwmul_vv_i16m2(v139, v140, v135);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v142 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v143 = __riscv_vwredsum_vs_i16m2_i32m1(v141, v142, v135);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v144 = __riscv_vmv_x_s_i32m1_i32(v143);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v145 = v62;
    int v146 = v145 + v144;
    v62 = v146;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v147 = __riscv_vsetvl_e8m1(16);
    size_t v148 = 7 * 16;
    const int8_t* v149 = v61 + v148;
    const int8_t* v150 = v8 + v148;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v151 = __riscv_vle8_v_i8m1(v149, v147);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v152 = __riscv_vle8_v_i8m1(v150, v147);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v153 = __riscv_vwmul_vv_i16m2(v151, v152, v147);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v154 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v155 = __riscv_vwredsum_vs_i16m2_i32m1(v153, v154, v147);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v156 = __riscv_vmv_x_s_i32m1_i32(v155);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v157 = v62;
    int v158 = v157 + v156;
    v62 = v158;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v159 = __riscv_vsetvl_e8m1(16);
    size_t v160 = 8 * 16;
    const int8_t* v161 = v61 + v160;
    const int8_t* v162 = v8 + v160;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v163 = __riscv_vle8_v_i8m1(v161, v159);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v164 = __riscv_vle8_v_i8m1(v162, v159);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v165 = __riscv_vwmul_vv_i16m2(v163, v164, v159);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v166 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v167 = __riscv_vwredsum_vs_i16m2_i32m1(v165, v166, v159);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v168 = __riscv_vmv_x_s_i32m1_i32(v167);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v169 = v62;
    int v170 = v169 + v168;
    v62 = v170;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v171 = __riscv_vsetvl_e8m1(16);
    size_t v172 = 9 * 16;
    const int8_t* v173 = v61 + v172;
    const int8_t* v174 = v8 + v172;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v175 = __riscv_vle8_v_i8m1(v173, v171);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v176 = __riscv_vle8_v_i8m1(v174, v171);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v177 = __riscv_vwmul_vv_i16m2(v175, v176, v171);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v178 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v179 = __riscv_vwredsum_vs_i16m2_i32m1(v177, v178, v171);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v180 = __riscv_vmv_x_s_i32m1_i32(v179);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v181 = v62;
    int v182 = v181 + v180;
    v62 = v182;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v183 = __riscv_vsetvl_e8m1(16);
    size_t v184 = 10 * 16;
    const int8_t* v185 = v61 + v184;
    const int8_t* v186 = v8 + v184;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v187 = __riscv_vle8_v_i8m1(v185, v183);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v188 = __riscv_vle8_v_i8m1(v186, v183);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v189 = __riscv_vwmul_vv_i16m2(v187, v188, v183);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v190 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v191 = __riscv_vwredsum_vs_i16m2_i32m1(v189, v190, v183);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v192 = __riscv_vmv_x_s_i32m1_i32(v191);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v193 = v62;
    int v194 = v193 + v192;
    v62 = v194;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v195 = __riscv_vsetvl_e8m1(16);
    size_t v196 = 11 * 16;
    const int8_t* v197 = v61 + v196;
    const int8_t* v198 = v8 + v196;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v199 = __riscv_vle8_v_i8m1(v197, v195);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v200 = __riscv_vle8_v_i8m1(v198, v195);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v201 = __riscv_vwmul_vv_i16m2(v199, v200, v195);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v202 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v203 = __riscv_vwredsum_vs_i16m2_i32m1(v201, v202, v195);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v204 = __riscv_vmv_x_s_i32m1_i32(v203);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v205 = v62;
    int v206 = v205 + v204;
    v62 = v206;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v207 = __riscv_vsetvl_e8m1(16);
    size_t v208 = 12 * 16;
    const int8_t* v209 = v61 + v208;
    const int8_t* v210 = v8 + v208;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v211 = __riscv_vle8_v_i8m1(v209, v207);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v212 = __riscv_vle8_v_i8m1(v210, v207);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v213 = __riscv_vwmul_vv_i16m2(v211, v212, v207);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v214 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v215 = __riscv_vwredsum_vs_i16m2_i32m1(v213, v214, v207);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v216 = __riscv_vmv_x_s_i32m1_i32(v215);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v217 = v62;
    int v218 = v217 + v216;
    v62 = v218;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v219 = __riscv_vsetvl_e8m1(16);
    size_t v220 = 13 * 16;
    const int8_t* v221 = v61 + v220;
    const int8_t* v222 = v8 + v220;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v223 = __riscv_vle8_v_i8m1(v221, v219);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v224 = __riscv_vle8_v_i8m1(v222, v219);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v225 = __riscv_vwmul_vv_i16m2(v223, v224, v219);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v226 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v227 = __riscv_vwredsum_vs_i16m2_i32m1(v225, v226, v219);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v228 = __riscv_vmv_x_s_i32m1_i32(v227);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v229 = v62;
    int v230 = v229 + v228;
    v62 = v230;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v231 = __riscv_vsetvl_e8m1(16);
    size_t v232 = 14 * 16;
    const int8_t* v233 = v61 + v232;
    const int8_t* v234 = v8 + v232;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v235 = __riscv_vle8_v_i8m1(v233, v231);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v236 = __riscv_vle8_v_i8m1(v234, v231);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v237 = __riscv_vwmul_vv_i16m2(v235, v236, v231);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v238 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v239 = __riscv_vwredsum_vs_i16m2_i32m1(v237, v238, v231);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v240 = __riscv_vmv_x_s_i32m1_i32(v239);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v241 = v62;
    int v242 = v241 + v240;
    v62 = v242;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v243 = __riscv_vsetvl_e8m1(16);
    size_t v244 = 15 * 16;
    const int8_t* v245 = v61 + v244;
    const int8_t* v246 = v8 + v244;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v247 = __riscv_vle8_v_i8m1(v245, v243);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v248 = __riscv_vle8_v_i8m1(v246, v243);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v249 = __riscv_vwmul_vv_i16m2(v247, v248, v243);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v250 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v251 = __riscv_vwredsum_vs_i16m2_i32m1(v249, v250, v243);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v252 = __riscv_vmv_x_s_i32m1_i32(v251);
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sumi_accumulate
    int v253 = v62;
    int v254 = v253 + v252;
    v62 = v254;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v255 = (const float*) v14;
    const float v256 = v255[0];
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_d
    const uint8_t* v257 = v12 + 64;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v258 = (float)*(const _Float16 *)(v257);
    float v259 = v256 * v258;
    // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold
    int v260 = v62;
    float v261 = v9;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v261 + (float) v260 * v259;
  }
  float v262 = v9;
  // tcrv_emitc.source_op=tcrv_rvv.tq2_0_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v262;
  return;
}


