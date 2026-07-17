#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q2_K_q8_K_kernel_ggml_vec_dot_q2_K_q8_K(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // tcrv_emitc.local_variable=sumf source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  float v9;
  v9 = 0.0f;
  // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v10 = 0; v10 < v6; v10 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v11 = v10 * 84;
    const uint8_t* v12 = v3 + v11;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v13 = v10 * 292;
    const uint8_t* v14 = v4 + v13;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_2bit
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v15 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v16 = v12 + 16;
    const uint8_t* v17 = (const uint8_t*) v16;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v18 = __riscv_vle8_v_u8m2(v17, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v19 = __riscv_vand_vx_u8m2(v18, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v20 = __riscv_vreinterpret_v_u8m2_i8m2(v19);
    int8_t* v21 = &v7[0];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v21, v20, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v22 = __riscv_vsrl_vx_u8m2(v18, 2, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v23 = __riscv_vand_vx_u8m2(v22, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v24 = __riscv_vreinterpret_v_u8m2_i8m2(v23);
    int8_t* v25 = &v7[32];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v25, v24, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v26 = __riscv_vsrl_vx_u8m2(v18, 4, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v27 = __riscv_vand_vx_u8m2(v26, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v28 = __riscv_vreinterpret_v_u8m2_i8m2(v27);
    int8_t* v29 = &v7[64];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v29, v28, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v30 = __riscv_vsrl_vx_u8m2(v18, 6, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v31 = __riscv_vand_vx_u8m2(v30, 0x03, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v32 = __riscv_vreinterpret_v_u8m2_i8m2(v31);
    int8_t* v33 = &v7[96];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v33, v32, v15);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v34 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v35 = v12 + 48;
    const uint8_t* v36 = (const uint8_t*) v35;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v37 = __riscv_vle8_v_u8m2(v36, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v38 = __riscv_vand_vx_u8m2(v37, 0x03, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v39 = __riscv_vreinterpret_v_u8m2_i8m2(v38);
    int8_t* v40 = &v7[128];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v40, v39, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v41 = __riscv_vsrl_vx_u8m2(v37, 2, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v42 = __riscv_vand_vx_u8m2(v41, 0x03, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v43 = __riscv_vreinterpret_v_u8m2_i8m2(v42);
    int8_t* v44 = &v7[160];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v44, v43, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v45 = __riscv_vsrl_vx_u8m2(v37, 4, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v46 = __riscv_vand_vx_u8m2(v45, 0x03, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v47 = __riscv_vreinterpret_v_u8m2_i8m2(v46);
    int8_t* v48 = &v7[192];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v48, v47, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v49 = __riscv_vsrl_vx_u8m2(v37, 6, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v50 = __riscv_vand_vx_u8m2(v49, 0x03, v34);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v51 = __riscv_vreinterpret_v_u8m2_i8m2(v50);
    int8_t* v52 = &v7[224];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v52, v51, v34);
    const uint8_t* v53 = (const uint8_t*) v12;
    const uint8_t* v54 = v14 + 4;
    const int8_t* v55 = (const int8_t*) v54;
    const uint8_t* v56 = v14 + 260;
    const int16_t* v57 = (const int16_t*) v56;
    // tcrv_emitc.local_variable=isum source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v58;
    v58 = 0;
    // tcrv_emitc.local_variable=summs source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    int v59;
    v59 = 0;
    const uint8_t v60 = v53[0];
    int v61 = (int) v60;
    int v62 = v61 & 0xF;
    int v63 = v61 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v64 = __riscv_vsetvl_e8m1(16);
    size_t v65 = 0 * 16;
    const int8_t* v66 = v55 + v65;
    const int8_t* v67 = v8 + v65;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v68 = __riscv_vle8_v_i8m1(v66, v64);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v69 = __riscv_vle8_v_i8m1(v67, v64);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v70 = __riscv_vwmul_vv_i16m2(v68, v69, v64);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v71 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v72 = __riscv_vwredsum_vs_i16m2_i32m1(v70, v71, v64);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v73 = __riscv_vmv_x_s_i32m1_i32(v72);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v74 = v62 * v73;
    int v75 = v58;
    int v76 = v75 + v74;
    v58 = v76;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v77 = v57[0];
    int v78 = (int) v77;
    int v79 = v78 * v63;
    int v80 = v59;
    int v81 = v80 + v79;
    v59 = v81;
    const uint8_t v82 = v53[1];
    int v83 = (int) v82;
    int v84 = v83 & 0xF;
    int v85 = v83 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v86 = __riscv_vsetvl_e8m1(16);
    size_t v87 = 1 * 16;
    const int8_t* v88 = v55 + v87;
    const int8_t* v89 = v8 + v87;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v90 = __riscv_vle8_v_i8m1(v88, v86);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v91 = __riscv_vle8_v_i8m1(v89, v86);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v92 = __riscv_vwmul_vv_i16m2(v90, v91, v86);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v93 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v94 = __riscv_vwredsum_vs_i16m2_i32m1(v92, v93, v86);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v95 = __riscv_vmv_x_s_i32m1_i32(v94);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v96 = v84 * v95;
    int v97 = v58;
    int v98 = v97 + v96;
    v58 = v98;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v99 = v57[1];
    int v100 = (int) v99;
    int v101 = v100 * v85;
    int v102 = v59;
    int v103 = v102 + v101;
    v59 = v103;
    const uint8_t v104 = v53[2];
    int v105 = (int) v104;
    int v106 = v105 & 0xF;
    int v107 = v105 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v108 = __riscv_vsetvl_e8m1(16);
    size_t v109 = 2 * 16;
    const int8_t* v110 = v55 + v109;
    const int8_t* v111 = v8 + v109;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v112 = __riscv_vle8_v_i8m1(v110, v108);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v113 = __riscv_vle8_v_i8m1(v111, v108);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v114 = __riscv_vwmul_vv_i16m2(v112, v113, v108);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v115 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v116 = __riscv_vwredsum_vs_i16m2_i32m1(v114, v115, v108);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v117 = __riscv_vmv_x_s_i32m1_i32(v116);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v118 = v106 * v117;
    int v119 = v58;
    int v120 = v119 + v118;
    v58 = v120;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v121 = v57[2];
    int v122 = (int) v121;
    int v123 = v122 * v107;
    int v124 = v59;
    int v125 = v124 + v123;
    v59 = v125;
    const uint8_t v126 = v53[3];
    int v127 = (int) v126;
    int v128 = v127 & 0xF;
    int v129 = v127 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v130 = __riscv_vsetvl_e8m1(16);
    size_t v131 = 3 * 16;
    const int8_t* v132 = v55 + v131;
    const int8_t* v133 = v8 + v131;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v134 = __riscv_vle8_v_i8m1(v132, v130);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v135 = __riscv_vle8_v_i8m1(v133, v130);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v136 = __riscv_vwmul_vv_i16m2(v134, v135, v130);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v137 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v138 = __riscv_vwredsum_vs_i16m2_i32m1(v136, v137, v130);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v139 = __riscv_vmv_x_s_i32m1_i32(v138);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v140 = v128 * v139;
    int v141 = v58;
    int v142 = v141 + v140;
    v58 = v142;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v143 = v57[3];
    int v144 = (int) v143;
    int v145 = v144 * v129;
    int v146 = v59;
    int v147 = v146 + v145;
    v59 = v147;
    const uint8_t v148 = v53[4];
    int v149 = (int) v148;
    int v150 = v149 & 0xF;
    int v151 = v149 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v152 = __riscv_vsetvl_e8m1(16);
    size_t v153 = 4 * 16;
    const int8_t* v154 = v55 + v153;
    const int8_t* v155 = v8 + v153;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v156 = __riscv_vle8_v_i8m1(v154, v152);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v157 = __riscv_vle8_v_i8m1(v155, v152);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v158 = __riscv_vwmul_vv_i16m2(v156, v157, v152);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v159 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v160 = __riscv_vwredsum_vs_i16m2_i32m1(v158, v159, v152);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v161 = __riscv_vmv_x_s_i32m1_i32(v160);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v162 = v150 * v161;
    int v163 = v58;
    int v164 = v163 + v162;
    v58 = v164;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v165 = v57[4];
    int v166 = (int) v165;
    int v167 = v166 * v151;
    int v168 = v59;
    int v169 = v168 + v167;
    v59 = v169;
    const uint8_t v170 = v53[5];
    int v171 = (int) v170;
    int v172 = v171 & 0xF;
    int v173 = v171 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v174 = __riscv_vsetvl_e8m1(16);
    size_t v175 = 5 * 16;
    const int8_t* v176 = v55 + v175;
    const int8_t* v177 = v8 + v175;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v178 = __riscv_vle8_v_i8m1(v176, v174);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v179 = __riscv_vle8_v_i8m1(v177, v174);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v180 = __riscv_vwmul_vv_i16m2(v178, v179, v174);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v181 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v182 = __riscv_vwredsum_vs_i16m2_i32m1(v180, v181, v174);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v183 = __riscv_vmv_x_s_i32m1_i32(v182);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v184 = v172 * v183;
    int v185 = v58;
    int v186 = v185 + v184;
    v58 = v186;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v187 = v57[5];
    int v188 = (int) v187;
    int v189 = v188 * v173;
    int v190 = v59;
    int v191 = v190 + v189;
    v59 = v191;
    const uint8_t v192 = v53[6];
    int v193 = (int) v192;
    int v194 = v193 & 0xF;
    int v195 = v193 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v196 = __riscv_vsetvl_e8m1(16);
    size_t v197 = 6 * 16;
    const int8_t* v198 = v55 + v197;
    const int8_t* v199 = v8 + v197;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v200 = __riscv_vle8_v_i8m1(v198, v196);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v201 = __riscv_vle8_v_i8m1(v199, v196);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v202 = __riscv_vwmul_vv_i16m2(v200, v201, v196);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v203 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v204 = __riscv_vwredsum_vs_i16m2_i32m1(v202, v203, v196);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v205 = __riscv_vmv_x_s_i32m1_i32(v204);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v206 = v194 * v205;
    int v207 = v58;
    int v208 = v207 + v206;
    v58 = v208;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v209 = v57[6];
    int v210 = (int) v209;
    int v211 = v210 * v195;
    int v212 = v59;
    int v213 = v212 + v211;
    v59 = v213;
    const uint8_t v214 = v53[7];
    int v215 = (int) v214;
    int v216 = v215 & 0xF;
    int v217 = v215 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v218 = __riscv_vsetvl_e8m1(16);
    size_t v219 = 7 * 16;
    const int8_t* v220 = v55 + v219;
    const int8_t* v221 = v8 + v219;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v222 = __riscv_vle8_v_i8m1(v220, v218);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v223 = __riscv_vle8_v_i8m1(v221, v218);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v224 = __riscv_vwmul_vv_i16m2(v222, v223, v218);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v225 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v226 = __riscv_vwredsum_vs_i16m2_i32m1(v224, v225, v218);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v227 = __riscv_vmv_x_s_i32m1_i32(v226);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v228 = v216 * v227;
    int v229 = v58;
    int v230 = v229 + v228;
    v58 = v230;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v231 = v57[7];
    int v232 = (int) v231;
    int v233 = v232 * v217;
    int v234 = v59;
    int v235 = v234 + v233;
    v59 = v235;
    const uint8_t v236 = v53[8];
    int v237 = (int) v236;
    int v238 = v237 & 0xF;
    int v239 = v237 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v240 = __riscv_vsetvl_e8m1(16);
    size_t v241 = 8 * 16;
    const int8_t* v242 = v55 + v241;
    const int8_t* v243 = v8 + v241;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v244 = __riscv_vle8_v_i8m1(v242, v240);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v245 = __riscv_vle8_v_i8m1(v243, v240);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v246 = __riscv_vwmul_vv_i16m2(v244, v245, v240);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v247 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v248 = __riscv_vwredsum_vs_i16m2_i32m1(v246, v247, v240);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v249 = __riscv_vmv_x_s_i32m1_i32(v248);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v250 = v238 * v249;
    int v251 = v58;
    int v252 = v251 + v250;
    v58 = v252;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v253 = v57[8];
    int v254 = (int) v253;
    int v255 = v254 * v239;
    int v256 = v59;
    int v257 = v256 + v255;
    v59 = v257;
    const uint8_t v258 = v53[9];
    int v259 = (int) v258;
    int v260 = v259 & 0xF;
    int v261 = v259 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v262 = __riscv_vsetvl_e8m1(16);
    size_t v263 = 9 * 16;
    const int8_t* v264 = v55 + v263;
    const int8_t* v265 = v8 + v263;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v266 = __riscv_vle8_v_i8m1(v264, v262);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v267 = __riscv_vle8_v_i8m1(v265, v262);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v268 = __riscv_vwmul_vv_i16m2(v266, v267, v262);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v269 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v270 = __riscv_vwredsum_vs_i16m2_i32m1(v268, v269, v262);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v271 = __riscv_vmv_x_s_i32m1_i32(v270);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v272 = v260 * v271;
    int v273 = v58;
    int v274 = v273 + v272;
    v58 = v274;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v275 = v57[9];
    int v276 = (int) v275;
    int v277 = v276 * v261;
    int v278 = v59;
    int v279 = v278 + v277;
    v59 = v279;
    const uint8_t v280 = v53[10];
    int v281 = (int) v280;
    int v282 = v281 & 0xF;
    int v283 = v281 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v284 = __riscv_vsetvl_e8m1(16);
    size_t v285 = 10 * 16;
    const int8_t* v286 = v55 + v285;
    const int8_t* v287 = v8 + v285;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v288 = __riscv_vle8_v_i8m1(v286, v284);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v289 = __riscv_vle8_v_i8m1(v287, v284);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v290 = __riscv_vwmul_vv_i16m2(v288, v289, v284);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v291 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v292 = __riscv_vwredsum_vs_i16m2_i32m1(v290, v291, v284);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v293 = __riscv_vmv_x_s_i32m1_i32(v292);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v294 = v282 * v293;
    int v295 = v58;
    int v296 = v295 + v294;
    v58 = v296;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v297 = v57[10];
    int v298 = (int) v297;
    int v299 = v298 * v283;
    int v300 = v59;
    int v301 = v300 + v299;
    v59 = v301;
    const uint8_t v302 = v53[11];
    int v303 = (int) v302;
    int v304 = v303 & 0xF;
    int v305 = v303 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v306 = __riscv_vsetvl_e8m1(16);
    size_t v307 = 11 * 16;
    const int8_t* v308 = v55 + v307;
    const int8_t* v309 = v8 + v307;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v310 = __riscv_vle8_v_i8m1(v308, v306);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v311 = __riscv_vle8_v_i8m1(v309, v306);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v312 = __riscv_vwmul_vv_i16m2(v310, v311, v306);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v313 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v314 = __riscv_vwredsum_vs_i16m2_i32m1(v312, v313, v306);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v315 = __riscv_vmv_x_s_i32m1_i32(v314);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v316 = v304 * v315;
    int v317 = v58;
    int v318 = v317 + v316;
    v58 = v318;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v319 = v57[11];
    int v320 = (int) v319;
    int v321 = v320 * v305;
    int v322 = v59;
    int v323 = v322 + v321;
    v59 = v323;
    const uint8_t v324 = v53[12];
    int v325 = (int) v324;
    int v326 = v325 & 0xF;
    int v327 = v325 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v328 = __riscv_vsetvl_e8m1(16);
    size_t v329 = 12 * 16;
    const int8_t* v330 = v55 + v329;
    const int8_t* v331 = v8 + v329;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v332 = __riscv_vle8_v_i8m1(v330, v328);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v333 = __riscv_vle8_v_i8m1(v331, v328);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v334 = __riscv_vwmul_vv_i16m2(v332, v333, v328);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v335 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v336 = __riscv_vwredsum_vs_i16m2_i32m1(v334, v335, v328);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v337 = __riscv_vmv_x_s_i32m1_i32(v336);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v338 = v326 * v337;
    int v339 = v58;
    int v340 = v339 + v338;
    v58 = v340;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v341 = v57[12];
    int v342 = (int) v341;
    int v343 = v342 * v327;
    int v344 = v59;
    int v345 = v344 + v343;
    v59 = v345;
    const uint8_t v346 = v53[13];
    int v347 = (int) v346;
    int v348 = v347 & 0xF;
    int v349 = v347 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v350 = __riscv_vsetvl_e8m1(16);
    size_t v351 = 13 * 16;
    const int8_t* v352 = v55 + v351;
    const int8_t* v353 = v8 + v351;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v354 = __riscv_vle8_v_i8m1(v352, v350);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v355 = __riscv_vle8_v_i8m1(v353, v350);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v356 = __riscv_vwmul_vv_i16m2(v354, v355, v350);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v357 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v358 = __riscv_vwredsum_vs_i16m2_i32m1(v356, v357, v350);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v359 = __riscv_vmv_x_s_i32m1_i32(v358);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v360 = v348 * v359;
    int v361 = v58;
    int v362 = v361 + v360;
    v58 = v362;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v363 = v57[13];
    int v364 = (int) v363;
    int v365 = v364 * v349;
    int v366 = v59;
    int v367 = v366 + v365;
    v59 = v367;
    const uint8_t v368 = v53[14];
    int v369 = (int) v368;
    int v370 = v369 & 0xF;
    int v371 = v369 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v372 = __riscv_vsetvl_e8m1(16);
    size_t v373 = 14 * 16;
    const int8_t* v374 = v55 + v373;
    const int8_t* v375 = v8 + v373;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v376 = __riscv_vle8_v_i8m1(v374, v372);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v377 = __riscv_vle8_v_i8m1(v375, v372);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v378 = __riscv_vwmul_vv_i16m2(v376, v377, v372);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v379 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v380 = __riscv_vwredsum_vs_i16m2_i32m1(v378, v379, v372);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v381 = __riscv_vmv_x_s_i32m1_i32(v380);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v382 = v370 * v381;
    int v383 = v58;
    int v384 = v383 + v382;
    v58 = v384;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v385 = v57[14];
    int v386 = (int) v385;
    int v387 = v386 * v371;
    int v388 = v59;
    int v389 = v388 + v387;
    v59 = v389;
    const uint8_t v390 = v53[15];
    int v391 = (int) v390;
    int v392 = v391 & 0xF;
    int v393 = v391 >> 4;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_dot
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v394 = __riscv_vsetvl_e8m1(16);
    size_t v395 = 15 * 16;
    const int8_t* v396 = v55 + v395;
    const int8_t* v397 = v8 + v395;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v398 = __riscv_vle8_v_i8m1(v396, v394);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
    vint8m1_t v399 = __riscv_vle8_v_i8m1(v397, v394);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m2
    vint16m2_t v400 = __riscv_vwmul_vv_i16m2(v398, v399, v394);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
    vint32m1_t v401 = __riscv_vmv_v_x_i32m1(0, 1);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwredsum_vs_i16m2_i32m1
    vint32m1_t v402 = __riscv_vwredsum_vs_i16m2_i32m1(v400, v401, v394);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_x_s_i32m1_i32
    int v403 = __riscv_vmv_x_s_i32m1_i32(v402);
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=isum_accumulate
    int v404 = v392 * v403;
    int v405 = v58;
    int v406 = v405 + v404;
    v58 = v406;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=summs_accumulate
    const int16_t v407 = v57[15];
    int v408 = (int) v407;
    int v409 = v408 * v393;
    int v410 = v59;
    int v411 = v410 + v409;
    v59 = v411;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_activation_d
    const float* v412 = (const float*) v14;
    const float v413 = v412[0];
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dall
    const uint8_t* v414 = v12 + 80;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v415 = (float)*(const _Float16 *)(v414);
    float v416 = v415 * v413;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v417 = v12 + 82;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=fcvt.s.h
    float v418 = (float)*(const _Float16 *)(v417);
    float v419 = v418 * v413;
    // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scalar_fold
    int v420 = v58;
    int v421 = v59;
    float v422 = v9;
    // tcrv_emitc.assign target=sumf source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
    v9 = v422 + (v416 * (float) v420 - v419 * (float) v421);
  }
  float v423 = v9;
  // tcrv_emitc.source_op=tcrv_rvv.q2_k_q8_k_block_dot role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_s
  v2[0] = v423;
  return;
}


