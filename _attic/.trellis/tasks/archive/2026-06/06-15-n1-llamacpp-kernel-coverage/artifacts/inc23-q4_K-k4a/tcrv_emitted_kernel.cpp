#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q4_K_q8_K_aux_kernel_ggml_vec_dot_q4_K_q8_K_aux(size_t v1, int32_t* v2, uint8_t* v3, const uint8_t* v4, const uint8_t* v5) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v6 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_count
  size_t v7 = v1 / 256;
  // tcrv_emitc.local_variable=aux8 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
  int8_t v8[256];
  const int8_t* v9 = &v8[0];
  // tcrv_emitc.local_variable=utmp source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
  uint32_t v10[4];
  // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v11 = 0; v11 < v7; v11 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_x
    size_t v12 = v11 * 144;
    const uint8_t* v13 = v4 + v12;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=super_block_base_y
    size_t v14 = v11 * 292;
    const uint8_t* v15 = v5 + v14;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=unpack_4bit
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v16 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v17 = v13 + 16;
    const uint8_t* v18 = (const uint8_t*) v17;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v19 = __riscv_vle8_v_u8m2(v18, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v20 = __riscv_vand_vx_u8m2(v19, 0x0F, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v21 = __riscv_vreinterpret_v_u8m2_i8m2(v20);
    int8_t* v22 = &v8[0];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v22, v21, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v23 = __riscv_vsrl_vx_u8m2(v19, 0x04, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v24 = __riscv_vreinterpret_v_u8m2_i8m2(v23);
    int8_t* v25 = &v8[32];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v25, v24, v16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v26 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v27 = v13 + 48;
    const uint8_t* v28 = (const uint8_t*) v27;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v29 = __riscv_vle8_v_u8m2(v28, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v30 = __riscv_vand_vx_u8m2(v29, 0x0F, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v31 = __riscv_vreinterpret_v_u8m2_i8m2(v30);
    int8_t* v32 = &v8[64];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v32, v31, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v33 = __riscv_vsrl_vx_u8m2(v29, 0x04, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v34 = __riscv_vreinterpret_v_u8m2_i8m2(v33);
    int8_t* v35 = &v8[96];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v35, v34, v26);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v36 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v37 = v13 + 80;
    const uint8_t* v38 = (const uint8_t*) v37;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v39 = __riscv_vle8_v_u8m2(v38, v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v40 = __riscv_vand_vx_u8m2(v39, 0x0F, v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v41 = __riscv_vreinterpret_v_u8m2_i8m2(v40);
    int8_t* v42 = &v8[128];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v42, v41, v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v43 = __riscv_vsrl_vx_u8m2(v39, 0x04, v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v44 = __riscv_vreinterpret_v_u8m2_i8m2(v43);
    int8_t* v45 = &v8[160];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v45, v44, v36);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v46 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v47 = v13 + 112;
    const uint8_t* v48 = (const uint8_t*) v47;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v49 = __riscv_vle8_v_u8m2(v48, v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v50 = __riscv_vand_vx_u8m2(v49, 0x0F, v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v51 = __riscv_vreinterpret_v_u8m2_i8m2(v50);
    int8_t* v52 = &v8[192];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v52, v51, v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v53 = __riscv_vsrl_vx_u8m2(v49, 0x04, v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v54 = __riscv_vreinterpret_v_u8m2_i8m2(v53);
    int8_t* v55 = &v8[224];
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v55, v54, v46);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_min_bit_dance
    const uint8_t* v56 = v13 + 4;
    const uint32_t* v57 = (const uint32_t*) v56;
    const uint32_t v58 = v57[0];
    uint32_t v59 = (uint32_t) v58;
    const uint32_t v60 = v57[1];
    uint32_t v61 = (uint32_t) v60;
    const uint32_t v62 = v57[2];
    uint32_t v63 = (uint32_t) v62;
    uint32_t v64 = v61 >> 6;
    uint32_t v65 = v64 & 0x03030303;
    uint32_t v66 = v65 << 4;
    uint32_t v67 = v63 >> 4;
    uint32_t v68 = v67 & 0x0f0f0f0f;
    uint32_t v69 = v68 | v66;
    uint32_t v70 = v61 & 0x3f3f3f3f;
    uint32_t v71 = v59 >> 6;
    uint32_t v72 = v71 & 0x03030303;
    uint32_t v73 = v72 << 4;
    uint32_t v74 = v63 & 0x0f0f0f0f;
    uint32_t v75 = v74 | v73;
    uint32_t v76 = v59 & 0x3f3f3f3f;
    v10[0] = v76;
    v10[1] = v75;
    v10[2] = v70;
    v10[3] = v69;
    uint32_t* v77 = &v10[0];
    const uint8_t* v78 = (const uint8_t*) v77;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_scale_min
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m1
    size_t v79 = __riscv_vsetvl_e8m1(16);
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
    vuint8m1_t v80 = __riscv_vle8_v_u8m1(v78, v79);
    size_t v81 = v11 * 16;
    uint8_t* v82 = v3 + v81;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse8_v_u8m1
    __riscv_vse8_v_u8m1(v82, v80, v79);
    // tcrv_emitc.local_variable=aux32 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
    vint32m2_t v83;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v84 = __riscv_vmv_v_x_i32m2(0, 8);
    v83 = v84;
    const uint8_t* v85 = v15 + 4;
    const int8_t* v86 = (const int8_t*) v85;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v87 = 0; v87 < 8; v87 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=scale_load
      const uint8_t v88 = v78[v87];
      int v89 = (int) v88;
      size_t v90 = v87 * 32;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v91 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v92 = v86 + v90;
      const int8_t* v93 = v9 + v90;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v94 = __riscv_vle8_v_i8mf2(v92, v91);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v95 = __riscv_vle8_v_i8mf2(v93, v91);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v96 = __riscv_vwmul_vv_i16m1(v94, v95, v91);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v97 = v83;
      vint32m2_t v98 = __riscv_vwmacc_vx_i32m2(v97, v89, v96, v91);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v83 = v98;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v99 = __riscv_vsetvl_e8mf2(8);
      size_t v100 = v90 + 8;
      const int8_t* v101 = v86 + v100;
      const int8_t* v102 = v9 + v100;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v103 = __riscv_vle8_v_i8mf2(v101, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v104 = __riscv_vle8_v_i8mf2(v102, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v105 = __riscv_vwmul_vv_i16m1(v103, v104, v99);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v106 = v83;
      vint32m2_t v107 = __riscv_vwmacc_vx_i32m2(v106, v89, v105, v99);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v83 = v107;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v108 = __riscv_vsetvl_e8mf2(8);
      size_t v109 = v90 + 16;
      const int8_t* v110 = v86 + v109;
      const int8_t* v111 = v9 + v109;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v112 = __riscv_vle8_v_i8mf2(v110, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v113 = __riscv_vle8_v_i8mf2(v111, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v114 = __riscv_vwmul_vv_i16m1(v112, v113, v108);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v115 = v83;
      vint32m2_t v116 = __riscv_vwmacc_vx_i32m2(v115, v89, v114, v108);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v83 = v116;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=sub_block_quarter
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v117 = __riscv_vsetvl_e8mf2(8);
      size_t v118 = v90 + 24;
      const int8_t* v119 = v86 + v118;
      const int8_t* v120 = v9 + v118;
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v121 = __riscv_vle8_v_i8mf2(v119, v117);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v122 = __riscv_vle8_v_i8mf2(v120, v117);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v123 = __riscv_vwmul_vv_i16m1(v121, v122, v117);
      // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v124 = v83;
      vint32m2_t v125 = __riscv_vwmacc_vx_i32m2(v124, v89, v123, v117);
      // tcrv_emitc.assign target=aux32 source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface
      v83 = v125;
    }
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=store_aux32
    size_t v126 = v11 * 8;
    int32_t* v127 = v2 + v126;
    vint32m2_t v128 = v83;
    // tcrv_emitc.source_op=tcrv_rvv.q4_k_q8_k_aux_partial role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m2
    __riscv_vse32_v_i32m2(v127, v128, 8);
  }
  return;
}


