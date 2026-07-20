#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_count
  size_t v6 = v1 / 256;
  // weft_emitc.local_variable=aux8 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  int8_t v7[256];
  const int8_t* v8 = &v7[0];
  // weft_emitc.local_variable=utmp source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  uint32_t v9[4];
  // weft_emitc.local_variable=sums8 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v10[8];
  // weft_emitc.local_variable=sums source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  vfloat32m2_t v11;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
  vfloat32m2_t v12 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  v11 = v12;
  // weft_emitc.local_variable=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  float v13;
  v13 = 0.0f;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_loop
  for (size_t v14 = 0; v14 < v6; v14 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_x
    size_t v15 = v14 * 176;
    const uint8_t* v16 = v3 + v15;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=super_block_base_y
    size_t v17 = v14 * 292;
    const uint8_t* v18 = v4 + v17;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=unpack_4bit
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v19 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v20 = v16 + 48;
    const uint8_t* v21 = (const uint8_t*) v20;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v22 = __riscv_vle8_v_u8m2(v21, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v23 = v16 + 16;
    const uint8_t* v24 = (const uint8_t*) v23;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v25 = __riscv_vle8_v_u8m2(v24, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v26 = __riscv_vand_vx_u8m2(v22, 0x0F, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v27 = __riscv_vand_vx_u8m2(v25, 1, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v28 = __riscv_vmsne_vx_u8m2_b4(v27, 0, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v29 = __riscv_vadd_vx_u8m2_mu(v28, v26, v26, 16, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v30 = __riscv_vreinterpret_v_u8m2_i8m2(v29);
    int8_t* v31 = &v7[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v31, v30, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v32 = __riscv_vsrl_vx_u8m2(v22, 0x04, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v33 = __riscv_vand_vx_u8m2(v25, 2, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v34 = __riscv_vmsne_vx_u8m2_b4(v33, 0, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v35 = __riscv_vadd_vx_u8m2_mu(v34, v32, v32, 16, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v36 = __riscv_vreinterpret_v_u8m2_i8m2(v35);
    int8_t* v37 = &v7[32];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v37, v36, v19);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v38 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v39 = v16 + 80;
    const uint8_t* v40 = (const uint8_t*) v39;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v41 = __riscv_vle8_v_u8m2(v40, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v42 = v16 + 16;
    const uint8_t* v43 = (const uint8_t*) v42;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v44 = __riscv_vle8_v_u8m2(v43, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v45 = __riscv_vand_vx_u8m2(v41, 0x0F, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v46 = __riscv_vand_vx_u8m2(v44, 4, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v47 = __riscv_vmsne_vx_u8m2_b4(v46, 0, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v48 = __riscv_vadd_vx_u8m2_mu(v47, v45, v45, 16, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v49 = __riscv_vreinterpret_v_u8m2_i8m2(v48);
    int8_t* v50 = &v7[64];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v50, v49, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v51 = __riscv_vsrl_vx_u8m2(v41, 0x04, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v52 = __riscv_vand_vx_u8m2(v44, 8, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v53 = __riscv_vmsne_vx_u8m2_b4(v52, 0, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v54 = __riscv_vadd_vx_u8m2_mu(v53, v51, v51, 16, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v55 = __riscv_vreinterpret_v_u8m2_i8m2(v54);
    int8_t* v56 = &v7[96];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v56, v55, v38);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v57 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v58 = v16 + 112;
    const uint8_t* v59 = (const uint8_t*) v58;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v60 = __riscv_vle8_v_u8m2(v59, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v61 = v16 + 16;
    const uint8_t* v62 = (const uint8_t*) v61;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v63 = __riscv_vle8_v_u8m2(v62, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v64 = __riscv_vand_vx_u8m2(v60, 0x0F, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v65 = __riscv_vand_vx_u8m2(v63, 16, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v66 = __riscv_vmsne_vx_u8m2_b4(v65, 0, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v67 = __riscv_vadd_vx_u8m2_mu(v66, v64, v64, 16, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v68 = __riscv_vreinterpret_v_u8m2_i8m2(v67);
    int8_t* v69 = &v7[128];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v69, v68, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v70 = __riscv_vsrl_vx_u8m2(v60, 0x04, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v71 = __riscv_vand_vx_u8m2(v63, 32, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v72 = __riscv_vmsne_vx_u8m2_b4(v71, 0, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v73 = __riscv_vadd_vx_u8m2_mu(v72, v70, v70, 16, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v74 = __riscv_vreinterpret_v_u8m2_i8m2(v73);
    int8_t* v75 = &v7[160];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v75, v74, v57);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8m2
    size_t v76 = __riscv_vsetvl_e8m2(32);
    const uint8_t* v77 = v16 + 144;
    const uint8_t* v78 = (const uint8_t*) v77;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v79 = __riscv_vle8_v_u8m2(v78, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_high_bit_plane
    const uint8_t* v80 = v16 + 16;
    const uint8_t* v81 = (const uint8_t*) v80;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m2
    vuint8m2_t v82 = __riscv_vle8_v_u8m2(v81, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v83 = __riscv_vand_vx_u8m2(v79, 0x0F, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v84 = __riscv_vand_vx_u8m2(v82, 64, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v85 = __riscv_vmsne_vx_u8m2_b4(v84, 0, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v86 = __riscv_vadd_vx_u8m2_mu(v85, v83, v83, 16, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v87 = __riscv_vreinterpret_v_u8m2_i8m2(v86);
    int8_t* v88 = &v7[192];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v88, v87, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m2
    vuint8m2_t v89 = __riscv_vsrl_vx_u8m2(v79, 0x04, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m2
    vuint8m2_t v90 = __riscv_vand_vx_u8m2(v82, 128, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmsne_vx_u8m2_b4
    vbool4_t v91 = __riscv_vmsne_vx_u8m2_b4(v90, 0, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u8m2_mu
    vuint8m2_t v92 = __riscv_vadd_vx_u8m2_mu(v91, v89, v89, 16, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m2_i8m2
    vint8m2_t v93 = __riscv_vreinterpret_v_u8m2_i8m2(v92);
    int8_t* v94 = &v7[224];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse8_v_i8m2
    __riscv_vse8_v_i8m2(v94, v93, v76);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_min_bit_dance
    const uint8_t* v95 = v16 + 4;
    const uint32_t* v96 = (const uint32_t*) v95;
    const uint32_t v97 = v96[0];
    uint32_t v98 = (uint32_t) v97;
    const uint32_t v99 = v96[1];
    uint32_t v100 = (uint32_t) v99;
    const uint32_t v101 = v96[2];
    uint32_t v102 = (uint32_t) v101;
    uint32_t v103 = v100 >> 6;
    uint32_t v104 = v103 & 0x03030303;
    uint32_t v105 = v104 << 4;
    uint32_t v106 = v102 >> 4;
    uint32_t v107 = v106 & 0x0f0f0f0f;
    uint32_t v108 = v107 | v105;
    uint32_t v109 = v100 & 0x3f3f3f3f;
    uint32_t v110 = v98 >> 6;
    uint32_t v111 = v110 & 0x03030303;
    uint32_t v112 = v111 << 4;
    uint32_t v113 = v102 & 0x0f0f0f0f;
    uint32_t v114 = v113 | v112;
    uint32_t v115 = v98 & 0x3f3f3f3f;
    v9[0] = v115;
    v9[1] = v114;
    v9[2] = v109;
    v9[3] = v108;
    uint32_t* v116 = &v9[0];
    const uint8_t* v117 = (const uint8_t*) v116;
    // weft_emitc.local_variable=aux32 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    vint32m2_t v118;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
    vint32m2_t v119 = __riscv_vmv_v_x_i32m2(0, 8);
    v118 = v119;
    const uint8_t* v120 = v18 + 4;
    const int8_t* v121 = (const int8_t*) v120;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_loop
    for (size_t v122 = 0; v122 < 8; v122 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=scale_load
      const uint8_t v123 = v117[v122];
      int v124 = (int) v123;
      size_t v125 = v122 * 32;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_quarter
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v126 = __riscv_vsetvl_e8mf2(8);
      const int8_t* v127 = v121 + v125;
      const int8_t* v128 = v8 + v125;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v129 = __riscv_vle8_v_i8mf2(v127, v126);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v130 = __riscv_vle8_v_i8mf2(v128, v126);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v131 = __riscv_vwmul_vv_i16m1(v129, v130, v126);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v132 = v118;
      vint32m2_t v133 = __riscv_vwmacc_vx_i32m2(v132, v124, v131, v126);
      // weft_emitc.assign target=aux32 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
      v118 = v133;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_quarter
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v134 = __riscv_vsetvl_e8mf2(8);
      size_t v135 = v125 + 8;
      const int8_t* v136 = v121 + v135;
      const int8_t* v137 = v8 + v135;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v138 = __riscv_vle8_v_i8mf2(v136, v134);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v139 = __riscv_vle8_v_i8mf2(v137, v134);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v140 = __riscv_vwmul_vv_i16m1(v138, v139, v134);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v141 = v118;
      vint32m2_t v142 = __riscv_vwmacc_vx_i32m2(v141, v124, v140, v134);
      // weft_emitc.assign target=aux32 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
      v118 = v142;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_quarter
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v143 = __riscv_vsetvl_e8mf2(8);
      size_t v144 = v125 + 16;
      const int8_t* v145 = v121 + v144;
      const int8_t* v146 = v8 + v144;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v147 = __riscv_vle8_v_i8mf2(v145, v143);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v148 = __riscv_vle8_v_i8mf2(v146, v143);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v149 = __riscv_vwmul_vv_i16m1(v147, v148, v143);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v150 = v118;
      vint32m2_t v151 = __riscv_vwmacc_vx_i32m2(v150, v124, v149, v143);
      // weft_emitc.assign target=aux32 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
      v118 = v151;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=sub_block_quarter
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e8mf2
      size_t v152 = __riscv_vsetvl_e8mf2(8);
      size_t v153 = v125 + 24;
      const int8_t* v154 = v121 + v153;
      const int8_t* v155 = v8 + v153;
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v156 = __riscv_vle8_v_i8mf2(v154, v152);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
      vint8mf2_t v157 = __riscv_vle8_v_i8mf2(v155, v152);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vv_i16m1
      vint16m1_t v158 = __riscv_vwmul_vv_i16m1(v156, v157, v152);
      // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i32m2
      vint32m2_t v159 = v118;
      vint32m2_t v160 = __riscv_vwmacc_vx_i32m2(v159, v124, v158, v152);
      // weft_emitc.assign target=aux32 source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
      v118 = v160;
    }
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fold_activation_d
    const float* v161 = (const float*) v18;
    const float v162 = v161[0];
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_term_bsums
    const uint8_t* v163 = v18 + 260;
    const int16_t* v164 = (const int16_t*) v163;
    // weft_emitc.local_variable=sumi source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    int v165;
    v165 = 0;
    const int16_t v166 = v164[0];
    int v167 = (int) v166;
    const uint8_t v168 = v117[8];
    int v169 = (int) v168;
    int v170 = v167 * v169;
    int v171 = v165;
    int v172 = v171 + v170;
    v165 = v172;
    const int16_t v173 = v164[1];
    int v174 = (int) v173;
    const uint8_t v175 = v117[8];
    int v176 = (int) v175;
    int v177 = v174 * v176;
    int v178 = v165;
    int v179 = v178 + v177;
    v165 = v179;
    const int16_t v180 = v164[2];
    int v181 = (int) v180;
    const uint8_t v182 = v117[9];
    int v183 = (int) v182;
    int v184 = v181 * v183;
    int v185 = v165;
    int v186 = v185 + v184;
    v165 = v186;
    const int16_t v187 = v164[3];
    int v188 = (int) v187;
    const uint8_t v189 = v117[9];
    int v190 = (int) v189;
    int v191 = v188 * v190;
    int v192 = v165;
    int v193 = v192 + v191;
    v165 = v193;
    const int16_t v194 = v164[4];
    int v195 = (int) v194;
    const uint8_t v196 = v117[10];
    int v197 = (int) v196;
    int v198 = v195 * v197;
    int v199 = v165;
    int v200 = v199 + v198;
    v165 = v200;
    const int16_t v201 = v164[5];
    int v202 = (int) v201;
    const uint8_t v203 = v117[10];
    int v204 = (int) v203;
    int v205 = v202 * v204;
    int v206 = v165;
    int v207 = v206 + v205;
    v165 = v207;
    const int16_t v208 = v164[6];
    int v209 = (int) v208;
    const uint8_t v210 = v117[11];
    int v211 = (int) v210;
    int v212 = v209 * v211;
    int v213 = v165;
    int v214 = v213 + v212;
    v165 = v214;
    const int16_t v215 = v164[7];
    int v216 = (int) v215;
    const uint8_t v217 = v117[11];
    int v218 = (int) v217;
    int v219 = v216 * v218;
    int v220 = v165;
    int v221 = v220 + v219;
    v165 = v221;
    const int16_t v222 = v164[8];
    int v223 = (int) v222;
    const uint8_t v224 = v117[12];
    int v225 = (int) v224;
    int v226 = v223 * v225;
    int v227 = v165;
    int v228 = v227 + v226;
    v165 = v228;
    const int16_t v229 = v164[9];
    int v230 = (int) v229;
    const uint8_t v231 = v117[12];
    int v232 = (int) v231;
    int v233 = v230 * v232;
    int v234 = v165;
    int v235 = v234 + v233;
    v165 = v235;
    const int16_t v236 = v164[10];
    int v237 = (int) v236;
    const uint8_t v238 = v117[13];
    int v239 = (int) v238;
    int v240 = v237 * v239;
    int v241 = v165;
    int v242 = v241 + v240;
    v165 = v242;
    const int16_t v243 = v164[11];
    int v244 = (int) v243;
    const uint8_t v245 = v117[13];
    int v246 = (int) v245;
    int v247 = v244 * v246;
    int v248 = v165;
    int v249 = v248 + v247;
    v165 = v249;
    const int16_t v250 = v164[12];
    int v251 = (int) v250;
    const uint8_t v252 = v117[14];
    int v253 = (int) v252;
    int v254 = v251 * v253;
    int v255 = v165;
    int v256 = v255 + v254;
    v165 = v256;
    const int16_t v257 = v164[13];
    int v258 = (int) v257;
    const uint8_t v259 = v117[14];
    int v260 = (int) v259;
    int v261 = v258 * v260;
    int v262 = v165;
    int v263 = v262 + v261;
    v165 = v263;
    const int16_t v264 = v164[14];
    int v265 = (int) v264;
    const uint8_t v266 = v117[15];
    int v267 = (int) v266;
    int v268 = v265 * v267;
    int v269 = v165;
    int v270 = v269 + v268;
    v165 = v270;
    const int16_t v271 = v164[15];
    int v272 = (int) v271;
    const uint8_t v273 = v117[15];
    int v274 = (int) v273;
    int v275 = v272 * v274;
    int v276 = v165;
    int v277 = v276 + v275;
    v165 = v277;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fold_scale_d
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v278 = (float)*(const _Float16 *)(v16);
    float v279 = v278 * v162;
    vint32m2_t v280 = v118;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
    vfloat32m2_t v281 = __riscv_vfcvt_f_x_v_f32m2(v280, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v282 = __riscv_vfmul_vf_f32m2(v281, v279, 8);
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
    vfloat32m2_t v283 = v11;
    vfloat32m2_t v284 = __riscv_vfadd_vv_f32m2(v283, v282, 8);
    // weft_emitc.assign target=sums source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v11 = v284;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fold_scale_dmin
    const uint8_t* v285 = v16 + 2;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=fcvt.s.h
    float v286 = (float)*(const _Float16 *)(v285);
    float v287 = v286 * v162;
    // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=min_subtract
    int v288 = v165;
    float v289 = v13;
    // weft_emitc.assign target=sumf source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
    v13 = v289 - v287 * (float) v288;
  }
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_sums_lanes
  float* v290 = &v10[0];
  vfloat32m2_t v291 = v11;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
  __riscv_vse32_v_f32m2(v290, v291, 8);
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=horizontal_sum
  float v292 = v13;
  float v293 = v10[0];
  float v294 = v292 + v293;
  float v295 = v10[1];
  float v296 = v294 + v295;
  float v297 = v10[2];
  float v298 = v296 + v297;
  float v299 = v10[3];
  float v300 = v298 + v299;
  float v301 = v10[4];
  float v302 = v300 + v301;
  float v303 = v10[5];
  float v304 = v302 + v303;
  float v305 = v10[6];
  float v306 = v304 + v305;
  float v307 = v10[7];
  float v308 = v306 + v307;
  // weft_emitc.source_op=weft_rvv.typed_super_block_block_dot_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=store_s
  v2[0] = v308;
  return;
}


