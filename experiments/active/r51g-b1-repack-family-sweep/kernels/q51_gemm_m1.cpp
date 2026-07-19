#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q51_gemm_m1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v3);
  // weft_emitc.loop_order_override selector=col_outer/prior realized=row_outer gate=measured-gate-blocks-unmeasured
  // weft_emitc.route_source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=block_count
  size_t v9 = v3 / 32;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v1 / 4;
  // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v5 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 144;
    const uint8_t* v15 = v7 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 384;
      const uint8_t* v19 = v6 + v18;
      for (size_t v20 = 0; v20 < 1; v20 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v22 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v23;
        v23 = v22;
        for (size_t v24 = 0; v24 < v9; v24 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v25 = v24 * 384;
          const uint8_t* v26 = v19 + v25;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v27 = v24 * 144;
          const uint8_t* v28 = v15 + v27;
          vint16m2_t v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v30 = __riscv_vmv_v_x_i16m2(0, 16);
          v29 = v30;
          vint16m2_t v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v32 = __riscv_vmv_v_x_i16m2(0, 16);
          v31 = v32;
          for (size_t v33 = 0; v33 < 16; v33 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v34 = v33 * 16;
            size_t v35 = 64 + v34;
            size_t v36 = v35 + v21;
            const uint8_t* v37 = v26 + v36;
            const uint8_t* v38 = (const uint8_t*) v37;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v39 = __riscv_vle8_v_u8m1(v38, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v40 = v33 * 2;
            size_t v41 = 320 + v40;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v42 = 320 + 32;
            size_t v43 = v42 + v40;
            const uint8_t* v44 = v26 + v41;
            const uint16_t* v45 = (const uint16_t*) v44;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v46 = (uint16_t)*(const uint16_t *)(v45);
            const uint8_t* v47 = v26 + v43;
            const uint16_t* v48 = (const uint16_t*) v47;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v49 = (uint16_t)*(const uint16_t *)(v48);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v50 = __riscv_vmv_v_x_u16m2(v46, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v51 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v52 = __riscv_vadd_vx_u16m2(v51, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v53 = __riscv_vsrl_vv_u16m2(v50, v52, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v54 = __riscv_vand_vx_u16m2(v53, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v55 = __riscv_vsll_vx_u16m2(v54, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v56 = __riscv_vncvt_x_x_w_u8m1(v55, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v57 = __riscv_vand_vx_u8m1(v39, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v58 = __riscv_vor_vv_u8m1(v57, v56, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v59 = __riscv_vreinterpret_v_u8m1_i8m1(v58);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v60 = __riscv_vmv_v_x_u16m2(v49, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v61 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v62 = __riscv_vadd_vx_u16m2(v61, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v63 = __riscv_vsrl_vv_u16m2(v60, v62, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v64 = __riscv_vand_vx_u16m2(v63, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v65 = __riscv_vsll_vx_u16m2(v64, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v66 = __riscv_vncvt_x_x_w_u8m1(v65, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v67 = __riscv_vsrl_vx_u8m1(v39, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v68 = __riscv_vor_vv_u8m1(v67, v66, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v69 = __riscv_vreinterpret_v_u8m1_i8m1(v68);
            size_t v70 = v33 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v71 = v70 + 0;
            size_t v72 = 16 + v71;
            const uint8_t* v73 = v28 + v72;
            const int8_t* v74 = (const int8_t*) v73;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v75 = *(const int8_t *)(v74);
            vint16m2_t v76 = v29;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v77 = __riscv_vwmacc_vx_i16m2(v76, v75, v59, 16);
            v29 = v77;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v78 = v70 + 0;
            size_t v79 = 16 + 64;
            size_t v80 = v79 + v78;
            const uint8_t* v81 = v28 + v80;
            const int8_t* v82 = (const int8_t*) v81;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v83 = *(const int8_t *)(v82);
            vint16m2_t v84 = v31;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v85 = __riscv_vwmacc_vx_i16m2(v84, v83, v69, 16);
            v31 = v85;
          }
          vint16m2_t v86 = v29;
          vint16m2_t v87 = v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v88 = __riscv_vwadd_vv_i32m4(v86, v87, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v89 = v21 * 2;
          const uint8_t* v90 = v26 + v89;
          const _Float16* v91 = (const _Float16*) v90;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v92 = __riscv_vle16_v_f16m2(v91, 16);
          size_t v93 = 32 + v89;
          const uint8_t* v94 = v26 + v93;
          const _Float16* v95 = (const _Float16*) v94;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v96 = __riscv_vle16_v_f16m2(v95, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v97 = v28 + 0;
          const _Float16* v98 = (const _Float16*) v97;
          _Float16 v99 = *(const _Float16 *)(v98);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v100 = __riscv_vfwmul_vf_f32m4(v92, v99, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v101 = __riscv_vfcvt_f_x_v_f32m4(v88, 16);
          vfloat32m4_t v102 = v23;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v103 = __riscv_vfmacc_vv_f32m4(v102, v101, v100, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v104 = v28 + 8;
          const _Float16* v105 = (const _Float16*) v104;
          _Float16 v106 = *(const _Float16 *)(v105);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v107 = __riscv_vfwmul_vf_f32m4(v96, v106, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v108 = __riscv_vfadd_vv_f32m4(v103, v107, 16);
          v23 = v108;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v109 = v12 * 4;
        size_t v110 = v109 + 0;
        size_t v111 = v110 * v2;
        size_t v112 = v16 * 16;
        size_t v113 = v111 + v112;
        size_t v114 = v113 + v21;
        float* v115 = v4 + v114;
        vfloat32m4_t v116 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v115, v116, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v117 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v118;
        v118 = v117;
        for (size_t v119 = 0; v119 < v9; v119 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v120 = v119 * 384;
          const uint8_t* v121 = v19 + v120;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v122 = v119 * 144;
          const uint8_t* v123 = v15 + v122;
          vint16m2_t v124;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v125 = __riscv_vmv_v_x_i16m2(0, 16);
          v124 = v125;
          vint16m2_t v126;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v127 = __riscv_vmv_v_x_i16m2(0, 16);
          v126 = v127;
          for (size_t v128 = 0; v128 < 16; v128 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v129 = v128 * 16;
            size_t v130 = 64 + v129;
            size_t v131 = v130 + v21;
            const uint8_t* v132 = v121 + v131;
            const uint8_t* v133 = (const uint8_t*) v132;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v134 = __riscv_vle8_v_u8m1(v133, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v135 = v128 * 2;
            size_t v136 = 320 + v135;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v137 = 320 + 32;
            size_t v138 = v137 + v135;
            const uint8_t* v139 = v121 + v136;
            const uint16_t* v140 = (const uint16_t*) v139;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v141 = (uint16_t)*(const uint16_t *)(v140);
            const uint8_t* v142 = v121 + v138;
            const uint16_t* v143 = (const uint16_t*) v142;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v144 = (uint16_t)*(const uint16_t *)(v143);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v145 = __riscv_vmv_v_x_u16m2(v141, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v146 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v147 = __riscv_vadd_vx_u16m2(v146, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v148 = __riscv_vsrl_vv_u16m2(v145, v147, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v149 = __riscv_vand_vx_u16m2(v148, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v150 = __riscv_vsll_vx_u16m2(v149, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v151 = __riscv_vncvt_x_x_w_u8m1(v150, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v152 = __riscv_vand_vx_u8m1(v134, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v153 = __riscv_vor_vv_u8m1(v152, v151, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v154 = __riscv_vreinterpret_v_u8m1_i8m1(v153);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v155 = __riscv_vmv_v_x_u16m2(v144, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v156 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v157 = __riscv_vadd_vx_u16m2(v156, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v158 = __riscv_vsrl_vv_u16m2(v155, v157, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v159 = __riscv_vand_vx_u16m2(v158, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v160 = __riscv_vsll_vx_u16m2(v159, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v161 = __riscv_vncvt_x_x_w_u8m1(v160, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v162 = __riscv_vsrl_vx_u8m1(v134, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v163 = __riscv_vor_vv_u8m1(v162, v161, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v164 = __riscv_vreinterpret_v_u8m1_i8m1(v163);
            size_t v165 = v128 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v166 = v165 + 1;
            size_t v167 = 16 + v166;
            const uint8_t* v168 = v123 + v167;
            const int8_t* v169 = (const int8_t*) v168;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v170 = *(const int8_t *)(v169);
            vint16m2_t v171 = v124;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v172 = __riscv_vwmacc_vx_i16m2(v171, v170, v154, 16);
            v124 = v172;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v173 = v165 + 1;
            size_t v174 = 16 + 64;
            size_t v175 = v174 + v173;
            const uint8_t* v176 = v123 + v175;
            const int8_t* v177 = (const int8_t*) v176;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v178 = *(const int8_t *)(v177);
            vint16m2_t v179 = v126;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v180 = __riscv_vwmacc_vx_i16m2(v179, v178, v164, 16);
            v126 = v180;
          }
          vint16m2_t v181 = v124;
          vint16m2_t v182 = v126;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v183 = __riscv_vwadd_vv_i32m4(v181, v182, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v184 = v21 * 2;
          const uint8_t* v185 = v121 + v184;
          const _Float16* v186 = (const _Float16*) v185;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v187 = __riscv_vle16_v_f16m2(v186, 16);
          size_t v188 = 32 + v184;
          const uint8_t* v189 = v121 + v188;
          const _Float16* v190 = (const _Float16*) v189;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v191 = __riscv_vle16_v_f16m2(v190, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v192 = v123 + 2;
          const _Float16* v193 = (const _Float16*) v192;
          _Float16 v194 = *(const _Float16 *)(v193);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v195 = __riscv_vfwmul_vf_f32m4(v187, v194, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v196 = __riscv_vfcvt_f_x_v_f32m4(v183, 16);
          vfloat32m4_t v197 = v118;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v198 = __riscv_vfmacc_vv_f32m4(v197, v196, v195, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v199 = v123 + 10;
          const _Float16* v200 = (const _Float16*) v199;
          _Float16 v201 = *(const _Float16 *)(v200);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v202 = __riscv_vfwmul_vf_f32m4(v191, v201, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v203 = __riscv_vfadd_vv_f32m4(v198, v202, 16);
          v118 = v203;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v204 = v12 * 4;
        size_t v205 = v204 + 1;
        size_t v206 = v205 * v2;
        size_t v207 = v16 * 16;
        size_t v208 = v206 + v207;
        size_t v209 = v208 + v21;
        float* v210 = v4 + v209;
        vfloat32m4_t v211 = v118;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v210, v211, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v212 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v213;
        v213 = v212;
        for (size_t v214 = 0; v214 < v9; v214 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v215 = v214 * 384;
          const uint8_t* v216 = v19 + v215;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v217 = v214 * 144;
          const uint8_t* v218 = v15 + v217;
          vint16m2_t v219;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v220 = __riscv_vmv_v_x_i16m2(0, 16);
          v219 = v220;
          vint16m2_t v221;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v222 = __riscv_vmv_v_x_i16m2(0, 16);
          v221 = v222;
          for (size_t v223 = 0; v223 < 16; v223 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v224 = v223 * 16;
            size_t v225 = 64 + v224;
            size_t v226 = v225 + v21;
            const uint8_t* v227 = v216 + v226;
            const uint8_t* v228 = (const uint8_t*) v227;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v229 = __riscv_vle8_v_u8m1(v228, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v230 = v223 * 2;
            size_t v231 = 320 + v230;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v232 = 320 + 32;
            size_t v233 = v232 + v230;
            const uint8_t* v234 = v216 + v231;
            const uint16_t* v235 = (const uint16_t*) v234;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v236 = (uint16_t)*(const uint16_t *)(v235);
            const uint8_t* v237 = v216 + v233;
            const uint16_t* v238 = (const uint16_t*) v237;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v239 = (uint16_t)*(const uint16_t *)(v238);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v240 = __riscv_vmv_v_x_u16m2(v236, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v241 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v242 = __riscv_vadd_vx_u16m2(v241, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v243 = __riscv_vsrl_vv_u16m2(v240, v242, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v244 = __riscv_vand_vx_u16m2(v243, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v245 = __riscv_vsll_vx_u16m2(v244, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v246 = __riscv_vncvt_x_x_w_u8m1(v245, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v247 = __riscv_vand_vx_u8m1(v229, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v248 = __riscv_vor_vv_u8m1(v247, v246, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v249 = __riscv_vreinterpret_v_u8m1_i8m1(v248);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v250 = __riscv_vmv_v_x_u16m2(v239, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v251 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v252 = __riscv_vadd_vx_u16m2(v251, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v253 = __riscv_vsrl_vv_u16m2(v250, v252, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v254 = __riscv_vand_vx_u16m2(v253, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v255 = __riscv_vsll_vx_u16m2(v254, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v256 = __riscv_vncvt_x_x_w_u8m1(v255, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v257 = __riscv_vsrl_vx_u8m1(v229, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v258 = __riscv_vor_vv_u8m1(v257, v256, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v259 = __riscv_vreinterpret_v_u8m1_i8m1(v258);
            size_t v260 = v223 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v261 = v260 + 2;
            size_t v262 = 16 + v261;
            const uint8_t* v263 = v218 + v262;
            const int8_t* v264 = (const int8_t*) v263;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v265 = *(const int8_t *)(v264);
            vint16m2_t v266 = v219;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v267 = __riscv_vwmacc_vx_i16m2(v266, v265, v249, 16);
            v219 = v267;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v268 = v260 + 2;
            size_t v269 = 16 + 64;
            size_t v270 = v269 + v268;
            const uint8_t* v271 = v218 + v270;
            const int8_t* v272 = (const int8_t*) v271;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v273 = *(const int8_t *)(v272);
            vint16m2_t v274 = v221;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v275 = __riscv_vwmacc_vx_i16m2(v274, v273, v259, 16);
            v221 = v275;
          }
          vint16m2_t v276 = v219;
          vint16m2_t v277 = v221;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v278 = __riscv_vwadd_vv_i32m4(v276, v277, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v279 = v21 * 2;
          const uint8_t* v280 = v216 + v279;
          const _Float16* v281 = (const _Float16*) v280;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v282 = __riscv_vle16_v_f16m2(v281, 16);
          size_t v283 = 32 + v279;
          const uint8_t* v284 = v216 + v283;
          const _Float16* v285 = (const _Float16*) v284;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v286 = __riscv_vle16_v_f16m2(v285, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v287 = v218 + 4;
          const _Float16* v288 = (const _Float16*) v287;
          _Float16 v289 = *(const _Float16 *)(v288);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v290 = __riscv_vfwmul_vf_f32m4(v282, v289, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v291 = __riscv_vfcvt_f_x_v_f32m4(v278, 16);
          vfloat32m4_t v292 = v213;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v293 = __riscv_vfmacc_vv_f32m4(v292, v291, v290, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v294 = v218 + 12;
          const _Float16* v295 = (const _Float16*) v294;
          _Float16 v296 = *(const _Float16 *)(v295);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v297 = __riscv_vfwmul_vf_f32m4(v286, v296, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v298 = __riscv_vfadd_vv_f32m4(v293, v297, 16);
          v213 = v298;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v299 = v12 * 4;
        size_t v300 = v299 + 2;
        size_t v301 = v300 * v2;
        size_t v302 = v16 * 16;
        size_t v303 = v301 + v302;
        size_t v304 = v303 + v21;
        float* v305 = v4 + v304;
        vfloat32m4_t v306 = v213;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v305, v306, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v307 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v308;
        v308 = v307;
        for (size_t v309 = 0; v309 < v9; v309 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v310 = v309 * 384;
          const uint8_t* v311 = v19 + v310;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v312 = v309 * 144;
          const uint8_t* v313 = v15 + v312;
          vint16m2_t v314;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v315 = __riscv_vmv_v_x_i16m2(0, 16);
          v314 = v315;
          vint16m2_t v316;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v317 = __riscv_vmv_v_x_i16m2(0, 16);
          v316 = v317;
          for (size_t v318 = 0; v318 < 16; v318 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v319 = v318 * 16;
            size_t v320 = 64 + v319;
            size_t v321 = v320 + v21;
            const uint8_t* v322 = v311 + v321;
            const uint8_t* v323 = (const uint8_t*) v322;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v324 = __riscv_vle8_v_u8m1(v323, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v325 = v318 * 2;
            size_t v326 = 320 + v325;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v327 = 320 + 32;
            size_t v328 = v327 + v325;
            const uint8_t* v329 = v311 + v326;
            const uint16_t* v330 = (const uint16_t*) v329;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v331 = (uint16_t)*(const uint16_t *)(v330);
            const uint8_t* v332 = v311 + v328;
            const uint16_t* v333 = (const uint16_t*) v332;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v334 = (uint16_t)*(const uint16_t *)(v333);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v335 = __riscv_vmv_v_x_u16m2(v331, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v336 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v337 = __riscv_vadd_vx_u16m2(v336, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v338 = __riscv_vsrl_vv_u16m2(v335, v337, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v339 = __riscv_vand_vx_u16m2(v338, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v340 = __riscv_vsll_vx_u16m2(v339, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v341 = __riscv_vncvt_x_x_w_u8m1(v340, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v342 = __riscv_vand_vx_u8m1(v324, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v343 = __riscv_vor_vv_u8m1(v342, v341, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v344 = __riscv_vreinterpret_v_u8m1_i8m1(v343);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v345 = __riscv_vmv_v_x_u16m2(v334, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v346 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v347 = __riscv_vadd_vx_u16m2(v346, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v348 = __riscv_vsrl_vv_u16m2(v345, v347, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v349 = __riscv_vand_vx_u16m2(v348, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v350 = __riscv_vsll_vx_u16m2(v349, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v351 = __riscv_vncvt_x_x_w_u8m1(v350, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v352 = __riscv_vsrl_vx_u8m1(v324, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v353 = __riscv_vor_vv_u8m1(v352, v351, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v354 = __riscv_vreinterpret_v_u8m1_i8m1(v353);
            size_t v355 = v318 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v356 = v355 + 3;
            size_t v357 = 16 + v356;
            const uint8_t* v358 = v313 + v357;
            const int8_t* v359 = (const int8_t*) v358;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v360 = *(const int8_t *)(v359);
            vint16m2_t v361 = v314;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v362 = __riscv_vwmacc_vx_i16m2(v361, v360, v344, 16);
            v314 = v362;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v363 = v355 + 3;
            size_t v364 = 16 + 64;
            size_t v365 = v364 + v363;
            const uint8_t* v366 = v313 + v365;
            const int8_t* v367 = (const int8_t*) v366;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v368 = *(const int8_t *)(v367);
            vint16m2_t v369 = v316;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v370 = __riscv_vwmacc_vx_i16m2(v369, v368, v354, 16);
            v316 = v370;
          }
          vint16m2_t v371 = v314;
          vint16m2_t v372 = v316;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v373 = __riscv_vwadd_vv_i32m4(v371, v372, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v374 = v21 * 2;
          const uint8_t* v375 = v311 + v374;
          const _Float16* v376 = (const _Float16*) v375;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v377 = __riscv_vle16_v_f16m2(v376, 16);
          size_t v378 = 32 + v374;
          const uint8_t* v379 = v311 + v378;
          const _Float16* v380 = (const _Float16*) v379;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v381 = __riscv_vle16_v_f16m2(v380, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v382 = v313 + 6;
          const _Float16* v383 = (const _Float16*) v382;
          _Float16 v384 = *(const _Float16 *)(v383);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v385 = __riscv_vfwmul_vf_f32m4(v377, v384, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v386 = __riscv_vfcvt_f_x_v_f32m4(v373, 16);
          vfloat32m4_t v387 = v308;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v388 = __riscv_vfmacc_vv_f32m4(v387, v386, v385, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v389 = v313 + 14;
          const _Float16* v390 = (const _Float16*) v389;
          _Float16 v391 = *(const _Float16 *)(v390);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v392 = __riscv_vfwmul_vf_f32m4(v381, v391, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v393 = __riscv_vfadd_vv_f32m4(v388, v392, 16);
          v308 = v393;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v394 = v12 * 4;
        size_t v395 = v394 + 3;
        size_t v396 = v395 * v2;
        size_t v397 = v16 * 16;
        size_t v398 = v396 + v397;
        size_t v399 = v398 + v21;
        float* v400 = v4 + v399;
        vfloat32m4_t v401 = v308;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v400, v401, 16);
      }
    }
  }
  return;
}


