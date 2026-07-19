#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q50_gemm_m1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v7 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 352;
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
          size_t v25 = v24 * 352;
          const uint8_t* v26 = v19 + v25;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v27 = v24 * 136;
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
            size_t v35 = 32 + v34;
            size_t v36 = v35 + v21;
            const uint8_t* v37 = v26 + v36;
            const uint8_t* v38 = (const uint8_t*) v37;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v39 = __riscv_vle8_v_u8m1(v38, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v40 = v33 * 2;
            size_t v41 = 288 + v40;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v42 = 288 + 32;
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
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v60 = __riscv_vsub_vx_i8m1(v59, 16, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v61 = __riscv_vmv_v_x_u16m2(v49, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v62 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v63 = __riscv_vadd_vx_u16m2(v62, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v64 = __riscv_vsrl_vv_u16m2(v61, v63, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v65 = __riscv_vand_vx_u16m2(v64, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v66 = __riscv_vsll_vx_u16m2(v65, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v67 = __riscv_vncvt_x_x_w_u8m1(v66, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v68 = __riscv_vsrl_vx_u8m1(v39, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v69 = __riscv_vor_vv_u8m1(v68, v67, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v70 = __riscv_vreinterpret_v_u8m1_i8m1(v69);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v71 = __riscv_vsub_vx_i8m1(v70, 16, 16);
            size_t v72 = v33 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v73 = v72 + 0;
            size_t v74 = 8 + v73;
            const uint8_t* v75 = v28 + v74;
            const int8_t* v76 = (const int8_t*) v75;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v77 = *(const int8_t *)(v76);
            vint16m2_t v78 = v29;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v79 = __riscv_vwmacc_vx_i16m2(v78, v77, v60, 16);
            v29 = v79;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v80 = v72 + 0;
            size_t v81 = 8 + 64;
            size_t v82 = v81 + v80;
            const uint8_t* v83 = v28 + v82;
            const int8_t* v84 = (const int8_t*) v83;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v85 = *(const int8_t *)(v84);
            vint16m2_t v86 = v31;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v87 = __riscv_vwmacc_vx_i16m2(v86, v85, v71, 16);
            v31 = v87;
          }
          vint16m2_t v88 = v29;
          vint16m2_t v89 = v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v90 = __riscv_vwadd_vv_i32m4(v88, v89, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v91 = v21 * 2;
          const uint8_t* v92 = v26 + v91;
          const _Float16* v93 = (const _Float16*) v92;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v94 = __riscv_vle16_v_f16m2(v93, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v95 = v28 + 0;
          const _Float16* v96 = (const _Float16*) v95;
          _Float16 v97 = *(const _Float16 *)(v96);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v98 = __riscv_vfwmul_vf_f32m4(v94, v97, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v99 = __riscv_vfcvt_f_x_v_f32m4(v90, 16);
          vfloat32m4_t v100 = v23;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v101 = __riscv_vfmacc_vv_f32m4(v100, v99, v98, 16);
          v23 = v101;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v102 = v12 * 4;
        size_t v103 = v102 + 0;
        size_t v104 = v103 * v2;
        size_t v105 = v16 * 16;
        size_t v106 = v104 + v105;
        size_t v107 = v106 + v21;
        float* v108 = v4 + v107;
        vfloat32m4_t v109 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v108, v109, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v110 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v111;
        v111 = v110;
        for (size_t v112 = 0; v112 < v9; v112 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v113 = v112 * 352;
          const uint8_t* v114 = v19 + v113;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v115 = v112 * 136;
          const uint8_t* v116 = v15 + v115;
          vint16m2_t v117;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v118 = __riscv_vmv_v_x_i16m2(0, 16);
          v117 = v118;
          vint16m2_t v119;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v120 = __riscv_vmv_v_x_i16m2(0, 16);
          v119 = v120;
          for (size_t v121 = 0; v121 < 16; v121 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v122 = v121 * 16;
            size_t v123 = 32 + v122;
            size_t v124 = v123 + v21;
            const uint8_t* v125 = v114 + v124;
            const uint8_t* v126 = (const uint8_t*) v125;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v127 = __riscv_vle8_v_u8m1(v126, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v128 = v121 * 2;
            size_t v129 = 288 + v128;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v130 = 288 + 32;
            size_t v131 = v130 + v128;
            const uint8_t* v132 = v114 + v129;
            const uint16_t* v133 = (const uint16_t*) v132;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v134 = (uint16_t)*(const uint16_t *)(v133);
            const uint8_t* v135 = v114 + v131;
            const uint16_t* v136 = (const uint16_t*) v135;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v137 = (uint16_t)*(const uint16_t *)(v136);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v138 = __riscv_vmv_v_x_u16m2(v134, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v139 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v140 = __riscv_vadd_vx_u16m2(v139, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v141 = __riscv_vsrl_vv_u16m2(v138, v140, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v142 = __riscv_vand_vx_u16m2(v141, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v143 = __riscv_vsll_vx_u16m2(v142, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v144 = __riscv_vncvt_x_x_w_u8m1(v143, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v145 = __riscv_vand_vx_u8m1(v127, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v146 = __riscv_vor_vv_u8m1(v145, v144, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v147 = __riscv_vreinterpret_v_u8m1_i8m1(v146);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v148 = __riscv_vsub_vx_i8m1(v147, 16, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v149 = __riscv_vmv_v_x_u16m2(v137, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v150 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v151 = __riscv_vadd_vx_u16m2(v150, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v152 = __riscv_vsrl_vv_u16m2(v149, v151, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v153 = __riscv_vand_vx_u16m2(v152, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v154 = __riscv_vsll_vx_u16m2(v153, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v155 = __riscv_vncvt_x_x_w_u8m1(v154, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v156 = __riscv_vsrl_vx_u8m1(v127, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v157 = __riscv_vor_vv_u8m1(v156, v155, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v158 = __riscv_vreinterpret_v_u8m1_i8m1(v157);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v159 = __riscv_vsub_vx_i8m1(v158, 16, 16);
            size_t v160 = v121 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v161 = v160 + 1;
            size_t v162 = 8 + v161;
            const uint8_t* v163 = v116 + v162;
            const int8_t* v164 = (const int8_t*) v163;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v165 = *(const int8_t *)(v164);
            vint16m2_t v166 = v117;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v167 = __riscv_vwmacc_vx_i16m2(v166, v165, v148, 16);
            v117 = v167;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v168 = v160 + 1;
            size_t v169 = 8 + 64;
            size_t v170 = v169 + v168;
            const uint8_t* v171 = v116 + v170;
            const int8_t* v172 = (const int8_t*) v171;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v173 = *(const int8_t *)(v172);
            vint16m2_t v174 = v119;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v175 = __riscv_vwmacc_vx_i16m2(v174, v173, v159, 16);
            v119 = v175;
          }
          vint16m2_t v176 = v117;
          vint16m2_t v177 = v119;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v178 = __riscv_vwadd_vv_i32m4(v176, v177, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v179 = v21 * 2;
          const uint8_t* v180 = v114 + v179;
          const _Float16* v181 = (const _Float16*) v180;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v182 = __riscv_vle16_v_f16m2(v181, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v183 = v116 + 2;
          const _Float16* v184 = (const _Float16*) v183;
          _Float16 v185 = *(const _Float16 *)(v184);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v186 = __riscv_vfwmul_vf_f32m4(v182, v185, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v187 = __riscv_vfcvt_f_x_v_f32m4(v178, 16);
          vfloat32m4_t v188 = v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v189 = __riscv_vfmacc_vv_f32m4(v188, v187, v186, 16);
          v111 = v189;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v190 = v12 * 4;
        size_t v191 = v190 + 1;
        size_t v192 = v191 * v2;
        size_t v193 = v16 * 16;
        size_t v194 = v192 + v193;
        size_t v195 = v194 + v21;
        float* v196 = v4 + v195;
        vfloat32m4_t v197 = v111;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v196, v197, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v198 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v199;
        v199 = v198;
        for (size_t v200 = 0; v200 < v9; v200 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v201 = v200 * 352;
          const uint8_t* v202 = v19 + v201;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v203 = v200 * 136;
          const uint8_t* v204 = v15 + v203;
          vint16m2_t v205;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v206 = __riscv_vmv_v_x_i16m2(0, 16);
          v205 = v206;
          vint16m2_t v207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v208 = __riscv_vmv_v_x_i16m2(0, 16);
          v207 = v208;
          for (size_t v209 = 0; v209 < 16; v209 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v210 = v209 * 16;
            size_t v211 = 32 + v210;
            size_t v212 = v211 + v21;
            const uint8_t* v213 = v202 + v212;
            const uint8_t* v214 = (const uint8_t*) v213;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v215 = __riscv_vle8_v_u8m1(v214, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v216 = v209 * 2;
            size_t v217 = 288 + v216;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v218 = 288 + 32;
            size_t v219 = v218 + v216;
            const uint8_t* v220 = v202 + v217;
            const uint16_t* v221 = (const uint16_t*) v220;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v222 = (uint16_t)*(const uint16_t *)(v221);
            const uint8_t* v223 = v202 + v219;
            const uint16_t* v224 = (const uint16_t*) v223;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v225 = (uint16_t)*(const uint16_t *)(v224);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v226 = __riscv_vmv_v_x_u16m2(v222, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v227 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v228 = __riscv_vadd_vx_u16m2(v227, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v229 = __riscv_vsrl_vv_u16m2(v226, v228, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v230 = __riscv_vand_vx_u16m2(v229, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v231 = __riscv_vsll_vx_u16m2(v230, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v232 = __riscv_vncvt_x_x_w_u8m1(v231, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v233 = __riscv_vand_vx_u8m1(v215, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v234 = __riscv_vor_vv_u8m1(v233, v232, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v235 = __riscv_vreinterpret_v_u8m1_i8m1(v234);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v236 = __riscv_vsub_vx_i8m1(v235, 16, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v237 = __riscv_vmv_v_x_u16m2(v225, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v238 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v239 = __riscv_vadd_vx_u16m2(v238, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v240 = __riscv_vsrl_vv_u16m2(v237, v239, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v241 = __riscv_vand_vx_u16m2(v240, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v242 = __riscv_vsll_vx_u16m2(v241, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v243 = __riscv_vncvt_x_x_w_u8m1(v242, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v244 = __riscv_vsrl_vx_u8m1(v215, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v245 = __riscv_vor_vv_u8m1(v244, v243, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v246 = __riscv_vreinterpret_v_u8m1_i8m1(v245);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v247 = __riscv_vsub_vx_i8m1(v246, 16, 16);
            size_t v248 = v209 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v249 = v248 + 2;
            size_t v250 = 8 + v249;
            const uint8_t* v251 = v204 + v250;
            const int8_t* v252 = (const int8_t*) v251;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v253 = *(const int8_t *)(v252);
            vint16m2_t v254 = v205;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v255 = __riscv_vwmacc_vx_i16m2(v254, v253, v236, 16);
            v205 = v255;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v256 = v248 + 2;
            size_t v257 = 8 + 64;
            size_t v258 = v257 + v256;
            const uint8_t* v259 = v204 + v258;
            const int8_t* v260 = (const int8_t*) v259;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v261 = *(const int8_t *)(v260);
            vint16m2_t v262 = v207;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v263 = __riscv_vwmacc_vx_i16m2(v262, v261, v247, 16);
            v207 = v263;
          }
          vint16m2_t v264 = v205;
          vint16m2_t v265 = v207;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v266 = __riscv_vwadd_vv_i32m4(v264, v265, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v267 = v21 * 2;
          const uint8_t* v268 = v202 + v267;
          const _Float16* v269 = (const _Float16*) v268;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v270 = __riscv_vle16_v_f16m2(v269, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v271 = v204 + 4;
          const _Float16* v272 = (const _Float16*) v271;
          _Float16 v273 = *(const _Float16 *)(v272);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v274 = __riscv_vfwmul_vf_f32m4(v270, v273, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v275 = __riscv_vfcvt_f_x_v_f32m4(v266, 16);
          vfloat32m4_t v276 = v199;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v277 = __riscv_vfmacc_vv_f32m4(v276, v275, v274, 16);
          v199 = v277;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v278 = v12 * 4;
        size_t v279 = v278 + 2;
        size_t v280 = v279 * v2;
        size_t v281 = v16 * 16;
        size_t v282 = v280 + v281;
        size_t v283 = v282 + v21;
        float* v284 = v4 + v283;
        vfloat32m4_t v285 = v199;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v284, v285, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v286 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v287;
        v287 = v286;
        for (size_t v288 = 0; v288 < v9; v288 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v289 = v288 * 352;
          const uint8_t* v290 = v19 + v289;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v291 = v288 * 136;
          const uint8_t* v292 = v15 + v291;
          vint16m2_t v293;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v294 = __riscv_vmv_v_x_i16m2(0, 16);
          v293 = v294;
          vint16m2_t v295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v296 = __riscv_vmv_v_x_i16m2(0, 16);
          v295 = v296;
          for (size_t v297 = 0; v297 < 16; v297 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v298 = v297 * 16;
            size_t v299 = 32 + v298;
            size_t v300 = v299 + v21;
            const uint8_t* v301 = v290 + v300;
            const uint8_t* v302 = (const uint8_t*) v301;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v303 = __riscv_vle8_v_u8m1(v302, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v304 = v297 * 2;
            size_t v305 = 288 + v304;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v306 = 288 + 32;
            size_t v307 = v306 + v304;
            const uint8_t* v308 = v290 + v305;
            const uint16_t* v309 = (const uint16_t*) v308;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v310 = (uint16_t)*(const uint16_t *)(v309);
            const uint8_t* v311 = v290 + v307;
            const uint16_t* v312 = (const uint16_t*) v311;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v313 = (uint16_t)*(const uint16_t *)(v312);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v314 = __riscv_vmv_v_x_u16m2(v310, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v315 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v316 = __riscv_vadd_vx_u16m2(v315, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v317 = __riscv_vsrl_vv_u16m2(v314, v316, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v318 = __riscv_vand_vx_u16m2(v317, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v319 = __riscv_vsll_vx_u16m2(v318, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v320 = __riscv_vncvt_x_x_w_u8m1(v319, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v321 = __riscv_vand_vx_u8m1(v303, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v322 = __riscv_vor_vv_u8m1(v321, v320, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v323 = __riscv_vreinterpret_v_u8m1_i8m1(v322);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v324 = __riscv_vsub_vx_i8m1(v323, 16, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m2
            vuint16m2_t v325 = __riscv_vmv_v_x_u16m2(v313, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m2
            vuint16m2_t v326 = __riscv_vid_v_u16m2(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m2
            vuint16m2_t v327 = __riscv_vadd_vx_u16m2(v326, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m2
            vuint16m2_t v328 = __riscv_vsrl_vv_u16m2(v325, v327, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m2
            vuint16m2_t v329 = __riscv_vand_vx_u16m2(v328, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m2
            vuint16m2_t v330 = __riscv_vsll_vx_u16m2(v329, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8m1
            vuint8m1_t v331 = __riscv_vncvt_x_x_w_u8m1(v330, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v332 = __riscv_vsrl_vx_u8m1(v303, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8m1
            vuint8m1_t v333 = __riscv_vor_vv_u8m1(v332, v331, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v334 = __riscv_vreinterpret_v_u8m1_i8m1(v333);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8m1
            vint8m1_t v335 = __riscv_vsub_vx_i8m1(v334, 16, 16);
            size_t v336 = v297 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v337 = v336 + 3;
            size_t v338 = 8 + v337;
            const uint8_t* v339 = v292 + v338;
            const int8_t* v340 = (const int8_t*) v339;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v341 = *(const int8_t *)(v340);
            vint16m2_t v342 = v293;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v343 = __riscv_vwmacc_vx_i16m2(v342, v341, v324, 16);
            v293 = v343;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v344 = v336 + 3;
            size_t v345 = 8 + 64;
            size_t v346 = v345 + v344;
            const uint8_t* v347 = v292 + v346;
            const int8_t* v348 = (const int8_t*) v347;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v349 = *(const int8_t *)(v348);
            vint16m2_t v350 = v295;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v351 = __riscv_vwmacc_vx_i16m2(v350, v349, v335, 16);
            v295 = v351;
          }
          vint16m2_t v352 = v293;
          vint16m2_t v353 = v295;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v354 = __riscv_vwadd_vv_i32m4(v352, v353, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v355 = v21 * 2;
          const uint8_t* v356 = v290 + v355;
          const _Float16* v357 = (const _Float16*) v356;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v358 = __riscv_vle16_v_f16m2(v357, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v359 = v292 + 6;
          const _Float16* v360 = (const _Float16*) v359;
          _Float16 v361 = *(const _Float16 *)(v360);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v362 = __riscv_vfwmul_vf_f32m4(v358, v361, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v363 = __riscv_vfcvt_f_x_v_f32m4(v354, 16);
          vfloat32m4_t v364 = v287;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v365 = __riscv_vfmacc_vv_f32m4(v364, v363, v362, 16);
          v287 = v365;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v366 = v12 * 4;
        size_t v367 = v366 + 3;
        size_t v368 = v367 * v2;
        size_t v369 = v16 * 16;
        size_t v370 = v368 + v369;
        size_t v371 = v370 + v21;
        float* v372 = v4 + v371;
        vfloat32m4_t v373 = v287;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v372, v373, 16);
      }
    }
  }
  return;
}


