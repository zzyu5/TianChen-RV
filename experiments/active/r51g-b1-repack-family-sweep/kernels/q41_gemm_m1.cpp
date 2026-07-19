#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q41_gemm_m1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
      size_t v18 = v17 * 320;
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
          size_t v25 = v24 * 320;
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
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v40 = __riscv_vand_vx_u8m1(v39, 0x0F, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v41 = __riscv_vreinterpret_v_u8m1_i8m1(v40);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v42 = __riscv_vsrl_vx_u8m1(v39, 0x04, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v43 = __riscv_vreinterpret_v_u8m1_i8m1(v42);
            size_t v44 = v33 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v45 = v44 + 0;
            size_t v46 = 16 + v45;
            const uint8_t* v47 = v28 + v46;
            const int8_t* v48 = (const int8_t*) v47;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v49 = *(const int8_t *)(v48);
            vint16m2_t v50 = v29;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v51 = __riscv_vwmacc_vx_i16m2(v50, v49, v41, 16);
            v29 = v51;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v52 = v44 + 0;
            size_t v53 = 16 + 64;
            size_t v54 = v53 + v52;
            const uint8_t* v55 = v28 + v54;
            const int8_t* v56 = (const int8_t*) v55;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v57 = *(const int8_t *)(v56);
            vint16m2_t v58 = v31;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v59 = __riscv_vwmacc_vx_i16m2(v58, v57, v43, 16);
            v31 = v59;
          }
          vint16m2_t v60 = v29;
          vint16m2_t v61 = v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v62 = __riscv_vwadd_vv_i32m4(v60, v61, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v63 = v21 * 2;
          const uint8_t* v64 = v26 + v63;
          const _Float16* v65 = (const _Float16*) v64;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v66 = __riscv_vle16_v_f16m2(v65, 16);
          size_t v67 = 32 + v63;
          const uint8_t* v68 = v26 + v67;
          const _Float16* v69 = (const _Float16*) v68;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v70 = __riscv_vle16_v_f16m2(v69, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v71 = v28 + 0;
          const _Float16* v72 = (const _Float16*) v71;
          _Float16 v73 = *(const _Float16 *)(v72);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v74 = __riscv_vfwmul_vf_f32m4(v66, v73, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v75 = __riscv_vfcvt_f_x_v_f32m4(v62, 16);
          vfloat32m4_t v76 = v23;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v77 = __riscv_vfmacc_vv_f32m4(v76, v75, v74, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v78 = v28 + 8;
          const _Float16* v79 = (const _Float16*) v78;
          _Float16 v80 = *(const _Float16 *)(v79);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v81 = __riscv_vfwmul_vf_f32m4(v70, v80, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v82 = __riscv_vfadd_vv_f32m4(v77, v81, 16);
          v23 = v82;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v83 = v12 * 4;
        size_t v84 = v83 + 0;
        size_t v85 = v84 * v2;
        size_t v86 = v16 * 16;
        size_t v87 = v85 + v86;
        size_t v88 = v87 + v21;
        float* v89 = v4 + v88;
        vfloat32m4_t v90 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v89, v90, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v91 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v92;
        v92 = v91;
        for (size_t v93 = 0; v93 < v9; v93 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v94 = v93 * 320;
          const uint8_t* v95 = v19 + v94;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v96 = v93 * 144;
          const uint8_t* v97 = v15 + v96;
          vint16m2_t v98;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v99 = __riscv_vmv_v_x_i16m2(0, 16);
          v98 = v99;
          vint16m2_t v100;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v101 = __riscv_vmv_v_x_i16m2(0, 16);
          v100 = v101;
          for (size_t v102 = 0; v102 < 16; v102 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v103 = v102 * 16;
            size_t v104 = 64 + v103;
            size_t v105 = v104 + v21;
            const uint8_t* v106 = v95 + v105;
            const uint8_t* v107 = (const uint8_t*) v106;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v108 = __riscv_vle8_v_u8m1(v107, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v109 = __riscv_vand_vx_u8m1(v108, 0x0F, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v110 = __riscv_vreinterpret_v_u8m1_i8m1(v109);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v111 = __riscv_vsrl_vx_u8m1(v108, 0x04, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v112 = __riscv_vreinterpret_v_u8m1_i8m1(v111);
            size_t v113 = v102 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v114 = v113 + 1;
            size_t v115 = 16 + v114;
            const uint8_t* v116 = v97 + v115;
            const int8_t* v117 = (const int8_t*) v116;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v118 = *(const int8_t *)(v117);
            vint16m2_t v119 = v98;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v120 = __riscv_vwmacc_vx_i16m2(v119, v118, v110, 16);
            v98 = v120;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v121 = v113 + 1;
            size_t v122 = 16 + 64;
            size_t v123 = v122 + v121;
            const uint8_t* v124 = v97 + v123;
            const int8_t* v125 = (const int8_t*) v124;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v126 = *(const int8_t *)(v125);
            vint16m2_t v127 = v100;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v128 = __riscv_vwmacc_vx_i16m2(v127, v126, v112, 16);
            v100 = v128;
          }
          vint16m2_t v129 = v98;
          vint16m2_t v130 = v100;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v131 = __riscv_vwadd_vv_i32m4(v129, v130, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v132 = v21 * 2;
          const uint8_t* v133 = v95 + v132;
          const _Float16* v134 = (const _Float16*) v133;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v135 = __riscv_vle16_v_f16m2(v134, 16);
          size_t v136 = 32 + v132;
          const uint8_t* v137 = v95 + v136;
          const _Float16* v138 = (const _Float16*) v137;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v139 = __riscv_vle16_v_f16m2(v138, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v140 = v97 + 2;
          const _Float16* v141 = (const _Float16*) v140;
          _Float16 v142 = *(const _Float16 *)(v141);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v143 = __riscv_vfwmul_vf_f32m4(v135, v142, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v144 = __riscv_vfcvt_f_x_v_f32m4(v131, 16);
          vfloat32m4_t v145 = v92;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v146 = __riscv_vfmacc_vv_f32m4(v145, v144, v143, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v147 = v97 + 10;
          const _Float16* v148 = (const _Float16*) v147;
          _Float16 v149 = *(const _Float16 *)(v148);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v150 = __riscv_vfwmul_vf_f32m4(v139, v149, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v151 = __riscv_vfadd_vv_f32m4(v146, v150, 16);
          v92 = v151;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v152 = v12 * 4;
        size_t v153 = v152 + 1;
        size_t v154 = v153 * v2;
        size_t v155 = v16 * 16;
        size_t v156 = v154 + v155;
        size_t v157 = v156 + v21;
        float* v158 = v4 + v157;
        vfloat32m4_t v159 = v92;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v158, v159, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v160 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v161;
        v161 = v160;
        for (size_t v162 = 0; v162 < v9; v162 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v163 = v162 * 320;
          const uint8_t* v164 = v19 + v163;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v165 = v162 * 144;
          const uint8_t* v166 = v15 + v165;
          vint16m2_t v167;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v168 = __riscv_vmv_v_x_i16m2(0, 16);
          v167 = v168;
          vint16m2_t v169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v170 = __riscv_vmv_v_x_i16m2(0, 16);
          v169 = v170;
          for (size_t v171 = 0; v171 < 16; v171 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v172 = v171 * 16;
            size_t v173 = 64 + v172;
            size_t v174 = v173 + v21;
            const uint8_t* v175 = v164 + v174;
            const uint8_t* v176 = (const uint8_t*) v175;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v177 = __riscv_vle8_v_u8m1(v176, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v178 = __riscv_vand_vx_u8m1(v177, 0x0F, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v179 = __riscv_vreinterpret_v_u8m1_i8m1(v178);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v180 = __riscv_vsrl_vx_u8m1(v177, 0x04, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v181 = __riscv_vreinterpret_v_u8m1_i8m1(v180);
            size_t v182 = v171 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v183 = v182 + 2;
            size_t v184 = 16 + v183;
            const uint8_t* v185 = v166 + v184;
            const int8_t* v186 = (const int8_t*) v185;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v187 = *(const int8_t *)(v186);
            vint16m2_t v188 = v167;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v189 = __riscv_vwmacc_vx_i16m2(v188, v187, v179, 16);
            v167 = v189;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v190 = v182 + 2;
            size_t v191 = 16 + 64;
            size_t v192 = v191 + v190;
            const uint8_t* v193 = v166 + v192;
            const int8_t* v194 = (const int8_t*) v193;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v195 = *(const int8_t *)(v194);
            vint16m2_t v196 = v169;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v197 = __riscv_vwmacc_vx_i16m2(v196, v195, v181, 16);
            v169 = v197;
          }
          vint16m2_t v198 = v167;
          vint16m2_t v199 = v169;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v200 = __riscv_vwadd_vv_i32m4(v198, v199, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v201 = v21 * 2;
          const uint8_t* v202 = v164 + v201;
          const _Float16* v203 = (const _Float16*) v202;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v204 = __riscv_vle16_v_f16m2(v203, 16);
          size_t v205 = 32 + v201;
          const uint8_t* v206 = v164 + v205;
          const _Float16* v207 = (const _Float16*) v206;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v208 = __riscv_vle16_v_f16m2(v207, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v209 = v166 + 4;
          const _Float16* v210 = (const _Float16*) v209;
          _Float16 v211 = *(const _Float16 *)(v210);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v212 = __riscv_vfwmul_vf_f32m4(v204, v211, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v213 = __riscv_vfcvt_f_x_v_f32m4(v200, 16);
          vfloat32m4_t v214 = v161;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v215 = __riscv_vfmacc_vv_f32m4(v214, v213, v212, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v216 = v166 + 12;
          const _Float16* v217 = (const _Float16*) v216;
          _Float16 v218 = *(const _Float16 *)(v217);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v219 = __riscv_vfwmul_vf_f32m4(v208, v218, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v220 = __riscv_vfadd_vv_f32m4(v215, v219, 16);
          v161 = v220;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v221 = v12 * 4;
        size_t v222 = v221 + 2;
        size_t v223 = v222 * v2;
        size_t v224 = v16 * 16;
        size_t v225 = v223 + v224;
        size_t v226 = v225 + v21;
        float* v227 = v4 + v226;
        vfloat32m4_t v228 = v161;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v227, v228, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v229 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v230;
        v230 = v229;
        for (size_t v231 = 0; v231 < v9; v231 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v232 = v231 * 320;
          const uint8_t* v233 = v19 + v232;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v234 = v231 * 144;
          const uint8_t* v235 = v15 + v234;
          vint16m2_t v236;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v237 = __riscv_vmv_v_x_i16m2(0, 16);
          v236 = v237;
          vint16m2_t v238;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v239 = __riscv_vmv_v_x_i16m2(0, 16);
          v238 = v239;
          for (size_t v240 = 0; v240 < 16; v240 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v241 = v240 * 16;
            size_t v242 = 64 + v241;
            size_t v243 = v242 + v21;
            const uint8_t* v244 = v233 + v243;
            const uint8_t* v245 = (const uint8_t*) v244;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8m1
            vuint8m1_t v246 = __riscv_vle8_v_u8m1(v245, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8m1
            vuint8m1_t v247 = __riscv_vand_vx_u8m1(v246, 0x0F, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v248 = __riscv_vreinterpret_v_u8m1_i8m1(v247);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8m1
            vuint8m1_t v249 = __riscv_vsrl_vx_u8m1(v246, 0x04, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8m1_i8m1
            vint8m1_t v250 = __riscv_vreinterpret_v_u8m1_i8m1(v249);
            size_t v251 = v240 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v252 = v251 + 3;
            size_t v253 = 16 + v252;
            const uint8_t* v254 = v235 + v253;
            const int8_t* v255 = (const int8_t*) v254;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v256 = *(const int8_t *)(v255);
            vint16m2_t v257 = v236;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v258 = __riscv_vwmacc_vx_i16m2(v257, v256, v248, 16);
            v236 = v258;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v259 = v251 + 3;
            size_t v260 = 16 + 64;
            size_t v261 = v260 + v259;
            const uint8_t* v262 = v235 + v261;
            const int8_t* v263 = (const int8_t*) v262;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v264 = *(const int8_t *)(v263);
            vint16m2_t v265 = v238;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v266 = __riscv_vwmacc_vx_i16m2(v265, v264, v250, 16);
            v238 = v266;
          }
          vint16m2_t v267 = v236;
          vint16m2_t v268 = v238;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v269 = __riscv_vwadd_vv_i32m4(v267, v268, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v270 = v21 * 2;
          const uint8_t* v271 = v233 + v270;
          const _Float16* v272 = (const _Float16*) v271;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v273 = __riscv_vle16_v_f16m2(v272, 16);
          size_t v274 = 32 + v270;
          const uint8_t* v275 = v233 + v274;
          const _Float16* v276 = (const _Float16*) v275;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v277 = __riscv_vle16_v_f16m2(v276, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v278 = v235 + 6;
          const _Float16* v279 = (const _Float16*) v278;
          _Float16 v280 = *(const _Float16 *)(v279);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v281 = __riscv_vfwmul_vf_f32m4(v273, v280, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v282 = __riscv_vfcvt_f_x_v_f32m4(v269, 16);
          vfloat32m4_t v283 = v230;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v284 = __riscv_vfmacc_vv_f32m4(v283, v282, v281, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v285 = v235 + 14;
          const _Float16* v286 = (const _Float16*) v285;
          _Float16 v287 = *(const _Float16 *)(v286);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v288 = __riscv_vfwmul_vf_f32m4(v277, v287, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m4
          vfloat32m4_t v289 = __riscv_vfadd_vv_f32m4(v284, v288, 16);
          v230 = v289;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v290 = v12 * 4;
        size_t v291 = v290 + 3;
        size_t v292 = v291 * v2;
        size_t v293 = v16 * 16;
        size_t v294 = v292 + v293;
        size_t v295 = v294 + v21;
        float* v296 = v4 + v295;
        vfloat32m4_t v297 = v230;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v296, v297, 16);
      }
    }
  }
  return;
}


