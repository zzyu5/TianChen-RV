#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q8_gemm_mf2(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
      size_t v18 = v17 * 544;
      const uint8_t* v19 = v6 + v18;
      for (size_t v20 = 0; v20 < 2; v20 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 8;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v22 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v24 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v25 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        vfloat32m2_t v26;
        v26 = v22;
        vfloat32m2_t v27;
        v27 = v23;
        vfloat32m2_t v28;
        v28 = v24;
        vfloat32m2_t v29;
        v29 = v25;
        for (size_t v30 = 0; v30 < v9; v30 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v31 = v30 * 544;
          const uint8_t* v32 = v19 + v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 136;
          const uint8_t* v34 = v15 + v33;
          vint32m2_t v35;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v36 = __riscv_vmv_v_x_i32m2(0, 8);
          v35 = v36;
          vint32m2_t v37;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v38 = __riscv_vmv_v_x_i32m2(0, 8);
          v37 = v38;
          vint32m2_t v39;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v40 = __riscv_vmv_v_x_i32m2(0, 8);
          v39 = v40;
          vint32m2_t v41;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m2
          vint32m2_t v42 = __riscv_vmv_v_x_i32m2(0, 8);
          v41 = v42;
          for (size_t v43 = 0; v43 < 32; v43 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v44 = v43 * 16;
            size_t v45 = 32 + v44;
            size_t v46 = v45 + v21;
            const uint8_t* v47 = v32 + v46;
            const int8_t* v48 = (const int8_t*) v47;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
            vint8mf2_t v49 = __riscv_vle8_v_i8mf2(v48, 8);
            size_t v50 = v43 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v51 = v50 + 0;
            size_t v52 = 8 + v51;
            const uint8_t* v53 = v34 + v52;
            const int8_t* v54 = (const int8_t*) v53;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v55 = *(const int8_t *)(v54);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v56 = __riscv_vwmul_vx_i16m1(v49, v55, 8);
            vint32m2_t v57 = v35;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v58 = __riscv_vwadd_wv_i32m2(v57, v56, 8);
            v35 = v58;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v59 = v50 + 1;
            size_t v60 = 8 + v59;
            const uint8_t* v61 = v34 + v60;
            const int8_t* v62 = (const int8_t*) v61;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v63 = *(const int8_t *)(v62);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v64 = __riscv_vwmul_vx_i16m1(v49, v63, 8);
            vint32m2_t v65 = v37;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v66 = __riscv_vwadd_wv_i32m2(v65, v64, 8);
            v37 = v66;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v67 = v50 + 2;
            size_t v68 = 8 + v67;
            const uint8_t* v69 = v34 + v68;
            const int8_t* v70 = (const int8_t*) v69;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v71 = *(const int8_t *)(v70);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v72 = __riscv_vwmul_vx_i16m1(v49, v71, 8);
            vint32m2_t v73 = v39;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v74 = __riscv_vwadd_wv_i32m2(v73, v72, 8);
            v39 = v74;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v75 = v50 + 3;
            size_t v76 = 8 + v75;
            const uint8_t* v77 = v34 + v76;
            const int8_t* v78 = (const int8_t*) v77;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v79 = *(const int8_t *)(v78);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m1
            vint16m1_t v80 = __riscv_vwmul_vx_i16m1(v49, v79, 8);
            vint32m2_t v81 = v41;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m2
            vint32m2_t v82 = __riscv_vwadd_wv_i32m2(v81, v80, 8);
            v41 = v82;
          }
          vint32m2_t v83 = v35;
          vint32m2_t v84 = v37;
          vint32m2_t v85 = v39;
          vint32m2_t v86 = v41;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v87 = v21 * 2;
          const uint8_t* v88 = v32 + v87;
          const _Float16* v89 = (const _Float16*) v88;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v90 = __riscv_vle16_v_f16m1(v89, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v91 = v34 + 0;
          const _Float16* v92 = (const _Float16*) v91;
          _Float16 v93 = *(const _Float16 *)(v92);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v94 = __riscv_vfwmul_vf_f32m2(v90, v93, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v95 = __riscv_vfcvt_f_x_v_f32m2(v83, 8);
          vfloat32m2_t v96 = v26;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v97 = __riscv_vfmacc_vv_f32m2(v96, v95, v94, 8);
          v26 = v97;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v98 = v34 + 2;
          const _Float16* v99 = (const _Float16*) v98;
          _Float16 v100 = *(const _Float16 *)(v99);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v101 = __riscv_vfwmul_vf_f32m2(v90, v100, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v102 = __riscv_vfcvt_f_x_v_f32m2(v84, 8);
          vfloat32m2_t v103 = v27;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v104 = __riscv_vfmacc_vv_f32m2(v103, v102, v101, 8);
          v27 = v104;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v105 = v34 + 4;
          const _Float16* v106 = (const _Float16*) v105;
          _Float16 v107 = *(const _Float16 *)(v106);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v108 = __riscv_vfwmul_vf_f32m2(v90, v107, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v109 = __riscv_vfcvt_f_x_v_f32m2(v85, 8);
          vfloat32m2_t v110 = v28;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v111 = __riscv_vfmacc_vv_f32m2(v110, v109, v108, 8);
          v28 = v111;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v112 = v34 + 6;
          const _Float16* v113 = (const _Float16*) v112;
          _Float16 v114 = *(const _Float16 *)(v113);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v115 = __riscv_vfwmul_vf_f32m2(v90, v114, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v116 = __riscv_vfcvt_f_x_v_f32m2(v86, 8);
          vfloat32m2_t v117 = v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v118 = __riscv_vfmacc_vv_f32m2(v117, v116, v115, 8);
          v29 = v118;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v119 = v12 * 4;
        size_t v120 = v119 + 0;
        size_t v121 = v120 * v2;
        size_t v122 = v16 * 16;
        size_t v123 = v121 + v122;
        size_t v124 = v123 + v21;
        float* v125 = v4 + v124;
        vfloat32m2_t v126 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v125, v126, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v127 = v12 * 4;
        size_t v128 = v127 + 1;
        size_t v129 = v128 * v2;
        size_t v130 = v16 * 16;
        size_t v131 = v129 + v130;
        size_t v132 = v131 + v21;
        float* v133 = v4 + v132;
        vfloat32m2_t v134 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v133, v134, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v135 = v12 * 4;
        size_t v136 = v135 + 2;
        size_t v137 = v136 * v2;
        size_t v138 = v16 * 16;
        size_t v139 = v137 + v138;
        size_t v140 = v139 + v21;
        float* v141 = v4 + v140;
        vfloat32m2_t v142 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v141, v142, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v143 = v12 * 4;
        size_t v144 = v143 + 3;
        size_t v145 = v144 * v2;
        size_t v146 = v16 * 16;
        size_t v147 = v145 + v146;
        size_t v148 = v147 + v21;
        float* v149 = v4 + v148;
        vfloat32m2_t v150 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v149, v150, 8);
      }
    }
  }
  return;
}


