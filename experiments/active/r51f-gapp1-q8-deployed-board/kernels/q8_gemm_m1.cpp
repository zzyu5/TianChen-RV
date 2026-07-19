#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q8_gemm_m1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
      for (size_t v20 = 0; v20 < 1; v20 += 1) {
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 16;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v22 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v23;
        v23 = v22;
        for (size_t v24 = 0; v24 < v9; v24 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v25 = v24 * 544;
          const uint8_t* v26 = v19 + v25;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v27 = v24 * 136;
          const uint8_t* v28 = v15 + v27;
          vint32m4_t v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
          vint32m4_t v30 = __riscv_vmv_v_x_i32m4(0, 16);
          v29 = v30;
          for (size_t v31 = 0; v31 < 32; v31 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v32 = v31 * 16;
            size_t v33 = 32 + v32;
            size_t v34 = v33 + v21;
            const uint8_t* v35 = v26 + v34;
            const int8_t* v36 = (const int8_t*) v35;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v37 = __riscv_vle8_v_i8m1(v36, 16);
            size_t v38 = v31 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v39 = v38 + 0;
            size_t v40 = 8 + v39;
            const uint8_t* v41 = v28 + v40;
            const int8_t* v42 = (const int8_t*) v41;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v43 = *(const int8_t *)(v42);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m2
            vint16m2_t v44 = __riscv_vwmul_vx_i16m2(v37, v43, 16);
            vint32m4_t v45 = v29;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m4
            vint32m4_t v46 = __riscv_vwadd_wv_i32m4(v45, v44, 16);
            v29 = v46;
          }
          vint32m4_t v47 = v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v48 = v21 * 2;
          const uint8_t* v49 = v26 + v48;
          const _Float16* v50 = (const _Float16*) v49;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v51 = __riscv_vle16_v_f16m2(v50, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v52 = v28 + 0;
          const _Float16* v53 = (const _Float16*) v52;
          _Float16 v54 = *(const _Float16 *)(v53);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v55 = __riscv_vfwmul_vf_f32m4(v51, v54, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v56 = __riscv_vfcvt_f_x_v_f32m4(v47, 16);
          vfloat32m4_t v57 = v23;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v58 = __riscv_vfmacc_vv_f32m4(v57, v56, v55, 16);
          v23 = v58;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v59 = v12 * 4;
        size_t v60 = v59 + 0;
        size_t v61 = v60 * v2;
        size_t v62 = v16 * 16;
        size_t v63 = v61 + v62;
        size_t v64 = v63 + v21;
        float* v65 = v4 + v64;
        vfloat32m4_t v66 = v23;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v65, v66, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v67 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v68;
        v68 = v67;
        for (size_t v69 = 0; v69 < v9; v69 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v70 = v69 * 544;
          const uint8_t* v71 = v19 + v70;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v72 = v69 * 136;
          const uint8_t* v73 = v15 + v72;
          vint32m4_t v74;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
          vint32m4_t v75 = __riscv_vmv_v_x_i32m4(0, 16);
          v74 = v75;
          for (size_t v76 = 0; v76 < 32; v76 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v77 = v76 * 16;
            size_t v78 = 32 + v77;
            size_t v79 = v78 + v21;
            const uint8_t* v80 = v71 + v79;
            const int8_t* v81 = (const int8_t*) v80;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v82 = __riscv_vle8_v_i8m1(v81, 16);
            size_t v83 = v76 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v84 = v83 + 1;
            size_t v85 = 8 + v84;
            const uint8_t* v86 = v73 + v85;
            const int8_t* v87 = (const int8_t*) v86;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v88 = *(const int8_t *)(v87);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m2
            vint16m2_t v89 = __riscv_vwmul_vx_i16m2(v82, v88, 16);
            vint32m4_t v90 = v74;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m4
            vint32m4_t v91 = __riscv_vwadd_wv_i32m4(v90, v89, 16);
            v74 = v91;
          }
          vint32m4_t v92 = v74;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v93 = v21 * 2;
          const uint8_t* v94 = v71 + v93;
          const _Float16* v95 = (const _Float16*) v94;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v96 = __riscv_vle16_v_f16m2(v95, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v97 = v73 + 2;
          const _Float16* v98 = (const _Float16*) v97;
          _Float16 v99 = *(const _Float16 *)(v98);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v100 = __riscv_vfwmul_vf_f32m4(v96, v99, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v101 = __riscv_vfcvt_f_x_v_f32m4(v92, 16);
          vfloat32m4_t v102 = v68;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v103 = __riscv_vfmacc_vv_f32m4(v102, v101, v100, 16);
          v68 = v103;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v104 = v12 * 4;
        size_t v105 = v104 + 1;
        size_t v106 = v105 * v2;
        size_t v107 = v16 * 16;
        size_t v108 = v106 + v107;
        size_t v109 = v108 + v21;
        float* v110 = v4 + v109;
        vfloat32m4_t v111 = v68;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v110, v111, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v112 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v113;
        v113 = v112;
        for (size_t v114 = 0; v114 < v9; v114 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v115 = v114 * 544;
          const uint8_t* v116 = v19 + v115;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v117 = v114 * 136;
          const uint8_t* v118 = v15 + v117;
          vint32m4_t v119;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
          vint32m4_t v120 = __riscv_vmv_v_x_i32m4(0, 16);
          v119 = v120;
          for (size_t v121 = 0; v121 < 32; v121 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v122 = v121 * 16;
            size_t v123 = 32 + v122;
            size_t v124 = v123 + v21;
            const uint8_t* v125 = v116 + v124;
            const int8_t* v126 = (const int8_t*) v125;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v127 = __riscv_vle8_v_i8m1(v126, 16);
            size_t v128 = v121 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v129 = v128 + 2;
            size_t v130 = 8 + v129;
            const uint8_t* v131 = v118 + v130;
            const int8_t* v132 = (const int8_t*) v131;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v133 = *(const int8_t *)(v132);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m2
            vint16m2_t v134 = __riscv_vwmul_vx_i16m2(v127, v133, 16);
            vint32m4_t v135 = v119;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m4
            vint32m4_t v136 = __riscv_vwadd_wv_i32m4(v135, v134, 16);
            v119 = v136;
          }
          vint32m4_t v137 = v119;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v138 = v21 * 2;
          const uint8_t* v139 = v116 + v138;
          const _Float16* v140 = (const _Float16*) v139;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v141 = __riscv_vle16_v_f16m2(v140, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v142 = v118 + 4;
          const _Float16* v143 = (const _Float16*) v142;
          _Float16 v144 = *(const _Float16 *)(v143);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v145 = __riscv_vfwmul_vf_f32m4(v141, v144, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v146 = __riscv_vfcvt_f_x_v_f32m4(v137, 16);
          vfloat32m4_t v147 = v113;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v148 = __riscv_vfmacc_vv_f32m4(v147, v146, v145, 16);
          v113 = v148;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v149 = v12 * 4;
        size_t v150 = v149 + 2;
        size_t v151 = v150 * v2;
        size_t v152 = v16 * 16;
        size_t v153 = v151 + v152;
        size_t v154 = v153 + v21;
        float* v155 = v4 + v154;
        vfloat32m4_t v156 = v113;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v155, v156, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v157 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v158;
        v158 = v157;
        for (size_t v159 = 0; v159 < v9; v159 += 1) {
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v160 = v159 * 544;
          const uint8_t* v161 = v19 + v160;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v162 = v159 * 136;
          const uint8_t* v163 = v15 + v162;
          vint32m4_t v164;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m4
          vint32m4_t v165 = __riscv_vmv_v_x_i32m4(0, 16);
          v164 = v165;
          for (size_t v166 = 0; v166 < 32; v166 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_quant_addr
            size_t v167 = v166 * 16;
            size_t v168 = 32 + v167;
            size_t v169 = v168 + v21;
            const uint8_t* v170 = v161 + v169;
            const int8_t* v171 = (const int8_t*) v170;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v172 = __riscv_vle8_v_i8m1(v171, 16);
            size_t v173 = v166 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr
            size_t v174 = v173 + 3;
            size_t v175 = 8 + v174;
            const uint8_t* v176 = v163 + v175;
            const int8_t* v177 = (const int8_t*) v176;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v178 = *(const int8_t *)(v177);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmul_vx_i16m2
            vint16m2_t v179 = __riscv_vwmul_vx_i16m2(v172, v178, 16);
            vint32m4_t v180 = v164;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_wv_i32m4
            vint32m4_t v181 = __riscv_vwadd_wv_i32m4(v180, v179, 16);
            v164 = v181;
          }
          vint32m4_t v182 = v164;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v183 = v21 * 2;
          const uint8_t* v184 = v161 + v183;
          const _Float16* v185 = (const _Float16*) v184;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v186 = __riscv_vle16_v_f16m2(v185, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v187 = v163 + 6;
          const _Float16* v188 = (const _Float16*) v187;
          _Float16 v189 = *(const _Float16 *)(v188);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v190 = __riscv_vfwmul_vf_f32m4(v186, v189, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v191 = __riscv_vfcvt_f_x_v_f32m4(v182, 16);
          vfloat32m4_t v192 = v158;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v193 = __riscv_vfmacc_vv_f32m4(v192, v191, v190, 16);
          v158 = v193;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v194 = v12 * 4;
        size_t v195 = v194 + 3;
        size_t v196 = v195 * v2;
        size_t v197 = v16 * 16;
        size_t v198 = v196 + v197;
        size_t v199 = v198 + v21;
        float* v200 = v4 + v199;
        vfloat32m4_t v201 = v158;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v200, v201, 16);
      }
    }
  }
  return;
}


