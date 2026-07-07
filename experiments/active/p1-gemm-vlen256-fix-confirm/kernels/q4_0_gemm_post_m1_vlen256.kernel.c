#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v3);
  // tcrv_emitc.route_source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v9 = v3 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v1 / 4;
  // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v5 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v7 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 288;
      const uint8_t* v19 = v6 + v18;
      for (size_t v20 = 0; v20 < 1; v20 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 16;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v22 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v23;
        v23 = v22;
        for (size_t v24 = 0; v24 < v9; v24 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v25 = v24 * 288;
          const uint8_t* v26 = v19 + v25;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v27 = v24 * 136;
          const uint8_t* v28 = v15 + v27;
          vint16m2_t v29;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v30 = __riscv_vmv_v_x_i16m2(0, 16);
          v29 = v30;
          vint16m2_t v31;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v32 = __riscv_vmv_v_x_i16m2(0, 16);
          v31 = v32;
          for (size_t v33 = 0; v33 < 16; v33 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v34 = v33 * 16;
            size_t v35 = 32 + v34;
            size_t v36 = v35 + v21;
            const uint8_t* v37 = v26 + v36;
            const int8_t* v38 = (const int8_t*) v37;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v39 = __riscv_vle8_v_i8m1(v38, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
            vint8m1_t v40 = __riscv_vsll_vx_i8m1(v39, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v41 = __riscv_vsra_vx_i8m1(v40, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v42 = __riscv_vsra_vx_i8m1(v39, 4, 16);
            size_t v43 = v33 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v44 = v43 + 0;
            size_t v45 = 8 + v44;
            const uint8_t* v46 = v28 + v45;
            const int8_t* v47 = (const int8_t*) v46;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v48 = *(const int8_t *)(v47);
            vint16m2_t v49 = v29;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v50 = __riscv_vwmacc_vx_i16m2(v49, v48, v41, 16);
            v29 = v50;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v51 = v43 + 0;
            size_t v52 = 8 + 64;
            size_t v53 = v52 + v51;
            const uint8_t* v54 = v28 + v53;
            const int8_t* v55 = (const int8_t*) v54;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v56 = *(const int8_t *)(v55);
            vint16m2_t v57 = v31;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v58 = __riscv_vwmacc_vx_i16m2(v57, v56, v42, 16);
            v31 = v58;
          }
          vint16m2_t v59 = v29;
          vint16m2_t v60 = v31;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v61 = __riscv_vwadd_vv_i32m4(v59, v60, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v62 = v21 * 2;
          const uint8_t* v63 = v26 + v62;
          const _Float16* v64 = (const _Float16*) v63;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v65 = __riscv_vle16_v_f16m2(v64, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v66 = v28 + 0;
          const _Float16* v67 = (const _Float16*) v66;
          _Float16 v68 = *(const _Float16 *)(v67);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v69 = __riscv_vfwmul_vf_f32m4(v65, v68, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v70 = __riscv_vfcvt_f_x_v_f32m4(v61, 16);
          vfloat32m4_t v71 = v23;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v72 = __riscv_vfmacc_vv_f32m4(v71, v70, v69, 16);
          v23 = v72;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v73 = v12 * 4;
        size_t v74 = v73 + 0;
        size_t v75 = v74 * v2;
        size_t v76 = v16 * 16;
        size_t v77 = v75 + v76;
        size_t v78 = v77 + v21;
        float* v79 = v4 + v78;
        vfloat32m4_t v80 = v23;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v79, v80, 16);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v81 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v82;
        v82 = v81;
        for (size_t v83 = 0; v83 < v9; v83 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v84 = v83 * 288;
          const uint8_t* v85 = v19 + v84;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v86 = v83 * 136;
          const uint8_t* v87 = v15 + v86;
          vint16m2_t v88;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v89 = __riscv_vmv_v_x_i16m2(0, 16);
          v88 = v89;
          vint16m2_t v90;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v91 = __riscv_vmv_v_x_i16m2(0, 16);
          v90 = v91;
          for (size_t v92 = 0; v92 < 16; v92 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v93 = v92 * 16;
            size_t v94 = 32 + v93;
            size_t v95 = v94 + v21;
            const uint8_t* v96 = v85 + v95;
            const int8_t* v97 = (const int8_t*) v96;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v98 = __riscv_vle8_v_i8m1(v97, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
            vint8m1_t v99 = __riscv_vsll_vx_i8m1(v98, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v100 = __riscv_vsra_vx_i8m1(v99, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v101 = __riscv_vsra_vx_i8m1(v98, 4, 16);
            size_t v102 = v92 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v103 = v102 + 1;
            size_t v104 = 8 + v103;
            const uint8_t* v105 = v87 + v104;
            const int8_t* v106 = (const int8_t*) v105;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v107 = *(const int8_t *)(v106);
            vint16m2_t v108 = v88;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v109 = __riscv_vwmacc_vx_i16m2(v108, v107, v100, 16);
            v88 = v109;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v110 = v102 + 1;
            size_t v111 = 8 + 64;
            size_t v112 = v111 + v110;
            const uint8_t* v113 = v87 + v112;
            const int8_t* v114 = (const int8_t*) v113;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v115 = *(const int8_t *)(v114);
            vint16m2_t v116 = v90;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v117 = __riscv_vwmacc_vx_i16m2(v116, v115, v101, 16);
            v90 = v117;
          }
          vint16m2_t v118 = v88;
          vint16m2_t v119 = v90;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v120 = __riscv_vwadd_vv_i32m4(v118, v119, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v121 = v21 * 2;
          const uint8_t* v122 = v85 + v121;
          const _Float16* v123 = (const _Float16*) v122;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v124 = __riscv_vle16_v_f16m2(v123, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v125 = v87 + 2;
          const _Float16* v126 = (const _Float16*) v125;
          _Float16 v127 = *(const _Float16 *)(v126);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v128 = __riscv_vfwmul_vf_f32m4(v124, v127, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v129 = __riscv_vfcvt_f_x_v_f32m4(v120, 16);
          vfloat32m4_t v130 = v82;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v131 = __riscv_vfmacc_vv_f32m4(v130, v129, v128, 16);
          v82 = v131;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v132 = v12 * 4;
        size_t v133 = v132 + 1;
        size_t v134 = v133 * v2;
        size_t v135 = v16 * 16;
        size_t v136 = v134 + v135;
        size_t v137 = v136 + v21;
        float* v138 = v4 + v137;
        vfloat32m4_t v139 = v82;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v138, v139, 16);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v140 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v141;
        v141 = v140;
        for (size_t v142 = 0; v142 < v9; v142 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v143 = v142 * 288;
          const uint8_t* v144 = v19 + v143;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v145 = v142 * 136;
          const uint8_t* v146 = v15 + v145;
          vint16m2_t v147;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v148 = __riscv_vmv_v_x_i16m2(0, 16);
          v147 = v148;
          vint16m2_t v149;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v150 = __riscv_vmv_v_x_i16m2(0, 16);
          v149 = v150;
          for (size_t v151 = 0; v151 < 16; v151 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v152 = v151 * 16;
            size_t v153 = 32 + v152;
            size_t v154 = v153 + v21;
            const uint8_t* v155 = v144 + v154;
            const int8_t* v156 = (const int8_t*) v155;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v157 = __riscv_vle8_v_i8m1(v156, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
            vint8m1_t v158 = __riscv_vsll_vx_i8m1(v157, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v159 = __riscv_vsra_vx_i8m1(v158, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v160 = __riscv_vsra_vx_i8m1(v157, 4, 16);
            size_t v161 = v151 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v162 = v161 + 2;
            size_t v163 = 8 + v162;
            const uint8_t* v164 = v146 + v163;
            const int8_t* v165 = (const int8_t*) v164;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v166 = *(const int8_t *)(v165);
            vint16m2_t v167 = v147;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v168 = __riscv_vwmacc_vx_i16m2(v167, v166, v159, 16);
            v147 = v168;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v169 = v161 + 2;
            size_t v170 = 8 + 64;
            size_t v171 = v170 + v169;
            const uint8_t* v172 = v146 + v171;
            const int8_t* v173 = (const int8_t*) v172;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v174 = *(const int8_t *)(v173);
            vint16m2_t v175 = v149;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v176 = __riscv_vwmacc_vx_i16m2(v175, v174, v160, 16);
            v149 = v176;
          }
          vint16m2_t v177 = v147;
          vint16m2_t v178 = v149;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v179 = __riscv_vwadd_vv_i32m4(v177, v178, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v180 = v21 * 2;
          const uint8_t* v181 = v144 + v180;
          const _Float16* v182 = (const _Float16*) v181;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v183 = __riscv_vle16_v_f16m2(v182, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v184 = v146 + 4;
          const _Float16* v185 = (const _Float16*) v184;
          _Float16 v186 = *(const _Float16 *)(v185);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v187 = __riscv_vfwmul_vf_f32m4(v183, v186, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v188 = __riscv_vfcvt_f_x_v_f32m4(v179, 16);
          vfloat32m4_t v189 = v141;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v190 = __riscv_vfmacc_vv_f32m4(v189, v188, v187, 16);
          v141 = v190;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v191 = v12 * 4;
        size_t v192 = v191 + 2;
        size_t v193 = v192 * v2;
        size_t v194 = v16 * 16;
        size_t v195 = v193 + v194;
        size_t v196 = v195 + v21;
        float* v197 = v4 + v196;
        vfloat32m4_t v198 = v141;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v197, v198, 16);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m4
        vfloat32m4_t v199 = __riscv_vfmv_v_f_f32m4(0.0f, 16);
        vfloat32m4_t v200;
        v200 = v199;
        for (size_t v201 = 0; v201 < v9; v201 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v202 = v201 * 288;
          const uint8_t* v203 = v19 + v202;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v204 = v201 * 136;
          const uint8_t* v205 = v15 + v204;
          vint16m2_t v206;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v207 = __riscv_vmv_v_x_i16m2(0, 16);
          v206 = v207;
          vint16m2_t v208;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m2
          vint16m2_t v209 = __riscv_vmv_v_x_i16m2(0, 16);
          v208 = v209;
          for (size_t v210 = 0; v210 < 16; v210 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v211 = v210 * 16;
            size_t v212 = 32 + v211;
            size_t v213 = v212 + v21;
            const uint8_t* v214 = v203 + v213;
            const int8_t* v215 = (const int8_t*) v214;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8m1
            vint8m1_t v216 = __riscv_vle8_v_i8m1(v215, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8m1
            vint8m1_t v217 = __riscv_vsll_vx_i8m1(v216, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v218 = __riscv_vsra_vx_i8m1(v217, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8m1
            vint8m1_t v219 = __riscv_vsra_vx_i8m1(v216, 4, 16);
            size_t v220 = v210 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v221 = v220 + 3;
            size_t v222 = 8 + v221;
            const uint8_t* v223 = v205 + v222;
            const int8_t* v224 = (const int8_t*) v223;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v225 = *(const int8_t *)(v224);
            vint16m2_t v226 = v206;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v227 = __riscv_vwmacc_vx_i16m2(v226, v225, v218, 16);
            v206 = v227;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v228 = v220 + 3;
            size_t v229 = 8 + 64;
            size_t v230 = v229 + v228;
            const uint8_t* v231 = v205 + v230;
            const int8_t* v232 = (const int8_t*) v231;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v233 = *(const int8_t *)(v232);
            vint16m2_t v234 = v208;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m2
            vint16m2_t v235 = __riscv_vwmacc_vx_i16m2(v234, v233, v219, 16);
            v208 = v235;
          }
          vint16m2_t v236 = v206;
          vint16m2_t v237 = v208;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m4
          vint32m4_t v238 = __riscv_vwadd_vv_i32m4(v236, v237, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v239 = v21 * 2;
          const uint8_t* v240 = v203 + v239;
          const _Float16* v241 = (const _Float16*) v240;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m2
          vfloat16m2_t v242 = __riscv_vle16_v_f16m2(v241, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v243 = v205 + 6;
          const _Float16* v244 = (const _Float16*) v243;
          _Float16 v245 = *(const _Float16 *)(v244);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m4
          vfloat32m4_t v246 = __riscv_vfwmul_vf_f32m4(v242, v245, 16);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m4
          vfloat32m4_t v247 = __riscv_vfcvt_f_x_v_f32m4(v238, 16);
          vfloat32m4_t v248 = v200;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m4
          vfloat32m4_t v249 = __riscv_vfmacc_vv_f32m4(v248, v247, v246, 16);
          v200 = v249;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v250 = v12 * 4;
        size_t v251 = v250 + 3;
        size_t v252 = v251 * v2;
        size_t v253 = v16 * 16;
        size_t v254 = v252 + v253;
        size_t v255 = v254 + v21;
        float* v256 = v4 + v255;
        vfloat32m4_t v257 = v200;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m4
        __riscv_vse32_v_f32m4(v256, v257, 16);
      }
    }
  }
  return;
}


