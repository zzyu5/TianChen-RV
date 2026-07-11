#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
    size_t v14 = v13 * 144;
    const uint8_t* v15 = v7 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 320;
      const uint8_t* v19 = v6 + v18;
      for (size_t v20 = 0; v20 < 2; v20 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 8;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v22 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v24 = __riscv_vfmv_v_f_f32m2(0.0f, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
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
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v31 = v30 * 320;
          const uint8_t* v32 = v19 + v31;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 144;
          const uint8_t* v34 = v15 + v33;
          vint16m1_t v35;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v36 = __riscv_vmv_v_x_i16m1(0, 8);
          v35 = v36;
          vint16m1_t v37;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v38 = __riscv_vmv_v_x_i16m1(0, 8);
          v37 = v38;
          vint16m1_t v39;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v40 = __riscv_vmv_v_x_i16m1(0, 8);
          v39 = v40;
          vint16m1_t v41;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v42 = __riscv_vmv_v_x_i16m1(0, 8);
          v41 = v42;
          vint16m1_t v43;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v44 = __riscv_vmv_v_x_i16m1(0, 8);
          v43 = v44;
          vint16m1_t v45;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v46 = __riscv_vmv_v_x_i16m1(0, 8);
          v45 = v46;
          vint16m1_t v47;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v48 = __riscv_vmv_v_x_i16m1(0, 8);
          v47 = v48;
          vint16m1_t v49;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v50 = __riscv_vmv_v_x_i16m1(0, 8);
          v49 = v50;
          for (size_t v51 = 0; v51 < 16; v51 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v52 = v51 * 16;
            size_t v53 = 64 + v52;
            size_t v54 = v53 + v21;
            const uint8_t* v55 = v32 + v54;
            const uint8_t* v56 = (const uint8_t*) v55;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
            vuint8mf2_t v57 = __riscv_vle8_v_u8mf2(v56, 8);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
            vuint8mf2_t v58 = __riscv_vand_vx_u8mf2(v57, 0x0F, 8);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v59 = __riscv_vreinterpret_v_u8mf2_i8mf2(v58);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
            vuint8mf2_t v60 = __riscv_vsrl_vx_u8mf2(v57, 0x04, 8);
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v61 = __riscv_vreinterpret_v_u8mf2_i8mf2(v60);
            size_t v62 = v51 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v63 = v62 + 0;
            size_t v64 = 16 + v63;
            const uint8_t* v65 = v34 + v64;
            const int8_t* v66 = (const int8_t*) v65;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v67 = *(const int8_t *)(v66);
            vint16m1_t v68 = v35;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v69 = __riscv_vwmacc_vx_i16m1(v68, v67, v59, 8);
            v35 = v69;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v70 = v62 + 0;
            size_t v71 = 16 + 64;
            size_t v72 = v71 + v70;
            const uint8_t* v73 = v34 + v72;
            const int8_t* v74 = (const int8_t*) v73;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v75 = *(const int8_t *)(v74);
            vint16m1_t v76 = v43;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v77 = __riscv_vwmacc_vx_i16m1(v76, v75, v61, 8);
            v43 = v77;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v78 = v62 + 1;
            size_t v79 = 16 + v78;
            const uint8_t* v80 = v34 + v79;
            const int8_t* v81 = (const int8_t*) v80;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v82 = *(const int8_t *)(v81);
            vint16m1_t v83 = v37;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v84 = __riscv_vwmacc_vx_i16m1(v83, v82, v59, 8);
            v37 = v84;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v85 = v62 + 1;
            size_t v86 = 16 + 64;
            size_t v87 = v86 + v85;
            const uint8_t* v88 = v34 + v87;
            const int8_t* v89 = (const int8_t*) v88;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v90 = *(const int8_t *)(v89);
            vint16m1_t v91 = v45;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v92 = __riscv_vwmacc_vx_i16m1(v91, v90, v61, 8);
            v45 = v92;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v93 = v62 + 2;
            size_t v94 = 16 + v93;
            const uint8_t* v95 = v34 + v94;
            const int8_t* v96 = (const int8_t*) v95;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v97 = *(const int8_t *)(v96);
            vint16m1_t v98 = v39;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v99 = __riscv_vwmacc_vx_i16m1(v98, v97, v59, 8);
            v39 = v99;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v100 = v62 + 2;
            size_t v101 = 16 + 64;
            size_t v102 = v101 + v100;
            const uint8_t* v103 = v34 + v102;
            const int8_t* v104 = (const int8_t*) v103;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v105 = *(const int8_t *)(v104);
            vint16m1_t v106 = v47;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v107 = __riscv_vwmacc_vx_i16m1(v106, v105, v61, 8);
            v47 = v107;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v108 = v62 + 3;
            size_t v109 = 16 + v108;
            const uint8_t* v110 = v34 + v109;
            const int8_t* v111 = (const int8_t*) v110;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v112 = *(const int8_t *)(v111);
            vint16m1_t v113 = v41;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v114 = __riscv_vwmacc_vx_i16m1(v113, v112, v59, 8);
            v41 = v114;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v115 = v62 + 3;
            size_t v116 = 16 + 64;
            size_t v117 = v116 + v115;
            const uint8_t* v118 = v34 + v117;
            const int8_t* v119 = (const int8_t*) v118;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v120 = *(const int8_t *)(v119);
            vint16m1_t v121 = v49;
            // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v122 = __riscv_vwmacc_vx_i16m1(v121, v120, v61, 8);
            v49 = v122;
          }
          vint16m1_t v123 = v35;
          vint16m1_t v124 = v43;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v125 = __riscv_vwadd_vv_i32m2(v123, v124, 8);
          vint16m1_t v126 = v37;
          vint16m1_t v127 = v45;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v128 = __riscv_vwadd_vv_i32m2(v126, v127, 8);
          vint16m1_t v129 = v39;
          vint16m1_t v130 = v47;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v131 = __riscv_vwadd_vv_i32m2(v129, v130, 8);
          vint16m1_t v132 = v41;
          vint16m1_t v133 = v49;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v134 = __riscv_vwadd_vv_i32m2(v132, v133, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v135 = v21 * 2;
          const uint8_t* v136 = v32 + v135;
          const _Float16* v137 = (const _Float16*) v136;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v138 = __riscv_vle16_v_f16m1(v137, 8);
          size_t v139 = 32 + v135;
          const uint8_t* v140 = v32 + v139;
          const _Float16* v141 = (const _Float16*) v140;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v142 = __riscv_vle16_v_f16m1(v141, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v143 = v34 + 0;
          const _Float16* v144 = (const _Float16*) v143;
          _Float16 v145 = *(const _Float16 *)(v144);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v146 = __riscv_vfwmul_vf_f32m2(v138, v145, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v147 = __riscv_vfcvt_f_x_v_f32m2(v125, 8);
          vfloat32m2_t v148 = v26;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v149 = __riscv_vfmacc_vv_f32m2(v148, v147, v146, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v150 = v34 + 8;
          const _Float16* v151 = (const _Float16*) v150;
          _Float16 v152 = *(const _Float16 *)(v151);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v153 = __riscv_vfwmul_vf_f32m2(v142, v152, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v154 = __riscv_vfadd_vv_f32m2(v149, v153, 8);
          v26 = v154;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v155 = v34 + 2;
          const _Float16* v156 = (const _Float16*) v155;
          _Float16 v157 = *(const _Float16 *)(v156);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v158 = __riscv_vfwmul_vf_f32m2(v138, v157, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v159 = __riscv_vfcvt_f_x_v_f32m2(v128, 8);
          vfloat32m2_t v160 = v27;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v161 = __riscv_vfmacc_vv_f32m2(v160, v159, v158, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v162 = v34 + 10;
          const _Float16* v163 = (const _Float16*) v162;
          _Float16 v164 = *(const _Float16 *)(v163);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v165 = __riscv_vfwmul_vf_f32m2(v142, v164, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v166 = __riscv_vfadd_vv_f32m2(v161, v165, 8);
          v27 = v166;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v167 = v34 + 4;
          const _Float16* v168 = (const _Float16*) v167;
          _Float16 v169 = *(const _Float16 *)(v168);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v170 = __riscv_vfwmul_vf_f32m2(v138, v169, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v171 = __riscv_vfcvt_f_x_v_f32m2(v131, 8);
          vfloat32m2_t v172 = v28;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v173 = __riscv_vfmacc_vv_f32m2(v172, v171, v170, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v174 = v34 + 12;
          const _Float16* v175 = (const _Float16*) v174;
          _Float16 v176 = *(const _Float16 *)(v175);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v177 = __riscv_vfwmul_vf_f32m2(v142, v176, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v178 = __riscv_vfadd_vv_f32m2(v173, v177, 8);
          v28 = v178;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v179 = v34 + 6;
          const _Float16* v180 = (const _Float16*) v179;
          _Float16 v181 = *(const _Float16 *)(v180);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v182 = __riscv_vfwmul_vf_f32m2(v138, v181, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v183 = __riscv_vfcvt_f_x_v_f32m2(v134, 8);
          vfloat32m2_t v184 = v29;
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v185 = __riscv_vfmacc_vv_f32m2(v184, v183, v182, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v186 = v34 + 14;
          const _Float16* v187 = (const _Float16*) v186;
          _Float16 v188 = *(const _Float16 *)(v187);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v189 = __riscv_vfwmul_vf_f32m2(v142, v188, 8);
          // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v190 = __riscv_vfadd_vv_f32m2(v185, v189, 8);
          v29 = v190;
        }
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v191 = v12 * 4;
        size_t v192 = v191 + 0;
        size_t v193 = v192 * v2;
        size_t v194 = v16 * 16;
        size_t v195 = v193 + v194;
        size_t v196 = v195 + v21;
        float* v197 = v4 + v196;
        vfloat32m2_t v198 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v197, v198, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v199 = v12 * 4;
        size_t v200 = v199 + 1;
        size_t v201 = v200 * v2;
        size_t v202 = v16 * 16;
        size_t v203 = v201 + v202;
        size_t v204 = v203 + v21;
        float* v205 = v4 + v204;
        vfloat32m2_t v206 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v205, v206, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v207 = v12 * 4;
        size_t v208 = v207 + 2;
        size_t v209 = v208 * v2;
        size_t v210 = v16 * 16;
        size_t v211 = v209 + v210;
        size_t v212 = v211 + v21;
        float* v213 = v4 + v212;
        vfloat32m2_t v214 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v213, v214, 8);
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v215 = v12 * 4;
        size_t v216 = v215 + 3;
        size_t v217 = v216 * v2;
        size_t v218 = v16 * 16;
        size_t v219 = v217 + v218;
        size_t v220 = v219 + v21;
        float* v221 = v4 + v220;
        vfloat32m2_t v222 = v29;
        // tcrv_emitc.source_op=tcrv_rvv.typed_repack_gemm_loop_body role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v221, v222, 8);
      }
    }
  }
  return;
}


