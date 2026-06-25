#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5, size_t v6, size_t v7) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=block_count
  size_t v9 = v1 / 32;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=row_group_count
  size_t v10 = v5 / 4;
  // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=col_group_count
  size_t v11 = v6 / 16;
  for (size_t v12 = 0; v12 < v10; v12 += 1) {
    // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_group_base
    size_t v13 = v12 * v9;
    size_t v14 = v13 * 136;
    const uint8_t* v15 = v4 + v14;
    for (size_t v16 = 0; v16 < v11; v16 += 1) {
      // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_group_base
      size_t v17 = v16 * v9;
      size_t v18 = v17 * 288;
      const uint8_t* v19 = v3 + v18;
      for (size_t v20 = 0; v20 < 1; v20 += 1) {
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=half_row_offset
        size_t v21 = v20 * 16;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v22 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v24 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v25 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        vfloat32m2_t v26;
        v26 = v22;
        vfloat32m2_t v27;
        v27 = v23;
        vfloat32m2_t v28;
        v28 = v24;
        vfloat32m2_t v29;
        v29 = v25;
        for (size_t v30 = 0; v30 < v9; v30 += 1) {
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_block_base
          size_t v31 = v30 * 288;
          const uint8_t* v32 = v19 + v31;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 136;
          const uint8_t* v34 = v15 + v33;
          vint16m1_t v35;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v36 = __riscv_vmv_v_x_i16m1(0, 16);
          v35 = v36;
          vint16m1_t v37;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v38 = __riscv_vmv_v_x_i16m1(0, 16);
          v37 = v38;
          vint16m1_t v39;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v40 = __riscv_vmv_v_x_i16m1(0, 16);
          v39 = v40;
          vint16m1_t v41;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v42 = __riscv_vmv_v_x_i16m1(0, 16);
          v41 = v42;
          vint16m1_t v43;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v44 = __riscv_vmv_v_x_i16m1(0, 16);
          v43 = v44;
          vint16m1_t v45;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v46 = __riscv_vmv_v_x_i16m1(0, 16);
          v45 = v46;
          vint16m1_t v47;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v48 = __riscv_vmv_v_x_i16m1(0, 16);
          v47 = v48;
          vint16m1_t v49;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v50 = __riscv_vmv_v_x_i16m1(0, 16);
          v49 = v50;
          for (size_t v51 = 0; v51 < 16; v51 += 1) {
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v52 = v51 * 16;
            size_t v53 = 32 + v52;
            size_t v54 = v53 + v21;
            const uint8_t* v55 = v32 + v54;
            const int8_t* v56 = (const int8_t*) v55;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle8_v_i8mf2
            vint8mf2_t v57 = __riscv_vle8_v_i8mf2(v56, 16);
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_i8mf2
            vint8mf2_t v58 = __riscv_vsll_vx_i8mf2(v57, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
            vint8mf2_t v59 = __riscv_vsra_vx_i8mf2(v58, 4, 16);
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsra_vx_i8mf2
            vint8mf2_t v60 = __riscv_vsra_vx_i8mf2(v57, 4, 16);
            size_t v61 = v51 * 4;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v62 = v61 + 0;
            size_t v63 = 8 + v62;
            const uint8_t* v64 = v34 + v63;
            const int8_t* v65 = (const int8_t*) v64;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v66 = *(const int8_t *)(v65);
            vint16m1_t v67 = v35;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v68 = __riscv_vwmacc_vx_i16m1(v67, v66, v59, 16);
            v35 = v68;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v69 = v61 + 0;
            size_t v70 = 8 + 64;
            size_t v71 = v70 + v69;
            const uint8_t* v72 = v34 + v71;
            const int8_t* v73 = (const int8_t*) v72;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v74 = *(const int8_t *)(v73);
            vint16m1_t v75 = v43;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v76 = __riscv_vwmacc_vx_i16m1(v75, v74, v60, 16);
            v43 = v76;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v77 = v61 + 1;
            size_t v78 = 8 + v77;
            const uint8_t* v79 = v34 + v78;
            const int8_t* v80 = (const int8_t*) v79;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v81 = *(const int8_t *)(v80);
            vint16m1_t v82 = v37;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v83 = __riscv_vwmacc_vx_i16m1(v82, v81, v59, 16);
            v37 = v83;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v84 = v61 + 1;
            size_t v85 = 8 + 64;
            size_t v86 = v85 + v84;
            const uint8_t* v87 = v34 + v86;
            const int8_t* v88 = (const int8_t*) v87;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v89 = *(const int8_t *)(v88);
            vint16m1_t v90 = v45;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v91 = __riscv_vwmacc_vx_i16m1(v90, v89, v60, 16);
            v45 = v91;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v92 = v61 + 2;
            size_t v93 = 8 + v92;
            const uint8_t* v94 = v34 + v93;
            const int8_t* v95 = (const int8_t*) v94;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v96 = *(const int8_t *)(v95);
            vint16m1_t v97 = v39;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v98 = __riscv_vwmacc_vx_i16m1(v97, v96, v59, 16);
            v39 = v98;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v99 = v61 + 2;
            size_t v100 = 8 + 64;
            size_t v101 = v100 + v99;
            const uint8_t* v102 = v34 + v101;
            const int8_t* v103 = (const int8_t*) v102;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v104 = *(const int8_t *)(v103);
            vint16m1_t v105 = v47;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v106 = __riscv_vwmacc_vx_i16m1(v105, v104, v60, 16);
            v47 = v106;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v107 = v61 + 3;
            size_t v108 = 8 + v107;
            const uint8_t* v109 = v34 + v108;
            const int8_t* v110 = (const int8_t*) v109;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v111 = *(const int8_t *)(v110);
            vint16m1_t v112 = v41;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v113 = __riscv_vwmacc_vx_i16m1(v112, v111, v59, 16);
            v41 = v113;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v114 = v61 + 3;
            size_t v115 = 8 + 64;
            size_t v116 = v115 + v114;
            const uint8_t* v117 = v34 + v116;
            const int8_t* v118 = (const int8_t*) v117;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v119 = *(const int8_t *)(v118);
            vint16m1_t v120 = v49;
            // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v121 = __riscv_vwmacc_vx_i16m1(v120, v119, v60, 16);
            v49 = v121;
          }
          vint16m1_t v122 = v35;
          vint16m1_t v123 = v43;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v124 = __riscv_vwadd_vv_i32m2(v122, v123, 16);
          vint16m1_t v125 = v37;
          vint16m1_t v126 = v45;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v127 = __riscv_vwadd_vv_i32m2(v125, v126, 16);
          vint16m1_t v128 = v39;
          vint16m1_t v129 = v47;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v130 = __riscv_vwadd_vv_i32m2(v128, v129, 16);
          vint16m1_t v131 = v41;
          vint16m1_t v132 = v49;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v133 = __riscv_vwadd_vv_i32m2(v131, v132, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v134 = v21 * 2;
          const uint8_t* v135 = v32 + v134;
          const _Float16* v136 = (const _Float16*) v135;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v137 = __riscv_vle16_v_f16m1(v136, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v138 = v34 + 0;
          const _Float16* v139 = (const _Float16*) v138;
          _Float16 v140 = *(const _Float16 *)(v139);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v141 = __riscv_vfwmul_vf_f32m2(v137, v140, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v142 = __riscv_vfcvt_f_x_v_f32m2(v124, 16);
          vfloat32m2_t v143 = v26;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v144 = __riscv_vfmacc_vv_f32m2(v143, v142, v141, 16);
          v26 = v144;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v145 = v34 + 2;
          const _Float16* v146 = (const _Float16*) v145;
          _Float16 v147 = *(const _Float16 *)(v146);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v148 = __riscv_vfwmul_vf_f32m2(v137, v147, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v149 = __riscv_vfcvt_f_x_v_f32m2(v127, 16);
          vfloat32m2_t v150 = v27;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v151 = __riscv_vfmacc_vv_f32m2(v150, v149, v148, 16);
          v27 = v151;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v152 = v34 + 4;
          const _Float16* v153 = (const _Float16*) v152;
          _Float16 v154 = *(const _Float16 *)(v153);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v155 = __riscv_vfwmul_vf_f32m2(v137, v154, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v156 = __riscv_vfcvt_f_x_v_f32m2(v130, 16);
          vfloat32m2_t v157 = v28;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v158 = __riscv_vfmacc_vv_f32m2(v157, v156, v155, 16);
          v28 = v158;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v159 = v34 + 6;
          const _Float16* v160 = (const _Float16*) v159;
          _Float16 v161 = *(const _Float16 *)(v160);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v162 = __riscv_vfwmul_vf_f32m2(v137, v161, 16);
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v163 = __riscv_vfcvt_f_x_v_f32m2(v133, 16);
          vfloat32m2_t v164 = v29;
          // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v165 = __riscv_vfmacc_vv_f32m2(v164, v163, v162, 16);
          v29 = v165;
        }
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v166 = v12 * 4;
        size_t v167 = v166 + 0;
        size_t v168 = v167 * v7;
        size_t v169 = v16 * 16;
        size_t v170 = v168 + v169;
        size_t v171 = v170 + v21;
        float* v172 = v2 + v171;
        vfloat32m2_t v173 = v26;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v172, v173, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v174 = v12 * 4;
        size_t v175 = v174 + 1;
        size_t v176 = v175 * v7;
        size_t v177 = v16 * 16;
        size_t v178 = v176 + v177;
        size_t v179 = v178 + v21;
        float* v180 = v2 + v179;
        vfloat32m2_t v181 = v27;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v180, v181, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v182 = v12 * 4;
        size_t v183 = v182 + 2;
        size_t v184 = v183 * v7;
        size_t v185 = v16 * 16;
        size_t v186 = v184 + v185;
        size_t v187 = v186 + v21;
        float* v188 = v2 + v187;
        vfloat32m2_t v189 = v28;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v188, v189, 16);
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=output_addr
        size_t v190 = v12 * 4;
        size_t v191 = v190 + 3;
        size_t v192 = v191 * v7;
        size_t v193 = v16 * 16;
        size_t v194 = v192 + v193;
        size_t v195 = v194 + v21;
        float* v196 = v2 + v195;
        vfloat32m2_t v197 = v29;
        // tcrv_emitc.source_op=tcrv_rvv.repack_gemm_q4_0_q8_0 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v196, v197, 16);
      }
    }
  }
  vint32m1_t v198 = __riscv_vmv_v_x_i32m1(0, 1);
  return;
}


