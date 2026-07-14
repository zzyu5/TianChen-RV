#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v8 = __riscv_vsetvl_e32m1(v3);
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
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v22 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v23 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
        vfloat32m2_t v24 = __riscv_vfmv_v_f_f32m2(0.0f, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
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
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_block_base
          size_t v31 = v30 * 384;
          const uint8_t* v32 = v19 + v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 144;
          const uint8_t* v34 = v15 + v33;
          vint16m1_t v35;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v36 = __riscv_vmv_v_x_i16m1(0, 16);
          v35 = v36;
          vint16m1_t v37;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v38 = __riscv_vmv_v_x_i16m1(0, 16);
          v37 = v38;
          vint16m1_t v39;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v40 = __riscv_vmv_v_x_i16m1(0, 16);
          v39 = v40;
          vint16m1_t v41;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v42 = __riscv_vmv_v_x_i16m1(0, 16);
          v41 = v42;
          vint16m1_t v43;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v44 = __riscv_vmv_v_x_i16m1(0, 16);
          v43 = v44;
          vint16m1_t v45;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v46 = __riscv_vmv_v_x_i16m1(0, 16);
          v45 = v46;
          vint16m1_t v47;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v48 = __riscv_vmv_v_x_i16m1(0, 16);
          v47 = v48;
          vint16m1_t v49;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v50 = __riscv_vmv_v_x_i16m1(0, 16);
          v49 = v50;
          for (size_t v51 = 0; v51 < 16; v51 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v52 = v51 * 16;
            size_t v53 = 64 + v52;
            size_t v54 = v53 + v21;
            const uint8_t* v55 = v32 + v54;
            const uint8_t* v56 = (const uint8_t*) v55;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
            vuint8mf2_t v57 = __riscv_vle8_v_u8mf2(v56, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v58 = v51 * 2;
            size_t v59 = 320 + v58;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v60 = 320 + 32;
            size_t v61 = v60 + v58;
            const uint8_t* v62 = v32 + v59;
            const uint16_t* v63 = (const uint16_t*) v62;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v64 = (uint16_t)*(const uint16_t *)(v63);
            const uint8_t* v65 = v32 + v61;
            const uint16_t* v66 = (const uint16_t*) v65;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_mask_scalar
            int32_t v67 = (uint16_t)*(const uint16_t *)(v66);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
            vuint16m1_t v68 = __riscv_vmv_v_x_u16m1(v64, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
            vuint16m1_t v69 = __riscv_vid_v_u16m1(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
            vuint16m1_t v70 = __riscv_vadd_vx_u16m1(v69, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
            vuint16m1_t v71 = __riscv_vsrl_vv_u16m1(v68, v70, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
            vuint16m1_t v72 = __riscv_vand_vx_u16m1(v71, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
            vuint16m1_t v73 = __riscv_vsll_vx_u16m1(v72, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
            vuint8mf2_t v74 = __riscv_vncvt_x_x_w_u8mf2(v73, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
            vuint8mf2_t v75 = __riscv_vand_vx_u8mf2(v57, 15, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
            vuint8mf2_t v76 = __riscv_vor_vv_u8mf2(v75, v74, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v77 = __riscv_vreinterpret_v_u8mf2_i8mf2(v76);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
            vuint16m1_t v78 = __riscv_vmv_v_x_u16m1(v67, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
            vuint16m1_t v79 = __riscv_vid_v_u16m1(16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
            vuint16m1_t v80 = __riscv_vadd_vx_u16m1(v79, v21, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
            vuint16m1_t v81 = __riscv_vsrl_vv_u16m1(v78, v80, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
            vuint16m1_t v82 = __riscv_vand_vx_u16m1(v81, 1, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
            vuint16m1_t v83 = __riscv_vsll_vx_u16m1(v82, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
            vuint8mf2_t v84 = __riscv_vncvt_x_x_w_u8mf2(v83, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
            vuint8mf2_t v85 = __riscv_vsrl_vx_u8mf2(v57, 4, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
            vuint8mf2_t v86 = __riscv_vor_vv_u8mf2(v85, v84, 16);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v87 = __riscv_vreinterpret_v_u8mf2_i8mf2(v86);
            size_t v88 = v51 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v89 = v88 + 0;
            size_t v90 = 16 + v89;
            const uint8_t* v91 = v34 + v90;
            const int8_t* v92 = (const int8_t*) v91;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v93 = *(const int8_t *)(v92);
            vint16m1_t v94 = v35;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v95 = __riscv_vwmacc_vx_i16m1(v94, v93, v77, 16);
            v35 = v95;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v96 = v88 + 0;
            size_t v97 = 16 + 64;
            size_t v98 = v97 + v96;
            const uint8_t* v99 = v34 + v98;
            const int8_t* v100 = (const int8_t*) v99;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v101 = *(const int8_t *)(v100);
            vint16m1_t v102 = v43;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v103 = __riscv_vwmacc_vx_i16m1(v102, v101, v87, 16);
            v43 = v103;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v104 = v88 + 1;
            size_t v105 = 16 + v104;
            const uint8_t* v106 = v34 + v105;
            const int8_t* v107 = (const int8_t*) v106;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v108 = *(const int8_t *)(v107);
            vint16m1_t v109 = v37;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v110 = __riscv_vwmacc_vx_i16m1(v109, v108, v77, 16);
            v37 = v110;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v111 = v88 + 1;
            size_t v112 = 16 + 64;
            size_t v113 = v112 + v111;
            const uint8_t* v114 = v34 + v113;
            const int8_t* v115 = (const int8_t*) v114;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v116 = *(const int8_t *)(v115);
            vint16m1_t v117 = v45;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v118 = __riscv_vwmacc_vx_i16m1(v117, v116, v87, 16);
            v45 = v118;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v119 = v88 + 2;
            size_t v120 = 16 + v119;
            const uint8_t* v121 = v34 + v120;
            const int8_t* v122 = (const int8_t*) v121;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v123 = *(const int8_t *)(v122);
            vint16m1_t v124 = v39;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v125 = __riscv_vwmacc_vx_i16m1(v124, v123, v77, 16);
            v39 = v125;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v126 = v88 + 2;
            size_t v127 = 16 + 64;
            size_t v128 = v127 + v126;
            const uint8_t* v129 = v34 + v128;
            const int8_t* v130 = (const int8_t*) v129;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v131 = *(const int8_t *)(v130);
            vint16m1_t v132 = v47;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v133 = __riscv_vwmacc_vx_i16m1(v132, v131, v87, 16);
            v47 = v133;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v134 = v88 + 3;
            size_t v135 = 16 + v134;
            const uint8_t* v136 = v34 + v135;
            const int8_t* v137 = (const int8_t*) v136;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v138 = *(const int8_t *)(v137);
            vint16m1_t v139 = v41;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v140 = __riscv_vwmacc_vx_i16m1(v139, v138, v77, 16);
            v41 = v140;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v141 = v88 + 3;
            size_t v142 = 16 + 64;
            size_t v143 = v142 + v141;
            const uint8_t* v144 = v34 + v143;
            const int8_t* v145 = (const int8_t*) v144;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v146 = *(const int8_t *)(v145);
            vint16m1_t v147 = v49;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v148 = __riscv_vwmacc_vx_i16m1(v147, v146, v87, 16);
            v49 = v148;
          }
          vint16m1_t v149 = v35;
          vint16m1_t v150 = v43;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v151 = __riscv_vwadd_vv_i32m2(v149, v150, 16);
          vint16m1_t v152 = v37;
          vint16m1_t v153 = v45;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v154 = __riscv_vwadd_vv_i32m2(v152, v153, 16);
          vint16m1_t v155 = v39;
          vint16m1_t v156 = v47;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v157 = __riscv_vwadd_vv_i32m2(v155, v156, 16);
          vint16m1_t v158 = v41;
          vint16m1_t v159 = v49;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v160 = __riscv_vwadd_vv_i32m2(v158, v159, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v161 = v21 * 2;
          const uint8_t* v162 = v32 + v161;
          const _Float16* v163 = (const _Float16*) v162;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v164 = __riscv_vle16_v_f16m1(v163, 16);
          size_t v165 = 32 + v161;
          const uint8_t* v166 = v32 + v165;
          const _Float16* v167 = (const _Float16*) v166;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v168 = __riscv_vle16_v_f16m1(v167, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v169 = v34 + 0;
          const _Float16* v170 = (const _Float16*) v169;
          _Float16 v171 = *(const _Float16 *)(v170);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v172 = __riscv_vfwmul_vf_f32m2(v164, v171, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v173 = __riscv_vfcvt_f_x_v_f32m2(v151, 16);
          vfloat32m2_t v174 = v26;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v175 = __riscv_vfmacc_vv_f32m2(v174, v173, v172, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v176 = v34 + 8;
          const _Float16* v177 = (const _Float16*) v176;
          _Float16 v178 = *(const _Float16 *)(v177);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v179 = __riscv_vfwmul_vf_f32m2(v168, v178, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v180 = __riscv_vfadd_vv_f32m2(v175, v179, 16);
          v26 = v180;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v181 = v34 + 2;
          const _Float16* v182 = (const _Float16*) v181;
          _Float16 v183 = *(const _Float16 *)(v182);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v184 = __riscv_vfwmul_vf_f32m2(v164, v183, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v185 = __riscv_vfcvt_f_x_v_f32m2(v154, 16);
          vfloat32m2_t v186 = v27;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v187 = __riscv_vfmacc_vv_f32m2(v186, v185, v184, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v188 = v34 + 10;
          const _Float16* v189 = (const _Float16*) v188;
          _Float16 v190 = *(const _Float16 *)(v189);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v191 = __riscv_vfwmul_vf_f32m2(v168, v190, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v192 = __riscv_vfadd_vv_f32m2(v187, v191, 16);
          v27 = v192;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v193 = v34 + 4;
          const _Float16* v194 = (const _Float16*) v193;
          _Float16 v195 = *(const _Float16 *)(v194);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v196 = __riscv_vfwmul_vf_f32m2(v164, v195, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v197 = __riscv_vfcvt_f_x_v_f32m2(v157, 16);
          vfloat32m2_t v198 = v28;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v199 = __riscv_vfmacc_vv_f32m2(v198, v197, v196, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v200 = v34 + 12;
          const _Float16* v201 = (const _Float16*) v200;
          _Float16 v202 = *(const _Float16 *)(v201);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v203 = __riscv_vfwmul_vf_f32m2(v168, v202, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v204 = __riscv_vfadd_vv_f32m2(v199, v203, 16);
          v28 = v204;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v205 = v34 + 6;
          const _Float16* v206 = (const _Float16*) v205;
          _Float16 v207 = *(const _Float16 *)(v206);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v208 = __riscv_vfwmul_vf_f32m2(v164, v207, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v209 = __riscv_vfcvt_f_x_v_f32m2(v160, 16);
          vfloat32m2_t v210 = v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v211 = __riscv_vfmacc_vv_f32m2(v210, v209, v208, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_sum_scalar
          const uint8_t* v212 = v34 + 14;
          const _Float16* v213 = (const _Float16*) v212;
          _Float16 v214 = *(const _Float16 *)(v213);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v215 = __riscv_vfwmul_vf_f32m2(v168, v214, 16);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfadd_vv_f32m2
          vfloat32m2_t v216 = __riscv_vfadd_vv_f32m2(v211, v215, 16);
          v29 = v216;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v217 = v12 * 4;
        size_t v218 = v217 + 0;
        size_t v219 = v218 * v2;
        size_t v220 = v16 * 16;
        size_t v221 = v219 + v220;
        size_t v222 = v221 + v21;
        float* v223 = v4 + v222;
        vfloat32m2_t v224 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v223, v224, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v225 = v12 * 4;
        size_t v226 = v225 + 1;
        size_t v227 = v226 * v2;
        size_t v228 = v16 * 16;
        size_t v229 = v227 + v228;
        size_t v230 = v229 + v21;
        float* v231 = v4 + v230;
        vfloat32m2_t v232 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v231, v232, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v233 = v12 * 4;
        size_t v234 = v233 + 2;
        size_t v235 = v234 * v2;
        size_t v236 = v16 * 16;
        size_t v237 = v235 + v236;
        size_t v238 = v237 + v21;
        float* v239 = v4 + v238;
        vfloat32m2_t v240 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v239, v240, 16);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v241 = v12 * 4;
        size_t v242 = v241 + 3;
        size_t v243 = v242 * v2;
        size_t v244 = v16 * 16;
        size_t v245 = v243 + v244;
        size_t v246 = v245 + v21;
        float* v247 = v4 + v246;
        vfloat32m2_t v248 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v247, v248, 16);
      }
    }
  }
  return;
}


