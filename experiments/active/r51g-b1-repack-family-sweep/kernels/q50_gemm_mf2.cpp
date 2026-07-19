#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void q50_gemm_mf2(size_t v1, size_t v2, size_t v3, float* v4, size_t v5, const uint8_t* v6, const uint8_t* v7) {
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
          size_t v31 = v30 * 352;
          const uint8_t* v32 = v19 + v31;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_block_base
          size_t v33 = v30 * 136;
          const uint8_t* v34 = v15 + v33;
          vint16m1_t v35;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v36 = __riscv_vmv_v_x_i16m1(0, 8);
          v35 = v36;
          vint16m1_t v37;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v38 = __riscv_vmv_v_x_i16m1(0, 8);
          v37 = v38;
          vint16m1_t v39;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v40 = __riscv_vmv_v_x_i16m1(0, 8);
          v39 = v40;
          vint16m1_t v41;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v42 = __riscv_vmv_v_x_i16m1(0, 8);
          v41 = v42;
          vint16m1_t v43;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v44 = __riscv_vmv_v_x_i16m1(0, 8);
          v43 = v44;
          vint16m1_t v45;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v46 = __riscv_vmv_v_x_i16m1(0, 8);
          v45 = v46;
          vint16m1_t v47;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v48 = __riscv_vmv_v_x_i16m1(0, 8);
          v47 = v48;
          vint16m1_t v49;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i16m1
          vint16m1_t v50 = __riscv_vmv_v_x_i16m1(0, 8);
          v49 = v50;
          for (size_t v51 = 0; v51 < 16; v51 += 1) {
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_nibble_addr
            size_t v52 = v51 * 16;
            size_t v53 = 32 + v52;
            size_t v54 = v53 + v21;
            const uint8_t* v55 = v32 + v54;
            const uint8_t* v56 = (const uint8_t*) v55;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle8_v_u8mf2
            vuint8mf2_t v57 = __riscv_vle8_v_u8mf2(v56, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_lo_addr
            size_t v58 = v51 * 2;
            size_t v59 = 288 + v58;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=qh_hi_addr
            size_t v60 = 288 + 32;
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
            vuint16m1_t v68 = __riscv_vmv_v_x_u16m1(v64, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
            vuint16m1_t v69 = __riscv_vid_v_u16m1(8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
            vuint16m1_t v70 = __riscv_vadd_vx_u16m1(v69, v21, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
            vuint16m1_t v71 = __riscv_vsrl_vv_u16m1(v68, v70, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
            vuint16m1_t v72 = __riscv_vand_vx_u16m1(v71, 1, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
            vuint16m1_t v73 = __riscv_vsll_vx_u16m1(v72, 4, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
            vuint8mf2_t v74 = __riscv_vncvt_x_x_w_u8mf2(v73, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u8mf2
            vuint8mf2_t v75 = __riscv_vand_vx_u8mf2(v57, 15, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
            vuint8mf2_t v76 = __riscv_vor_vv_u8mf2(v75, v74, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v77 = __riscv_vreinterpret_v_u8mf2_i8mf2(v76);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
            vint8mf2_t v78 = __riscv_vsub_vx_i8mf2(v77, 16, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u16m1
            vuint16m1_t v79 = __riscv_vmv_v_x_u16m1(v67, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vid_v_u16m1
            vuint16m1_t v80 = __riscv_vid_v_u16m1(8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vx_u16m1
            vuint16m1_t v81 = __riscv_vadd_vx_u16m1(v80, v21, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vv_u16m1
            vuint16m1_t v82 = __riscv_vsrl_vv_u16m1(v79, v81, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vand_vx_u16m1
            vuint16m1_t v83 = __riscv_vand_vx_u16m1(v82, 1, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsll_vx_u16m1
            vuint16m1_t v84 = __riscv_vsll_vx_u16m1(v83, 4, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vncvt_x_x_w_u8mf2
            vuint8mf2_t v85 = __riscv_vncvt_x_x_w_u8mf2(v84, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsrl_vx_u8mf2
            vuint8mf2_t v86 = __riscv_vsrl_vx_u8mf2(v57, 4, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vor_vv_u8mf2
            vuint8mf2_t v87 = __riscv_vor_vv_u8mf2(v86, v85, 8);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u8mf2_i8mf2
            vint8mf2_t v88 = __riscv_vreinterpret_v_u8mf2_i8mf2(v87);
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsub_vx_i8mf2
            vint8mf2_t v89 = __riscv_vsub_vx_i8mf2(v88, 16, 8);
            size_t v90 = v51 * 4;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v91 = v90 + 0;
            size_t v92 = 8 + v91;
            const uint8_t* v93 = v34 + v92;
            const int8_t* v94 = (const int8_t*) v93;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v95 = *(const int8_t *)(v94);
            vint16m1_t v96 = v35;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v97 = __riscv_vwmacc_vx_i16m1(v96, v95, v78, 8);
            v35 = v97;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v98 = v90 + 0;
            size_t v99 = 8 + 64;
            size_t v100 = v99 + v98;
            const uint8_t* v101 = v34 + v100;
            const int8_t* v102 = (const int8_t*) v101;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v103 = *(const int8_t *)(v102);
            vint16m1_t v104 = v43;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v105 = __riscv_vwmacc_vx_i16m1(v104, v103, v89, 8);
            v43 = v105;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v106 = v90 + 1;
            size_t v107 = 8 + v106;
            const uint8_t* v108 = v34 + v107;
            const int8_t* v109 = (const int8_t*) v108;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v110 = *(const int8_t *)(v109);
            vint16m1_t v111 = v37;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v112 = __riscv_vwmacc_vx_i16m1(v111, v110, v78, 8);
            v37 = v112;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v113 = v90 + 1;
            size_t v114 = 8 + 64;
            size_t v115 = v114 + v113;
            const uint8_t* v116 = v34 + v115;
            const int8_t* v117 = (const int8_t*) v116;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v118 = *(const int8_t *)(v117);
            vint16m1_t v119 = v45;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v120 = __riscv_vwmacc_vx_i16m1(v119, v118, v89, 8);
            v45 = v120;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v121 = v90 + 2;
            size_t v122 = 8 + v121;
            const uint8_t* v123 = v34 + v122;
            const int8_t* v124 = (const int8_t*) v123;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v125 = *(const int8_t *)(v124);
            vint16m1_t v126 = v39;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v127 = __riscv_vwmacc_vx_i16m1(v126, v125, v78, 8);
            v39 = v127;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v128 = v90 + 2;
            size_t v129 = 8 + 64;
            size_t v130 = v129 + v128;
            const uint8_t* v131 = v34 + v130;
            const int8_t* v132 = (const int8_t*) v131;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v133 = *(const int8_t *)(v132);
            vint16m1_t v134 = v47;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v135 = __riscv_vwmacc_vx_i16m1(v134, v133, v89, 8);
            v47 = v135;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_lo
            size_t v136 = v90 + 3;
            size_t v137 = 8 + v136;
            const uint8_t* v138 = v34 + v137;
            const int8_t* v139 = (const int8_t*) v138;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v140 = *(const int8_t *)(v139);
            vint16m1_t v141 = v41;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v142 = __riscv_vwmacc_vx_i16m1(v141, v140, v78, 8);
            v41 = v142;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_addr_hi
            size_t v143 = v90 + 3;
            size_t v144 = 8 + 64;
            size_t v145 = v144 + v143;
            const uint8_t* v146 = v34 + v145;
            const int8_t* v147 = (const int8_t*) v146;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_quant_scalar
            int32_t v148 = *(const int8_t *)(v147);
            vint16m1_t v149 = v49;
            // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwmacc_vx_i16m1
            vint16m1_t v150 = __riscv_vwmacc_vx_i16m1(v149, v148, v89, 8);
            v49 = v150;
          }
          vint16m1_t v151 = v35;
          vint16m1_t v152 = v43;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v153 = __riscv_vwadd_vv_i32m2(v151, v152, 8);
          vint16m1_t v154 = v37;
          vint16m1_t v155 = v45;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v156 = __riscv_vwadd_vv_i32m2(v154, v155, 8);
          vint16m1_t v157 = v39;
          vint16m1_t v158 = v47;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v159 = __riscv_vwadd_vv_i32m2(v157, v158, 8);
          vint16m1_t v160 = v41;
          vint16m1_t v161 = v49;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vwadd_vv_i32m2
          vint32m2_t v162 = __riscv_vwadd_vv_i32m2(v160, v161, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weight_scale_addr
          size_t v163 = v21 * 2;
          const uint8_t* v164 = v32 + v163;
          const _Float16* v165 = (const _Float16*) v164;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle16_v_f16m1
          vfloat16m1_t v166 = __riscv_vle16_v_f16m1(v165, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v167 = v34 + 0;
          const _Float16* v168 = (const _Float16*) v167;
          _Float16 v169 = *(const _Float16 *)(v168);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v170 = __riscv_vfwmul_vf_f32m2(v166, v169, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v171 = __riscv_vfcvt_f_x_v_f32m2(v153, 8);
          vfloat32m2_t v172 = v26;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v173 = __riscv_vfmacc_vv_f32m2(v172, v171, v170, 8);
          v26 = v173;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v174 = v34 + 2;
          const _Float16* v175 = (const _Float16*) v174;
          _Float16 v176 = *(const _Float16 *)(v175);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v177 = __riscv_vfwmul_vf_f32m2(v166, v176, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v178 = __riscv_vfcvt_f_x_v_f32m2(v156, 8);
          vfloat32m2_t v179 = v27;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v180 = __riscv_vfmacc_vv_f32m2(v179, v178, v177, 8);
          v27 = v180;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v181 = v34 + 4;
          const _Float16* v182 = (const _Float16*) v181;
          _Float16 v183 = *(const _Float16 *)(v182);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v184 = __riscv_vfwmul_vf_f32m2(v166, v183, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v185 = __riscv_vfcvt_f_x_v_f32m2(v159, 8);
          vfloat32m2_t v186 = v28;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v187 = __riscv_vfmacc_vv_f32m2(v186, v185, v184, 8);
          v28 = v187;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=act_scale_scalar
          const uint8_t* v188 = v34 + 6;
          const _Float16* v189 = (const _Float16*) v188;
          _Float16 v190 = *(const _Float16 *)(v189);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfwmul_vf_f32m2
          vfloat32m2_t v191 = __riscv_vfwmul_vf_f32m2(v166, v190, 8);
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m2
          vfloat32m2_t v192 = __riscv_vfcvt_f_x_v_f32m2(v162, 8);
          vfloat32m2_t v193 = v29;
          // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
          vfloat32m2_t v194 = __riscv_vfmacc_vv_f32m2(v193, v192, v191, 8);
          v29 = v194;
        }
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v195 = v12 * 4;
        size_t v196 = v195 + 0;
        size_t v197 = v196 * v2;
        size_t v198 = v16 * 16;
        size_t v199 = v197 + v198;
        size_t v200 = v199 + v21;
        float* v201 = v4 + v200;
        vfloat32m2_t v202 = v26;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v201, v202, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v203 = v12 * 4;
        size_t v204 = v203 + 1;
        size_t v205 = v204 * v2;
        size_t v206 = v16 * 16;
        size_t v207 = v205 + v206;
        size_t v208 = v207 + v21;
        float* v209 = v4 + v208;
        vfloat32m2_t v210 = v27;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v209, v210, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v211 = v12 * 4;
        size_t v212 = v211 + 2;
        size_t v213 = v212 * v2;
        size_t v214 = v16 * 16;
        size_t v215 = v213 + v214;
        size_t v216 = v215 + v21;
        float* v217 = v4 + v216;
        vfloat32m2_t v218 = v28;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v217, v218, 8);
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=output_addr
        size_t v219 = v12 * 4;
        size_t v220 = v219 + 3;
        size_t v221 = v220 * v2;
        size_t v222 = v16 * 16;
        size_t v223 = v221 + v222;
        size_t v224 = v223 + v21;
        float* v225 = v4 + v224;
        vfloat32m2_t v226 = v29;
        // weft_emitc.source_op=weft_rvv.typed_repack_gemm_loop_body role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
        __riscv_vse32_v_f32m2(v225, v226, 8);
      }
    }
  }
  return;
}


