// [G7-L3 q5_0@k1] VLEN=256 one-strip (half_lanes=16) repack GEVM for q5_0.
// Authored typed region (the arm the front door DECLINES at K1-VLEN256 decode: the
// q5_0 decode quant_contraction ERRORS "no block-dot decline path" because the 5th
// bit qh cannot be block-dot lowered as q4_0-nibble; so the repack GEVM one-strip is
// authored directly here), adapted from the q4_0-vlen256 gemv one-strip fixture with
// the q5_0 deltas captured from the VLEN128 decode front-door typed region:
//   * scale_model "...-five-bit" carried by fold d-ONLY (lane_wise_vector_scale, NO min)
//   * weight_nibble_unsigned + weight_offset_bias=16 (q5_0 5-bit `((nib)|(qh<<4))-16`)
//   * weight_qh_byte_offset=288 (transposed qh plane), weight_quant_byte_offset=32
//   * block_q5_0x16 stride 352, nibbles @32, qh @288; plain block_q8_0 stride 34, qs @2.
// numHalves collapsed 2->1 (VLEN256 e16m1 = 16 lanes = one 16-block-as-lane strip).
// ABI IDENTICAL to the vlen128 q5_0 GEVM (n,s,bs,vx,bx,vy,by,nrc) so the deploy call
// site is unchanged.
// RUN: weft-opt %s --weft-rvv-lower-to-emitc

module {
  weft.exec.kernel @ggml_vec_dot_q5_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %bs attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", fold_model = "lane_wise_vector_scale", qk = 32 : i64, weight_block_stride = 352 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">):
          %sumi = weft_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index : index {kind = "repack_lane_wise_q4_x_i8_dot", weight_nibble_unsigned, weight_offset_bias = 16 : i64, weight_qh_byte_offset = 288 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
          %an0 = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi, %acc0, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %an0 : !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}
