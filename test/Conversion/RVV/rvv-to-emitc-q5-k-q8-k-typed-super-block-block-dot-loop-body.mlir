// RUN: weft-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=RETIRED

// [QH-MASK] q5_K super-block block-dot loop body (Region-A nibble unpack WITH the
// qh 5th-bit inject). This is the q4_K super-block scaffold's q5_K sibling: BRICK 1
// carries the block_q5_K format facts (weight_block_stride 176, qs@48, and the
// q5_K-only weight_qh_byte_offset@16), so emitQ4_KPlainNibbleUnpack runs under
// cx.hasQh and the SHARED [QH-MASK] emitNativeMaskStaticBitBias helper fires.
//
// The qh plane is a SINGLE-bit-plane (bit_plane_width==1) -- the native-interleaved-
// static ANALOG of q5_0/q5_1's 5th bit -- so per half h the helper isolates bit h IN
// PLACE (vand(1<<h)), lifts it to a per-lane bool (vmsne==0, TRUE where the bit is
// SET), and FUSES the +16 into ONE masked op vadd_vx_u8m2_mu on exactly the SET lanes.
// This is BYTE-EXACT to _generic's `a[l] += (hm[l] & m ? 16 : 0)` and to the RETIRED
// OLD vsrl|vand|vsll|vadd_vv expand chain: for a SINGLE bit, `nib + (bit ? 16 : 0)`
// is precisely the masked add. Both stay in the UINT8 domain before the u8->i8
// reinterpret (q5 in [0,31] stays non-negative). The nibble extract + the widening
// dot are UNCHANGED. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv).

module {
  weft.exec.kernel @q5_k_super_block_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @q5_k_super_block_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %aux8 = weft_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q5-unpacked-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scales = weft_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-scales-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %aux32 = weft_rvv.runtime_abi_value {c_name = "aux32", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q5-aux32-scratch", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q5_k_super_block_loop_body, sew = 32 : i64, source_kernel = "q5_k_super_block_loop_body_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 176 : i64, activation_block_stride = 292 : i64, fold_model = "super_block_two_level_scale_min", integer_core_lmul = "m2"} {
        ^bb0(%super_block_index: index, %sums: !weft_rvv.vector<f32, "m2">, %sumf: f32):
          // BRICK 1: plain 4-bit nibble unpack + qh 5th-bit inject -> aux8[256]
          // (Region A). block_q5_K format: stride 176, qs@48, and the q5_K-only
          // weight_qh_byte_offset@16 that flips cx.hasQh -> the [QH-MASK] helper.
          %b1 = weft_rvv.q4_k_nibble_unpack %vx, %vl block %super_block_index : index {kind = "q4_k_nibble_unpack", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 176 : i64, weight_qs_byte_offset = 48 : i64, weight_qh_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 2: 6-bit scale/min bit-dance -> utmp[4] scratch (Region B).
          %b2 = weft_rvv.q4_k_scale_min_bit_dance %vx, %vl block %super_block_index : index {kind = "q4_k_scale_min_bit_dance", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 176 : i64, weight_scales_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 3: per-sub-block uint6-scaled i32 dot + fold-back (Region C).
          %b3 = weft_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl block %super_block_index : index {kind = "q4_k_scaled_dot", integer_core_lmul = "m2", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 176 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 4: MIN term (sumf -= dmin * Σ(mins * bsums)) -- SCALAR sumf chain.
          %b4 = weft_rvv.q4_k_min_term %vx, %scales, %vy, %vl block %super_block_index : index {kind = "q4_k_min_term", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, bsums_byte_offset = 260 : i64, weight_dmin_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          // BRICK 6: deferred positive fold (sums += d * (float)aux32) -- 8-lane
          // fp32 VECTOR sums chain.
          %b6 = weft_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl block %super_block_index : index {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          weft_rvv.typed_super_block_block_dot_loop_yield %sums, %sumf : !weft_rvv.vector<f32, "m2">, f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// Parse + verify: the q5_K nibble-unpack brick carries the qh 5th-bit plane
// (weight_qh_byte_offset) on the block_q5_K format (stride 176, qs@48).
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_body
// VERIFY: weft_rvv.q4_k_nibble_unpack
// VERIFY-SAME: weight_qh_byte_offset = 16
// VERIFY: weft_rvv.typed_super_block_block_dot_loop_yield

// [QH-MASK] Region-A qh inject: qs strip load, the qh plane load, the low nibble
// (vand 0x0F), then the SINGLE qh bit isolated IN PLACE (vand 1<<h) + lifted to a
// per-lane bool (vmsne==0, SET lanes) and the +16 FUSED into ONE vadd_vx_u8m2_mu on
// the SET lanes, reinterpreted u8->i8. Byte-exact to the retired vsll<<4 | vadd_vv.
// EMIT: callee=qh_high_bit_plane
// EMIT: call_opaque "__riscv_vle8_v_u8m2"
// EMIT: literal "0x0F"
// EMIT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT: call_opaque "__riscv_vand_vx_u8m2"
// EMIT: call_opaque "__riscv_vmsne_vx_u8m2_b4"
// EMIT: literal "16"
// EMIT: call_opaque "__riscv_vadd_vx_u8m2_mu"
// EMIT: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"

// The OLD per-lane qh expand chain (vsll<<4 then vector-vector vadd of the 0/16
// contribution) is fully RETIRED by the SHARED [QH-MASK] native-mask helper -- the
// contribution vsll and the vadd_vv add do NOT survive anywhere in the q5_K unpack.
// (The vsrl_vx_u8m2 that extracts the HIGH nibble is a DIFFERENT op and remains.)
// RETIRED-NOT: call_opaque "__riscv_vadd_vv_u8m2"
// RETIRED-NOT: call_opaque "__riscv_vsll_vx_u8m2"
