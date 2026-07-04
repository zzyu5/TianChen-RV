// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

// M-FLAT verifier fail-closed (anti attribute-derived-emission lie): the
// tcrv_rvv.typed_flat_block_dot_loop_body lowering derives the integer-core
// LMUL from the region's per-block vector loads (the packed/widening product op
// 硬钉s that width via its product_relation), NOT from integer_core_lmul. An
// integer_core_lmul that the region cannot express would be SILENTLY ignored --
// emit ships the region width while the attr claims another (lying IR). The
// verifier now rejects the mismatch fail-closed.

// This is the byte-exact q4_0 (left_assoc) full body whose region integer core
// is硬钉ed to i8m1 by "offset-binary-i4m1-x-i8m1x2-to-i16m2", so the ONLY
// honorable integer_core_lmul is "m1". Here the attr LIES "m2" -> reject.
module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_q4_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %5 = tcrv_rvv.setvl %0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q4_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{integer_core_lmul "m2" does not match the region integer-core load LMUL "m1"}}
        tcrv_rvv.typed_flat_block_dot_loop_body %2, %3, %1, %0 attributes {activation_block_stride = 34 : i64, fold_model = "left_assoc", integer_core_lmul = "m2", kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, strip_elision = "robust", weight_block_stride = 18 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %6 = tcrv_rvv.block_fp16_scale_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 18 : i64, rhs_block_stride = 34 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          %7 = tcrv_rvv.load %2, %5 block %arg0 : index {block_stride = 18 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %8 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %9 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 18 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %10 = tcrv_rvv.packed_i4_offset_binary_x_i8_product %7, %8, %9, %5 {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4m1-x-i8m1x2-to-i16m2"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
          %11 = tcrv_rvv.standalone_reduce %10, %4, %5 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %12 = tcrv_rvv.typed_vector_lane0_to_scalar_extract %11, %5 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          %13 = tcrv_rvv.block_computed_scale_dequant %12, %6 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %14 = tcrv_rvv.cross_block_f32_accumulate %arg1, %13 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %14 : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
