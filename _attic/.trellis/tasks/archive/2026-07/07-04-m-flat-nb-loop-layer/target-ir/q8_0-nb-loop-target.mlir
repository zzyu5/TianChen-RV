// ============================================================================
// TARGET IR (曳光弹 / tracer bullet) — THE SINGLE MAIN LINE.
// ============================================================================
// The minimal q8_0 nb-loop chain we are building the dialect's loop layer to
// carry. This is the END-STATE we want: a typed_flat_block_dot_loop_body whose
// region holds the FULL per-block typed primitive chain (no monolith helper,
// no direct-sumi shortcut). Missing capability is held by an EXPLICIT placeholder
// (W2: the per-block-addressed load — the existing tcrv_rvv.load has no block
// operand). Ugly is allowed.
//
// Advance this file, in order:  parse -> verify -> lower -> byte-exact(vs scalar
// oracle).  Tear whichever of the four walls blocks it FIRST (order back-derived
// live, not pre-written):
//   W1  scope-result-exit  : the reduce/extract i32m1 (SEW32) domain living
//                            under the loop's outer with_vl (SEW8/m2), and the
//                            scalar sumi reaching brick 2 without crossing a
//                            result-less with_vl. (nesting relax direct->ancestor
//                            already landed; this file tests whether that suffices.)
//   W2  per-block addressing: load from base + ib*stride (+quant_byte_offset).
//   W3  reduction pinning    : standalone_reduce's producer/seed constraints
//                            (getDefiningOp<WideningProductOp> / RuntimeABIValueOp).
//   W4  lowering-walks-chain  : emitTypedFlatBlockDotLoopBody must LOWER each
//                            typed op in the region (not re-emit via the helper).
//
// q8_0: QK=32, block stride 34 (=2 fp16 scale bytes + 32 i8), plain i8 (NO nibble
// decode), fold = sumf + (float)sumi*(d_x*d_y), integer core e8m2.
// ============================================================================

module {
  tcrv.exec.kernel @q8_0_nb_loop_target_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q8_0_nb_loop_target attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s  = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n  = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      // The per-block reduce seed (0) — standalone_reduce takes a runtime ABI
      // scalar seed. Each block reduces fresh; cross-block fold is brick 3 (f32).
      %zero_seed = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q8_0_nb_loop_target, sew = 8 : i64, source_kernel = "q8_0_nb_loop_target_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%ib: index, %acc: f32):
          // --- fp16 scale rebuild (brick 1, reused as chain primitive) ---
          %scale = tcrv_rvv.block_fp16_scale_product %vx, %vy block %ib : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          // --- W2 per-block i8 load x2 (base + ib*stride + quant_byte_offset) ---
          // PLACEHOLDER: intended extended form of tcrv_rvv.load carrying the
          // block induction var. The existing load has no `block` operand, so
          // this is expected to block at PARSE -> tear W2 to the minimal form.
          %wv = tcrv_rvv.load %vx, %vl block %ib : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          %av = tcrv_rvv.load %vy, %vl block %ib : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
          // --- vector dot: widening product (signed i8m2 x i8m2 -> i16m4) ---
          %prod = tcrv_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
          // --- reduce i16m4 -> i32m1 lane0 (W3: producer/seed pinning) ---
          %red = tcrv_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          // --- scalar out-of-domain: extract lane0 -> scalar i32 sumi (W1) ---
          %sumi = tcrv_rvv.typed_vector_lane0_to_scalar_extract %red, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          // --- brick 2: (float)sumi * scale ---
          %bterm = tcrv_rvv.block_computed_scale_dequant %sumi, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
          // --- brick 3: cross-block f32 accumulate (strict ascending) ---
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %bterm {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
