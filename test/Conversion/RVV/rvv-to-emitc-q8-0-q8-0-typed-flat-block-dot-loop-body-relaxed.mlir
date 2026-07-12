// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --implicit-check-not=vfredosum
// The SAME typed body with numerics_tier REMOVED (absent = strict, fail-closed)
// falls back to the strict deferred-ordered schedule: the §1 byte-exact fold
// (vfmul x2 + per-batch ORDERED vfredosum.vs), with NO vfmacc / vfredusum / acc_vec
// anywhere. This locks that numerics_tier is the knob that toggles the numeric
// oracle and that the strict default is byte-unchanged (zero-regression).
// RUN: sed 's/, numerics_tier = "relaxed"//' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=STRICT --implicit-check-not=vfmacc --implicit-check-not=vfredusum --implicit-check-not=acc_vec
// A garbage numerics_tier is rejected fail-closed by the op verifier (I7).
// RUN: sed 's/numerics_tier = "relaxed"/numerics_tier = "reassoc"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADTIER

// [GAP-NUM] the q8_0 (sumi_times_scales) typed flat block-dot LOOP body with
// fold_structure = "deferred-ordered" AND numerics_tier = "relaxed" -- the §5
// policy-gated REASSOCIATION variant of the deferred-ordered schedule
// (testing/flat-block-dot-fp-fold-oracle.md §5). numerics_tier is ORTHOGONAL to
// fold_model (the arithmetic tree) and fold_structure (how the STRICT fold is
// issued): it selects WHICH numeric oracle governs the fp fold. "relaxed" is
// admitted ONLY behind the numerics.reassoc_ok (kind=policy) capability fact
// (fail-closed: absent => strict) and is verified against the reassoc-tolerant
// oracle + a DECLARED ULP UPPER BOUND, NEVER the §1 byte-exact oracle, and NEVER a
// headline ([K-5] fp gate = ULP-bound declaration).
//
// PHASE A is UNCHANGED from the strict deferred body (the four region-sourced
// integer cores packed by vslide1down + the vlse16 strided scale gather). PHASE B
// is the relaxed delta: instead of the §1 {vfmul x2, per-batch vfredosum.vs}, it
// (1) PREMULTIPLIES the two scales into d_xy with ONE vfmul (the §1-FORBIDDEN
// d_x*d_y premultiply, which §5 ALLOWS), (2) fuses sumi*d_xy into a PERSISTENT
// B-lane fp32 accumulator (acc_vec) with ONE vfmacc.vv (a single fused rounding
// replacing three §1 ops), and (3) does NO per-batch reduction -- the whole
// cross-block collapse is DEFERRED to ONE unordered vfredusum.vs after the loop.
// That is the K-way accumulator of §5: batches are independent (the serial fold
// chain is broken), a latency win that is pending-hardware. The nb % 4 tail keeps
// the strict scalar §1 fold; the final vfredusum, SEEDED by the running sumf,
// folds the vector accumulator onto it.

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q8_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %zero_seed = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q8_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided", multi_block_factor = 4 : i64, fold_structure = "deferred-ordered", numerics_tier = "relaxed"} {
        ^bb0(%block_index: index, %acc: f32):
          %dd = weft_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %wv = weft_rvv.load %vx, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %av = weft_rvv.load %vy, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
          %prod = weft_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
          %red = weft_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          %sumi = weft_rvv.typed_vector_lane0_to_scalar_extract %red, %vl {kind = "vector_lane0_to_scalar_i32_extract", extract_relation = "i32m1-lane0-to-scalar-i32"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
          %bterm = weft_rvv.block_computed_scale_dequant %sumi, %dd {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
          %acc_next = weft_rvv.cross_block_f32_accumulate %acc, %bterm {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// The function-scoped fp32 sumf accumulator.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// [GAP-NUM] the PERSISTENT B-lane fp32 vector accumulator (acc_vec), seeded to 0.0
// ONCE before the main loop -- the K-way accumulator that breaks the serial fold
// chain. It has NO analogue in the strict tier.
// CHECK: %[[ACCVEC:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vfloat32m1_t">>
// CHECK: %[[ZEROVEC:.*]] = call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: assign %[[ZEROVEC]] : !emitc.opaque<"vfloat32m1_t"> to %[[ACCVEC]]
// The by-4 batch main loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// --- PHASE A (UNCHANGED from strict): the four region-sourced integer cores
// --- packed into one i32m1 by four vslide1down (the final lane == block, ascending)
// --- + the two strided fp16 scale gathers widened to f32. ---
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: %[[SLIDE0:.*]] = call_opaque "__riscv_vslide1down_vx_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vslide1down_vx_i32m1"(%[[SLIDE0]]
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vslide1down_vx_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: %[[SUMIVEC:.*]] = call_opaque "__riscv_vslide1down_vx_i32m1"
// CHECK: call_opaque "__riscv_vlse16_v_f16mf2"
// CHECK: %[[DXVEC:.*]] = call_opaque "__riscv_vfwcvt_f_f_v_f32m1"
// CHECK: call_opaque "__riscv_vlse16_v_f16mf2"
// CHECK: %[[DYVEC:.*]] = call_opaque "__riscv_vfwcvt_f_f_v_f32m1"
// --- PHASE B (RELAXED §5): sumi int32->f32, then ONE vfmul PREMULTIPLYING the two
// --- scales (d_x*d_y -- FORBIDDEN in §1, ALLOWED in §5), then ONE fused vfmacc
// --- accumulating sumi*d_xy into acc_vec. NO per-batch vfredosum. ---
// CHECK: %[[SUMIF:.*]] = call_opaque "__riscv_vfcvt_f_x_v_f32m1"(%[[SUMIVEC]]
// CHECK: %[[DXY:.*]] = call_opaque "__riscv_vfmul_vv_f32m1"(%[[DXVEC]], %[[DYVEC]]
// CHECK: %[[ACCCUR:.*]] = load %[[ACCVEC]] : <!emitc.opaque<"vfloat32m1_t">>
// CHECK: %[[ACCNEXT:.*]] = call_opaque "__riscv_vfmacc_vv_f32m1"(%[[ACCCUR]], %[[SUMIF]], %[[DXY]]
// CHECK: assign %[[ACCNEXT]] : !emitc.opaque<"vfloat32m1_t"> to %[[ACCVEC]]
// --- the nb % 4 strict scalar tail: the SAME §1 SeparatedLeftAssoc fold
// --- (cast/mul/mul/add) into sumf -- the remainder is not reassociated. ---
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: %[[TC:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TT:.*]] = mul %[[TC]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[TT2:.*]] = mul %[[TT]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: add %{{.*}}, %[[TT2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// --- [GAP-NUM] the DEFERRED final collapse: ONE unordered vfredusum.vs folds the
// --- B lane accumulators into sumf (seeded by the running sumf). vfredusum is a
// --- TREE reduction (lane order NOT preserved) -- the reassociation §5 permits
// --- and §1 forbids. Exactly one, AFTER both loops. ---
// CHECK: %[[SUMFCUR:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vfmv_v_f_f32m1"(%[[SUMFCUR]]
// CHECK: %[[ACCFINAL:.*]] = load %[[ACCVEC]] : <!emitc.opaque<"vfloat32m1_t">>
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vfredusum_vs_f32m1_f32m1"(%[[ACCFINAL]], %[[SEED]]
// CHECK: %[[OUT:.*]] = call_opaque "__riscv_vfmv_f_s_f32m1_f32"(%[[RED]]
// CHECK: assign %[[OUT]] : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: subscript
// CHECK: return

// The strict tier (numerics_tier absent = strict): the byte-exact §1 deferred
// fold -- vfmul x2 + per-batch ORDERED vfredosum.vs, seeded by sumf. NO vfmacc, NO
// vfredusum, NO acc_vec (implicit-check-not above). This is the paper-headline
// numeric contract.
// STRICT: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// STRICT: %[[SSEED:.*]] = call_opaque "__riscv_vfmv_v_f_f32m1"
// STRICT: %[[SSUMIF:.*]] = call_opaque "__riscv_vfcvt_f_x_v_f32m1"
// STRICT: %[[ST0:.*]] = call_opaque "__riscv_vfmul_vv_f32m1"(%[[SSUMIF]]
// STRICT: %[[ST1:.*]] = call_opaque "__riscv_vfmul_vv_f32m1"(%[[ST0]]
// STRICT: call_opaque "__riscv_vfredosum_vs_f32m1_f32m1"(%[[ST1]], %[[SSEED]]
// STRICT: return

// BADTIER: only accepts numerics_tier "strict" or "relaxed"; got "reassoc"
