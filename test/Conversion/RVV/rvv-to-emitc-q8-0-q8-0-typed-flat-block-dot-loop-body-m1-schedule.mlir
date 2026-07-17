// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --implicit-check-not=i8m2 --implicit-check-not=i16m4

// M-FLAT schedule-parametrization step 1b: the q8_0 (sumi_times_scales) typed flat
// block-dot LOOP body materialized at the m1 integer-core anchor -- the
// constructible LMUL档 sibling of the m2 anchor pinned in
// rvv-to-emitc-q8-0-q8-0-typed-flat-block-dot-loop-body-schedule.mlir. q8_0's
// native core is the WHOLE-block plain-i8 signed widening product; at m1 the byte
// anchor is i8m1 x i8m1 -> i16m2 (one EMUL rung NARROWER than the m2 anchor's
// i8m2 -> i16m4), reduced into the shared i32m1 lane. This combo is
// {integer_core_lmul=m1, multi_block_factor=1 (absent), strip_elision=robust}: a
// single by-1 block loop with the VLEN-robust inner strip loop (m1 VLMAX=16 at
// VLEN128 re-strips the qk=32 block; the {m1,elided} single-cover form is
// VLEN>=256-only, so it is INTENTIONALLY omitted here) and the PINNED
// SeparatedLeftAssoc oracle fold ((sumi*d_x)*d_y). The m1 arithmetic is
// bit-identical to the m2 anchor (LMUL is a schedule knob, not an arithmetic one),
// so m1's byte-exact-vs-ggml is inherited by construction from the m2 anchor's
// k1-hardware seal. Numerical bit-exact-vs-ggml is pending-hardware. The
// --implicit-check-not=i8m2/i16m4 locks that NO m2-core / m4-wide width leaks
// (the emit must follow the m1 region types, never the hardcoded m2 default).

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q8_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %zero_seed = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q8_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m1", strip_elision = "robust"} {
        ^bb0(%block_index: index, %acc: f32):
          %dd = weft_rvv.block_fp16_scale_product %vx, %vy block %block_index : index {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64, rhs_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %wv = weft_rvv.load %vx, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %av = weft_rvv.load %vy, %vl block %block_index : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %prod = weft_rvv.widening_product %wv, %av, %vl {kind = "signed_widening_product", product_relation = "signed-i8m1xi8m1-to-i16m2"} : !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m2">
          %red = weft_rvv.standalone_reduce %prod, %zero_seed, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
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
// The function-scoped fp32 accumulator + block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The multi_block_factor==1 (no-unroll) robust form: a single by-1 block loop with a
// robust whole-block i8m1 strip core (m1 VLMAX re-strips qk=32), then ONE fold + the
// scalar store (no main/tail split).
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[SUMI0:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// --- the VLEN-robust inner strip loop over the qk=32 block at the m1 byte anchor ---
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- THEN the single fold, the PINNED SeparatedLeftAssoc oracle
// --- [measurement/浮点折叠oracle.md §1]: SEPARATE cast/mul/mul/add emitc
// --- statements, NOT a fused emitc.expression, so clang cannot contract
// --- (t*d_y)+sumf into fmaf; ((sumi*d_x)*d_y) with NO d_x*d_y premultiply. ---
// CHECK: %[[C0:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[T0:.*]] = mul %[[C0]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[T0B:.*]] = mul %[[T0]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[ACC0:.*]] = add %{{.*}}, %[[T0B]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: assign %[[ACC0]] : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript
// CHECK: return
