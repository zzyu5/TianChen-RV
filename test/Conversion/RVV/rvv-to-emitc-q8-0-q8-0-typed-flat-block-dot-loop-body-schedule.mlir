// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// The mbf==1 default combo of the SAME typed body still lowers (strip robust, no unroll).
// RUN: sed 's/multi_block_factor = 2 : i64/multi_block_factor = 1 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MBF1
// The multi_block_factor==4 + elided combo of the SAME typed body lowers to the
// 4-way unroll with the elided single-cover cores + robust tail.
// RUN: sed 's/multi_block_factor = 2 : i64/multi_block_factor = 4 : i64/; s/strip_elision = "robust"/strip_elision = "elided"/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MBF4

// M-FLAT schedule-parametrization step 2: the q8_0 (sumi_times_scales) typed flat
// block-dot LOOP body materialized across the FULL legal {multi_block_factor,
// strip_elision} cross product at the m2 anchor, driven by the loop-body knobs
// but sourced OP-BY-OP from the region ops (ported from the q4_0 scaffold). q8_0's
// native core is the WHOLE-block plain-i8 signed widening product (m2 native,
// blockLen = qk = 32, two i8m2 loads, direct vwmul into i16m4) -- one LMUL step UP
// from the q4_0 sibling's half-block packed-i4 core. This combo is
// {integer_core_lmul=m2, multi_block_factor=2, strip_elision=robust}: the outer
// block loop steps by 2, emitting TWO independent per-block integer cores (each
// keeping its VLEN-robust inner strip loop with the sumi-carry seed), THEN the two
// fp32 folds in STRICT ascending block order (the fp non-associativity boundary),
// plus an `nb % 2` robust single-block scalar tail. Byte-identical to the
// monolithic q8_0 GgmlBlockDotQ80Q80Op at the SAME knobs (verified out-of-tree by
// a provenance/ABI-normalized diff of the two lowerings across all six combos);
// the integer core stays region-driven (anti-bypass W4). Numerical
// bit-exact-vs-ggml is pending-hardware.

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
        weft_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "robust", multi_block_factor = 2 : i64, fold_structure = "per-block", numerics_tier = "strict"} {
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
// The function-scoped fp32 accumulator + block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The multi-block main loop bound nb_main = nb - nb % 2, then the by-2 outer loop.
// CHECK: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// CHECK: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// --- block 0 core: address arithmetic, scales, robust whole-block i8m2 strip ---
// CHECK: %[[SUMI0:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK-NOT: call_opaque "__riscv_vwmacc_vv_i16
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// --- block 1 core at ib + 1: a SECOND independent robust core, emitted BEFORE
// --- either fold (latency overlap) ---
// CHECK: add %[[IB]], %{{.*}}
// CHECK: %[[SUMI1:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// --- THEN the two folds in STRICT ascending block order, each the PINNED
// --- SeparatedLeftAssoc oracle [measurement/浮点折叠oracle.md §1]:
// --- SEPARATE cast/mul/mul/add emitc statements, NOT a fused emitc.expression,
// --- so clang cannot contract (t*d_y)+sumf into fmaf; ((sumi*d_x)*d_y) with NO
// --- d_x*d_y premultiply. The assign takes the ADD result directly (in a fused
// --- expression it would take the expression result -- this locks separation).
// CHECK: %[[C0:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[T0:.*]] = mul %[[C0]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[T0B:.*]] = mul %[[T0]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[ACC0:.*]] = add %{{.*}}, %[[T0B]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: assign %[[ACC0]] : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: %[[C1:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[T1:.*]] = mul %[[C1]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[T1B:.*]] = mul %[[T1]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[ACC1:.*]] = add %{{.*}}, %[[T1B]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: assign %[[ACC1]] : !emitc.opaque<"float"> to %[[SUMF]]
// --- the nb % 2 robust single-block scalar tail loop ---
// CHECK: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// CHECK: %[[SUMIT:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// The *s scalar store through the output pointer.
// CHECK: subscript
// CHECK: return

// The multi_block_factor==1 (no-unroll) robust form: a single by-1 block loop
// with a robust whole-block i8m2 strip core + one fold + the scalar store (no
// main/tail split).
// MBF1: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
// MBF1: for %{{.*}} = %{{.*}} to %{{.*}} step
// MBF1: for %{{.*}} = %{{.*}} to %{{.*}} step
// MBF1: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// The single fold: the separated SeparatedLeftAssoc oracle (no fused expression).
// MBF1: %[[M1C:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// MBF1: %[[M1T:.*]] = mul %[[M1C]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF1: %[[M1T2:.*]] = mul %[[M1T]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF1: %[[M1ACC:.*]] = add %{{.*}}, %[[M1T2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF1: assign %[[M1ACC]] : !emitc.opaque<"float"> to
// MBF1: return

// The multi_block_factor==4 + elided combo: the by-4 main loop emits four elided
// single-cover cores (no inner strip loop) then four ascending folds, plus the
// robust nb % 4 tail. The four vwredsum calls mark the four unrolled cores.
// MBF4: for %[[IB4:.*]] = %{{.*}} to %{{.*}} step
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// The four folds are the separated SeparatedLeftAssoc oracle (no fused expression);
// check the first fold's cast/mul/mul/add chain.
// MBF4: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// MBF4: %[[F4T:.*]] = mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF4: %[[F4T2:.*]] = mul %[[F4T]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF4: add %{{.*}}, %[[F4T2]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// MBF4: return
