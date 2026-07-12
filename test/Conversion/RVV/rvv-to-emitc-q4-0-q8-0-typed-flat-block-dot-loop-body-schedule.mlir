// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// The mbf==1 default combo of the SAME typed body still lowers (strip robust, no unroll).
// RUN: sed 's/multi_block_factor = 2 : i64, //' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MBF1
// The multi_block_factor==4 + elided combo of the SAME typed body lowers to the
// 4-way unroll with the elided single-cover cores + robust tail.
// RUN: sed 's/multi_block_factor = 2 : i64/multi_block_factor = 4 : i64/; s/strip_elision = "robust"/strip_elision = "elided"/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MBF4

// M-FLAT schedule-parametrization step 1: the q4_0 (left_assoc) typed flat
// block-dot LOOP body materialized across the FULL legal {multi_block_factor,
// strip_elision} cross product at the m1 anchor, driven by the loop-body knobs
// but sourced OP-BY-OP from the region ops. This combo is
// {integer_core_lmul=m1, multi_block_factor=2, strip_elision=robust}: the outer
// block loop steps by 2, emitting TWO independent per-block integer cores (each
// keeping its VLEN-robust inner strip loop with the sumi-carry seed), THEN the
// two left-associative fp32 folds in STRICT ascending block order (the fp
// non-associativity boundary), plus an `nb % 2` robust single-block scalar tail.
// Byte-identical to the monolithic q4_0 GgmlBlockDotQ40Op at the SAME knobs
// (verified out-of-tree by a comment/ABI-normalized diff of the two lowerings);
// the integer core stays region-driven (anti-bypass W4). Numerical
// bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q4_0_q8_0_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %3 = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %4 = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %5 = weft_rvv.setvl %0 {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q4_0_q8_0_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %2, %3, %1, %0 attributes {activation_block_stride = 34 : i64, fold_model = "left_assoc", integer_core_lmul = "m1", kind = "typed_flat_block_dot_loop_body", multi_block_factor = 2 : i64, qk = 32 : i64, strip_elision = "robust", weight_block_stride = 18 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %6 = weft_rvv.block_fp16_scale_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 18 : i64, rhs_block_stride = 34 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %7 = weft_rvv.load %2, %5 block %arg0 : index {block_stride = 18 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %8 = weft_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %9 = weft_rvv.load %3, %5 block %arg0 : index {block_stride = 34 : i64, quant_byte_offset = 18 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %10 = weft_rvv.packed_i4_offset_binary_x_i8_product %7, %8, %9, %5 {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4m1-x-i8m1x2-to-i16m2"} : !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m2">
          %11 = weft_rvv.standalone_reduce %10, %4, %5 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          %12 = weft_rvv.typed_vector_lane0_to_scalar_extract %11, %5 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
          %13 = weft_rvv.block_computed_scale_dequant %12, %6 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %14 = weft_rvv.cross_block_f32_accumulate %arg1, %13 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %14 : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(
// The function-scoped fp32 accumulator + block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The multi-block main loop bound nb_main = nb - nb % 2, then the by-2 outer loop.
// CHECK: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// CHECK: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// --- block 0 core: address arithmetic, scales, robust integer strip loop ---
// CHECK: %[[SUMI0:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- block 1 core at ib + 1: a SECOND independent robust core, emitted BEFORE
// --- either fold (latency overlap) ---
// CHECK: add %[[IB]], %{{.*}}
// CHECK: %[[SUMI1:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- THEN the two folds in STRICT ascending block order ---
// CHECK: %[[ACC0:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: assign %[[ACC0]] : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: %[[ACC1:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: assign %[[ACC1]] : !emitc.opaque<"float"> to %[[SUMF]]
// --- the nb % 2 robust single-block scalar tail loop ---
// CHECK: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// CHECK: %[[SUMIT:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The *s scalar store through the output pointer.
// CHECK: subscript
// CHECK: return

// The multi_block_factor==1 (no-unroll) robust form: a single by-1 block loop
// with a robust strip core + one fold + the scalar store (no main/tail split).
// MBF1: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(
// MBF1: for %{{.*}} = %{{.*}} to %{{.*}} step
// MBF1: for %{{.*}} = %{{.*}} to %{{.*}} step
// MBF1: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// MBF1: %[[M1ACC:.*]] = expression : !emitc.opaque<"float"> {
// MBF1: return

// The multi_block_factor==4 + elided combo: the by-4 main loop emits four elided
// single-cover cores (no inner strip loop) then four ascending folds, plus the
// robust nb % 4 tail. The four vwredsum calls mark the four unrolled cores.
// MBF4: for %[[IB4:.*]] = %{{.*}} to %{{.*}} step
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// MBF4: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// MBF4: expression : !emitc.opaque<"float"> {
// MBF4: return
