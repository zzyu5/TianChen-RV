// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/cross_block_f32_scalar_accumulate/plain_f32_accumulate/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/strict-ascending-block-carried/unordered-block-fold/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADORDER
// RUN: sed 's/f32, f32 -> f32/f32, f32 -> i32/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADRESULT

// M-FLAT milestone brick 3/5 -- the cross-block f32 scalar ACCUMULATE fold
// primitive (dismantling wall 3 of the four-walls flat-block-dot analysis:
// "the cross-block fold `sumf += (float)sumi * (d_x*d_y)` is a raw emitc.add,
// and the two typed accumulators the dialect already carries --
// weft_rvv.widening_accumulate / weft_rvv.deferred_accumulate -- are i32
// INTRA-STRIP accumulators, wrong dtype AND wrong scope"). The typed
// weft_rvv.cross_block_f32_accumulate op takes the block-carried f32
// accumulator and one f32 per-block term (the SSA output of brick 2's
// weft_rvv.block_computed_scale_dequant) and folds `acc + term` into one f32
// scalar. Its lowering is BYTE-IDENTICAL AT THE OPERATION-SPELLING LEVEL to
// the monolithic block-dot cross-block fold: the SAME scalar float emitc.add
// the monolith produces inline for `sumf + <block term>`. This is an
// emit-consistency ("CORE == emission-plans") lit, NOT a numerical oracle:
// f32 fully covers the accumulator and term domains, so the fold is byte-exact
// by construction; bit-exact-vs-ggml is pending-hardware (ssh rvv). The op
// STOPS at the fold -- the nb block loop and the final scalar store are brick
// 4 / brick 5, not built here.
//
// This fixture exercises a two-block cross-block fold: two per-block terms
// (term0, term1) are folded acc=term0, term=term1 -> sumf = term0 + term1, in
// STRICT ASCENDING block order (the fp non-associativity ggml preserves).

module {
  weft.exec.kernel @rvv_cross_block_f32_accumulate_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_cross_block_f32_accumulate attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "cross-block-f32-accumulate:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "cross-block-f32-accumulate:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %sumi0 = weft_rvv.runtime_abi_value {c_name = "sumi0", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "cross-block-f32-accumulate:sumi0", role = "rhs-scalar-value"} : i32
      %sumi1 = weft_rvv.runtime_abi_value {c_name = "sumi1", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "cross-block-f32-accumulate:sumi1", role = "rhs-secondary-scalar-value"} : i32
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "cross-block-f32-accumulate:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
        // brick 1: reconstruct the COMPUTED per-block scale d_x * d_y (f32).
        %scale = weft_rvv.block_fp16_scale_product %lhs, %rhs {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
        // brick 2: fold each block's scalar i32 sumi by the COMPUTED scale -> f32 term.
        %term0 = weft_rvv.block_computed_scale_dequant %sumi0, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
        %term1 = weft_rvv.block_computed_scale_dequant %sumi1, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
        // brick 3: cross-block fp32 accumulate sumf = term0 + term1 (strict ascending block order).
        %sumf = weft_rvv.cross_block_f32_accumulate %term0, %term1 {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_cross_block_f32_accumulate_kernel_rvv_cross_block_f32_accumulate(%[[VX:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %[[VY:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %[[SUMI0:.*]]: !emitc.opaque<"int32_t">, %[[SUMI1:.*]]: !emitc.opaque<"int32_t">, %{{.*}}: !emitc.opaque<"size_t">)
// brick 1 output: the COMPUTED per-block scale d_x * d_y (one scalar float mul).
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VX]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VY]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[SCALE:.*]] = mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// brick 2 terms: (float)sumi * scale for each block.
// CHECK: %[[SUMIF0:.*]] = cast %[[SUMI0]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TERM0:.*]] = mul %[[SUMIF0]], %[[SCALE]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[SUMIF1:.*]] = cast %[[SUMI1]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TERM1:.*]] = mul %[[SUMIF1]], %[[SCALE]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// brick 3 fold: sumf = term0 + term1 -- the SAME scalar float emitc.add the
// monolithic block-dot emitters produce inline for the cross-block `sumf + term`.
// CHECK: add %[[TERM0]], %[[TERM1]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: return

// The bounded surface is fail-closed on the accumulate kind and order facts (I7).
// BADKIND: currently supports only kind "cross_block_f32_scalar_accumulate"
// BADORDER: currently supports only accumulate_order "strict-ascending-block-carried"
// Wall-3 dtype contrast: an i32 result (an intra-strip integer accumulator
// form) is rejected fail-closed -- this is the scalar f32 cross-block fold, not
// the i32 intra-strip widening/deferred accumulate (the whole reason the op
// exists requires f32 dtype + scope).
// BADRESULT: requires an f32 scalar result
