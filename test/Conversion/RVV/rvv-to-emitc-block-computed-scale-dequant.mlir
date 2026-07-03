// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/computed_scale_sumi_dequant/plain_i8_sumi/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/scalar-i32-sumi-to-f32-computed-scale-f32/scalar-i32-sumi-to-f32-bad-relation/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADRELATION

// M-FLAT milestone brick 2/5 -- the per-block COMPUTED-scale i32-sumi dequant
// fold primitive (dismantling wall 2 of the four-walls flat-block-dot
// analysis: "tcrv_rvv.dequantize hard-requires an IMPORTED runtime ABI float
// scale, so it cannot consume a per-iteration COMPUTED scale"). The typed
// tcrv_rvv.block_computed_scale_dequant op takes the scalar i32 per-block sumi
// and the COMPUTED f32 per-block scale (the SSA output of brick 1's
// tcrv_rvv.block_fp16_scale_product), converts the sumi i32 -> f32 and
// multiplies into one f32 scalar `f32(sumi) * scale`. Its lowering is
// BYTE-IDENTICAL AT THE OPERATION-SPELLING LEVEL to the monolithic block-dot
// per-block fold: the SAME i32 -> float emitc.cast (the `(float)sumi` sitofp)
// + the SAME scalar float emitc.mul the monolith produces inline. This is an
// emit-consistency ("CORE == emission-plans") lit, NOT a numerical oracle:
// f32 fully covers the scalar i32 sumi and the f32 scale domains, so the fold
// is byte-exact by construction; bit-exact-vs-ggml is pending-hardware
// (ssh rvv). The op STOPS at the per-block term -- the cross-block fp32
// accumulate (`sumf += term`) is brick 3, not built here.

module {
  tcrv.exec.kernel @rvv_block_computed_scale_dequant_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_block_computed_scale_dequant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %sumi = tcrv_rvv.runtime_abi_value {c_name = "sumi", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant:sumi", role = "rhs-scalar-value"} : i32
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "block-computed-scale-dequant:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_block_computed_scale_dequant, sew = 16 : i64, source_kernel = "rvv_block_computed_scale_dequant_kernel", status = "selected-lowering-boundary"} {
        // brick 1: reconstruct the COMPUTED per-block scale d_x * d_y (f32).
        %scale = tcrv_rvv.block_fp16_scale_product %lhs, %rhs {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
        // brick 2: fold the scalar i32 sumi by the COMPUTED scale -> f32 term.
        %term = tcrv_rvv.block_computed_scale_dequant %sumi, %scale {kind = "computed_scale_sumi_dequant", dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32"} : i32, f32 -> f32
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_block_computed_scale_dequant_kernel_rvv_block_computed_scale_dequant(%[[VX:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %[[VY:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %[[SUMI:.*]]: !emitc.opaque<"int32_t">, %{{.*}}: !emitc.opaque<"size_t">)
// brick 1 output: the COMPUTED per-block scale d_x * d_y (one scalar float mul).
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VX]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VY]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[SCALE:.*]] = mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// brick 2 fold: (float)sumi * scale -- the SAME sitofp cast + scalar float mul
// the monolithic block-dot emitters produce inline for the per-block term.
// CHECK: %[[SUMIF:.*]] = cast %[[SUMI]] : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %[[SUMIF]], %[[SCALE]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: return

// The bounded surface is fail-closed on the dequant kind and relation facts (I7).
// BADKIND: currently supports only kind "computed_scale_sumi_dequant"
// BADRELATION: currently supports only dequant_relation "scalar-i32-sumi-to-f32-computed-scale-f32"
