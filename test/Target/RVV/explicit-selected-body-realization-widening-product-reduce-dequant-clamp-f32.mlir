// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_body/s/op_kind = "widening_product_reduce_dequant_clamp_f32"/op_kind = "metadata_route"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-OP
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_body/s/product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"/product_reduction_chain_relation = "metadata-product-reduction"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-REDUCTION
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=DTYPE-CHAIN
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/c_name = "lower_bound"/s/role = "lower-bound-scalar-value"/role = "upper-bound-scalar-value"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-LOWER
// RUN: sed 's/#weft_rvv.policy<tail = agnostic, mask = agnostic>/#weft_rvv.policy<tail = undisturbed, mask = agnostic>/g' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-POLICY
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_body/s/operand_encoding = "unpacked_i8", policy = /operand_encoding = "unpacked_i8", route_id = "rvv-i32m1", policy = /' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH
// Stage 3 single-scope grouped flip: the handoff op is retired from the typed
// compute op list; injecting a stray handoff back into the list must fail-closed
// at manifest validation (the structural op list is checked).

// Explicit selected-body input for the bounded Stage 2 signed i8 product ->
// i16 product -> i32 reduction -> runtime-scale f32 dequantization -> runtime
// f32 lower/upper clamp/select -> f32 output store chain. The explicit
// compound body is the selected typed body that the RVV plugin must realize
// before provider route planning.

module {
  weft.exec.kernel @explicit_wprdc_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_rvv_wprdc attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_body %lhs, %rhs, %acc, %scale, %lower, %upper, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-clamped-dequantized-f32-vector-to-output-buffer", lower_predicate_kind = "slt", memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32", op_kind = "widening_product_reduce_dequant_clamp_f32", operand_encoding = "unpacked_i8", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", source_lmul = "mf4", source_sew = 8 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @explicit_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_rvv_wprdc {origin = "rvv-plugin", policy = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32-case"}
      weft.exec.fallback @explicit_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-product-reduce-dequant-clamp-f32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_body
// Stage 3 single-scope grouped flip: the realized body is one weft_rvv.with_vl
// scope carrying a plain typed weft_rvv.widening_product head + the inline dequant
// chain + the f32 splat/compare/select clamp epilogue, with NO
// weft_rvv.gearbox_cross_region_handoff carrier and NO weft_rvv.vsetvl_region_marker
// placeholders. The structural unroll_factor (=2) the conversion reads is stamped on
// with_vl; the resource facts survive on the single scope.
// REALIZED-DAG: selected_variant = @explicit_rvv_wprdc
// REALIZED-DAG: unroll_factor = 2 : i64
// REALIZED-DAG: %[[PRODUCT:.*]] = weft_rvv.widening_product %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}}
// REALIZED-DAG: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
// REALIZED-DAG: weft_rvv.standalone_reduce
// REALIZED-DAG: kind = "signed_widening_reduce_add"
// REALIZED-DAG: weft_rvv.dequantize
// REALIZED-DAG: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
// REALIZED-DAG: weft_rvv.splat
// REALIZED-DAG: weft_rvv.compare
// REALIZED-DAG: weft_rvv.select
// REALIZED-DAG: weft_rvv.store
// The deleted two-scope carrier/markers are genuinely absent across the WHOLE
// realized body (the REALIZED RUN line's --implicit-check-not enforces this globally).
// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: target = @explicit_rvv_wprdc

// HEADER: weft.rvv.selected_variant: @explicit_rvv_wprdc
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_wprdc_kernel_explicit_rvv_wprdc(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

// MISSING-OP: currently supports only op_kind
// MISSING-OP-SAME: widening_product_reduce_dequant_clamp_f32

// MISSING-REDUCTION: product_reduction_chain_relation
// MISSING-REDUCTION-SAME: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32

// DTYPE-CHAIN: requires typed product-reduction-dequant-clamp config

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-LOWER: lower bound scalar
// MISSING-LOWER-SAME: lower-bound-scalar-value

// UNSUPPORTED-POLICY: requires tail agnostic, mask agnostic policy

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id




