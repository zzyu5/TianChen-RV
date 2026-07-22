// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// Stage 3 single-scope grouped flip: the realized body is single-scope with no
// handoff, so re-running on it is no longer
// meaningful (that pass consumes the pre-realization handoff form). The consumer-scope
// resource facts it used to re-stamp now live on the lone realized with_vl scope.
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/c_name = "lower_bound"/s/role = "lower-bound-scalar-value"/role = "upper-bound-scalar-value"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-LOWER
// RUN: sed '/c_name = "lower_bound"/s/role = "lower-bound-scalar-value"/role = "upper-bound-scalar-value"/;/c_name = "upper_bound"/s/role = "upper-bound-scalar-value"/role = "lower-bound-scalar-value"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=SWAPPED-BOUNDS
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"/product_reduction_chain_relation = "metadata-product-reduction"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-REDUCTION
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=DTYPE-CHAIN
// RUN: sed 's/#weft_rvv.policy<tail = agnostic, mask = agnostic>/#weft_rvv.policy<tail = undisturbed, mask = agnostic>/g' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-POLICY
// Stage 3 single-scope grouped flip: the gearbox handoff/markers are deleted, so
// these probes mutate the surviving with_vl-carried facts (planning_contract,
// clamp_select_layout) instead of the deleted handoff/marker ops; confirmed to
// still fail-closed at emission-plan time with the specific resource-fact reason.
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/operand_encoding = "unpacked_i8", policy = /operand_encoding = "unpacked_i8", route_id = "rvv-i32m1", policy = /' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH
// Stage 3 single-scope grouped flip: the handoff op is retired, so the typed
// compute op list no longer carries it; injecting a stray handoff back into the
// list must fail-closed at manifest validation (the structural op list is checked).

// Pre-realized selected-body input for the bounded Stage 2 signed i8 product
// -> i16 product -> i32 reduction -> runtime-scale f32 dequantization
// -> runtime f32 lower/upper clamp/select -> f32 output store chain. The RVV
// plugin must realize the typed body before route planning; route ids,
// q-names, artifact names, and mirror metadata are not route authority.

module {
  weft.exec.kernel @pre_realized_body_product_reduce_dequant_clamp_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_product_reduce_dequant_clamp attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body %lhs, %rhs, %acc, %scale, %lower, %upper, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-clamped-dequantized-f32-vector-to-output-buffer", lower_predicate_kind = "slt", memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32", op_kind = "widening_product_reduce_dequant_clamp_f32", operand_encoding = "unpacked_i8", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", source_lmul = "mf4", source_sew = 8 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_product_reduce_dequant_clamp {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body
// Stage 3 single-scope grouped flip: the realized body is one weft_rvv.with_vl
// scope carrying a plain typed weft_rvv.widening_product head + the inline dequant
// chain + the f32 splat/compare/select clamp epilogue, with NO
// weft_rvv.gearbox_cross_region_handoff carrier and NO weft_rvv.vsetvl_region_marker
// placeholders. The structural unroll_factor (=2) the conversion reads is stamped on
// with_vl; the low_precision_resource.* facts below survive on the single scope (same
// selected Gearbox candidate source -> fact VALUES unchanged). Numerics HW-validated
// on ssh rvv (tolerance=1e-05).
// The structural unroll_factor (=2) the conversion reads is stamped on with_vl, and
// the typed body is a single product/reduce slice + inline dequant + f32 clamp + store.
// REALIZED-DAG: unroll_factor = 2 : i64
// REALIZED-DAG: weft_rvv.widening_product
// REALIZED-DAG: kind = "signed_widening_product"
// REALIZED-DAG: weft_rvv.standalone_reduce
// REALIZED-DAG: weft_rvv.dequantize
// REALIZED-DAG: weft_rvv.splat
// REALIZED-DAG: weft_rvv.compare
// REALIZED-DAG: weft_rvv.select
// REALIZED-DAG: weft_rvv.store
// The deleted two-scope carrier/markers are genuinely absent across the WHOLE
// realized body (the REALIZED RUN line's --implicit-check-not enforces this
// globally, fail-closed: a stray handoff or marker anywhere = incomplete flip).
// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body

// Stage 3 single-scope grouped flip: the gearbox handoff op is retired; the
// resource facts it used to carry now live on the lone with_vl scope (the RUN
// line's --implicit-check-not enforces the handoff's global absence).
// GEARBOX-CONSUME: weft_rvv.with_vl

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_dequant_clamp

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequant_clamp
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-LOWER: lower bound scalar
// MISSING-LOWER-SAME: lower-bound-scalar-value

// SWAPPED-BOUNDS: lower bound scalar
// SWAPPED-BOUNDS-SAME: lower-bound-scalar-value

// MISSING-REDUCTION: product_reduction_chain_relation
// MISSING-REDUCTION-SAME: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32

// DTYPE-CHAIN: requires typed product-reduction-dequant-clamp config

// UNSUPPORTED-POLICY: requires tail agnostic, mask agnostic policy

// MISSING-RESOURCE-FACTS: requires pass-produced low-precision direct-contraction resource fact

// STALE-RESOURCE-ACC: cannot consume stale or unsupported low-precision direct-contraction resource fact
// STALE-RESOURCE-ACC-SAME: expected 'i32'
// STALE-RESOURCE-ACC-SAME: found 'i16'

// STALE-RESOURCE-REDUCTION-LAYOUT: cannot consume stale or unsupported low-precision direct-contraction resource fact
// STALE-RESOURCE-REDUCTION-LAYOUT-SAME: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1
// STALE-RESOURCE-REDUCTION-LAYOUT-SAME: metadata-layout

// STALE-RESOURCE-MEMORY: low-precision direct-contraction resource selection
// STALE-RESOURCE-MEMORY-SAME: memory form
// STALE-RESOURCE-MEMORY-SAME: unit-stride-widening-product-reduce-dequant-clamp-f32
// STALE-RESOURCE-MEMORY-SAME: unit-stride-widening-product-reduce-dequantize-f32
// STALE-RESOURCE-MEMORY-SAME: selected candidate
// STALE-RESOURCE-MEMORY-SAME: product-reduction-dequant-clamp-f32

// STALE-RESOURCE-CLAMP-PHASE: low-precision direct-contraction resource selection requires clamp phase
// STALE-RESOURCE-CLAMP-PHASE-SAME: dequant-clamp-store
// STALE-RESOURCE-CLAMP-PHASE-SAME: metadata-clamp-store

// STALE-PLANNING: requires planning contract 'rvv-low-precision-production-resource-planning-contract.v1'
// STALE-PLANNING-SAME: but found 'metadata-derived-resource-planning-contract'

// STALE-REALIZATION-DECISION: requires realization decision 'consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1'
// STALE-REALIZATION-DECISION-SAME: but found 'metadata-only'

// STALE-RESOURCE-CLAMP-SELECT: requires clamp select layout 'clamp-lower-then-upper'
// STALE-RESOURCE-CLAMP-SELECT-SAME: but found 'metadata-select-layout'

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id













// STALE-DEQUANT: candidate metadata must carry exactly

// STALE-DEQUANT-SCALE: candidate metadata must carry exactly



// STALE-ARTIFACT-CLAMP-SELECT: metadata key '{{.*}}low_precision_resource.clamp_select_layout'{{.*}}'clamp-lower-then-upper' but was 'metadata-select-layout'
