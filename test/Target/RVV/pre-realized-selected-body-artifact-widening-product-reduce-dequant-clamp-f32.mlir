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
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-contraction-family-plan-validated/s//provider_supported_mirror:rvv-artifact-name-authority/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n/s//weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,lower_bound,scale,upper_bound,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr/s//lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// Stage 3 single-scope grouped flip: the handoff op is retired, so the typed
// compute op list no longer carries it; injecting a stray handoff back into the
// list must fail-closed at manifest validation (the structural op list is checked).
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '/rvv_selected_body_typed_compute_op/s/weft_rvv.widening_product+/weft_rvv.widening_product+weft_rvv.gearbox_cross_region_handoff+/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HANDOFF
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"/s//weft_rvv.required_header_declarations", value = "stddef.h,stdint.h"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/dequant-splat\/clamped:float-e32m1/s//converted\/scaled\/clamped:float-e32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"/s//weft_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-product-only-leaf-profile.v1"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LEAF
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"/s//weft_rvv.reduction_accumulator_layout", value = "metadata-accumulator-layout"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ACC-LAYOUT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"/s//weft_rvv.reduction_result_layout", value = "store-vector-result-from-metadata"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESULT-LAYOUT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.product_vector_c_type", value = "vint16mf2_t"/s//weft_rvv.product_vector_c_type", value = "vint32m1_t"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCT-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/__riscv_vwmul_vv_i16mf2/s//__riscv_vwmul_vv_i32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-WPROD
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/__riscv_vwredsum_vs_i16mf2_i32m1/s//__riscv_vredsum_vs_i32m1_i32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-WRED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}/s//weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}, {key = "weft_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}/s//weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}, {key = "weft_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT-SCALE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.lower_bound_c_type", value = "float"/s//weft_rvv.lower_bound_c_type", value = "double"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOWER-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"/s//weft_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m2"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BOUND-SPLAT

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
// REALIZED-DAG: selected_variant = @pre_realized_body_rvv_product_reduce_dequant_clamp
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
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_dequant_clamp_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product+weft_rvv.standalone_reduce+weft_rvv.dequantize+weft_rvv.compare+weft_rvv.select"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-widening-product-reduce-dequant-clamp-f32"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1;lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;out=output-buffer:out:abi|cdeq|store|f32m1|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-contraction-family-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "weft_rvv.product_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.product_vector_type", value = "!weft_rvv.vector<i16, \22mf2\22>"}
// PLAN-SAME: {key = "weft_rvv.product_vector_c_type", value = "vint16mf2_t"}
// PLAN-SAME: {key = "weft_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "weft_rvv.widening_product_relation", value = "signed-i8mf4xi8mf4-to-i16mf2"}
// PLAN-SAME: {key = "weft_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "weft_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i16mf2"}
// PLAN-SAME: {key = "weft_rvv.widening_reduction_intrinsic", value = "__riscv_vwredsum_vs_i16mf2_i32m1"}
// PLAN-SAME: {key = "weft_rvv.scalar_seed_splat_intrinsic", value = "__riscv_vmv_v_x_i32m1"}
// PLAN-SAME: {key = "weft_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: {key = "weft_rvv.scalar_result_runtime_boundary", value = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1"}
// PLAN-SAME: {key = "weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: {key = "weft_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_role", value = "lower-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_role", value = "upper-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.bound_order", value = "lower-bound-before-upper-bound"}
// PLAN-SAME: {key = "weft_rvv.clamp_relation", value = "signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1"}
// PLAN-SAME: {key = "weft_rvv.select_layout", value = "clamp-lower-then-upper"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.compare_intrinsic", value = "__riscv_vmflt_vv_f32m1_b32"}
// PLAN-SAME: {key = "weft_rvv.secondary_compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.secondary_compare_intrinsic", value = "__riscv_vmflt_vv_f32m1_b32"}
// PLAN-SAME: {key = "weft_rvv.masked_merge_intrinsic", value = "__riscv_vmerge_vvm_f32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_dequant_clamp

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequant_clamp
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1;lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;out=output-buffer:out:abi|cdeq|store|f32m1|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER-DAG: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER-DAG: weft.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float
// HEADER-DAG: weft.rvv.source_sew: 8
// HEADER-DAG: weft.rvv.source_lmul: mf4
// HEADER-DAG: weft.rvv.product_sew: 16
// HEADER-DAG: weft.rvv.product_lmul: mf2
// HEADER-DAG: weft.rvv.product_vector_type: !weft_rvv.vector<i16, "mf2">
// HEADER-DAG: weft.rvv.product_vector_c_type: vint16mf2_t
// HEADER-DAG: weft.rvv.accumulator_sew: 32
// HEADER-DAG: weft.rvv.accumulator_lmul: m1
// HEADER-DAG: weft.rvv.result_sew: 32
// HEADER-DAG: weft.rvv.result_lmul: m1
// HEADER-DAG: weft.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: weft.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: weft.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input
// HEADER-DAG: weft.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar
// HEADER-DAG: weft.rvv.widening_product_relation: signed-i8mf4xi8mf4-to-i16mf2
// HEADER-DAG: weft.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32
// HEADER-DAG: weft.rvv.widening_product_intrinsic: __riscv_vwmul_vv_i16mf2
// HEADER-DAG: weft.rvv.widening_reduction_intrinsic: __riscv_vwredsum_vs_i16mf2_i32m1
// HEADER-DAG: weft.rvv.scalar_seed_splat_intrinsic: __riscv_vmv_v_x_i32m1
// HEADER-DAG: weft.rvv.reduction_store_vl: 1
// HEADER-DAG: weft.rvv.standalone_reduction_scalar_result_runtime_boundary: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1
// HEADER-DAG: weft.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER-DAG: weft.rvv.dequant_scale_role: dequant-scale-value
// HEADER-DAG: weft.rvv.dequant_scale_c_type: float
// HEADER-DAG: weft.rvv.dequant_scale_name: scale
// HEADER-DAG: weft.rvv.lower_bound_role: lower-bound-scalar-value
// HEADER-DAG: weft.rvv.upper_bound_role: upper-bound-scalar-value
// HEADER-DAG: weft.rvv.lower_bound_c_type: float
// HEADER-DAG: weft.rvv.upper_bound_c_type: float
// HEADER-DAG: weft.rvv.bound_order: lower-bound-before-upper-bound
// HEADER-DAG: weft.rvv.clamp_relation: signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1
// HEADER-DAG: weft.rvv.select_layout: clamp-lower-then-upper
// HEADER-DAG: weft.rvv.compare_predicate_kind: slt
// HEADER-DAG: weft.rvv.secondary_compare_predicate_kind: slt
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

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: provider_supported_mirror
// STALE-PROVIDER-SAME: rvv-artifact-name-authority

// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: weft_rvv.runtime_abi_order
// STALE-ABI-SAME: lhs,rhs,acc,lower_bound,scale,upper_bound,out,n

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: weft_rvv.route_operand_binding_operands
// STALE-BINDING-SAME: must mirror
// STALE-BINDING-SAME: lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel

// STALE-HANDOFF: manifest invalid
// STALE-HANDOFF-SAME: rvv_selected_body_typed_compute_op
// STALE-HANDOFF-SAME: weft_rvv.widening_product+weft_rvv.standalone_reduce+weft_rvv.dequantize+weft_rvv.compare+weft_rvv.select

// STALE-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-HEADER: weft_rvv.required_header_declarations
// STALE-HEADER-SAME: must mirror
// STALE-HEADER-SAME: stddef.h,stdint.h

// STALE-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-CTYPE: weft_rvv.c_type_mapping
// STALE-CTYPE-SAME: must mirror
// STALE-CTYPE-SAME: converted/scaled/clamped:float-e32m1

// STALE-LEAF: RVV materialized EmitC target artifact bridge failed
// STALE-LEAF: weft_rvv.target_leaf_profile
// STALE-LEAF-SAME: product-only-leaf-profile

// STALE-ACC-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-ACC-LAYOUT: weft_rvv.reduction_accumulator_layout
// STALE-ACC-LAYOUT-SAME: metadata-accumulator-layout

// STALE-RESULT-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-RESULT-LAYOUT: weft_rvv.reduction_result_layout
// STALE-RESULT-LAYOUT-SAME: store-vector-result-from-metadata

// STALE-PRODUCT-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-PRODUCT-CTYPE: weft_rvv.product_vector_c_type
// STALE-PRODUCT-CTYPE-SAME: vint32m1_t

// STALE-WPROD: RVV materialized EmitC target artifact bridge failed
// STALE-WPROD: widening_product_intrinsic
// STALE-WPROD-SAME: __riscv_vwmul_vv_i16mf2
// STALE-WPROD-SAME: __riscv_vwmul_vv_i32m1

// STALE-WRED: RVV materialized EmitC target artifact bridge failed
// STALE-WRED: reduction_intrinsic
// STALE-WRED-SAME: __riscv_vwredsum_vs_i16mf2_i32m1
// STALE-WRED-SAME: __riscv_vredsum_vs_i32m1_i32m1

// STALE-DEQUANT: candidate metadata must carry exactly

// STALE-DEQUANT-SCALE: candidate metadata must carry exactly

// STALE-LOWER-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-LOWER-CTYPE: weft_rvv.lower_bound_c_type
// STALE-LOWER-CTYPE-SAME: double

// STALE-BOUND-SPLAT: RVV materialized EmitC target artifact bridge failed
// STALE-BOUND-SPLAT: weft_rvv.rhs_broadcast_intrinsic
// STALE-BOUND-SPLAT-SAME: __riscv_vfmv_v_f_f32m2

// STALE-ARTIFACT-CLAMP-SELECT: metadata key '{{.*}}low_precision_resource.clamp_select_layout'{{.*}}'clamp-lower-then-upper' but was 'metadata-select-layout'
