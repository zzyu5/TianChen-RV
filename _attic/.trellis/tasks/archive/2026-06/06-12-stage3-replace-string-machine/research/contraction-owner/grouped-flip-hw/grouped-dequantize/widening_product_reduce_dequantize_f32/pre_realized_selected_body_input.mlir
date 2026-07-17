// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=tcrv_rvv.gearbox_cross_region_handoff --implicit-check-not=tcrv_rvv.vsetvl_region_marker
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | FileCheck %s --check-prefix=GEARBOX-SCHEDULE-PRIMITIVE
// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-RESOURCE-PASS
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.selected_candidate = "[^"]*"/tcrv_rvv.low_precision_resource.selected_candidate = "artifact-name-derived-resource-candidate"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"/tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i32m1_i32m1"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER-PRIMITIVE-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed 's/tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"/tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i32m1_i32m1"/' | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-SCHEDULE-PRIMITIVE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64/tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 1 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZATION-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '0,/tcrv_rvv.low_precision_resource.product_phase = "tail-product-reduce"/s//tcrv_rvv.low_precision_resource.product_phase = "artifact-metadata-region"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZATION-PRODUCT-PHASE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed '0,/tcrv_rvv.low_precision_resource.dequant_region_index = 3 : i64/s//tcrv_rvv.low_precision_resource.dequant_region_index = 1 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZATION-DEQUANT-REGION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64/s//tcrv_rvv.low_precision_resource.vector_register_budget = 3 : i64/' | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=PRUNED-RESOURCE-BUDGET
// Stage 3 single-scope grouped flip: the gearbox handoff/markers are deleted, so
// the resource facts they used to carry now live on the lone with_vl scope. These
// negative probes mutate the surviving with_vl-carried facts (not the deleted
// handoff/marker ops) and are confirmed to still fail-closed at emission-plan time.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.dequant_phase = "dequant-store"/tcrv_rvv.low_precision_resource.dequant_phase = "artifact-metadata-region"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT-PHASE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1"/tcrv_rvv.low_precision_resource.realization_decision = "artifact-name-derived-resource-decision"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZATION-DECISION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.planning_contract = "rvv-low-precision-production-resource-planning-contract.v1"/tcrv_rvv.low_precision_resource.planning_contract = "metadata-derived-resource-planning-contract"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PLANNING
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64/tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 99 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PEAK-LIVE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.unpack_intent = "none-direct-widening-product"/tcrv_rvv.low_precision_resource.unpack_intent = "metadata-only-unpack-plan"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-UNPACK-POLICY
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.operand_form\", value = \"unpacked-byte-elements\"/s//tcrv_rvv.low_precision_resource.operand_form\", value = \"packed-i4-nibbles\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PACKED-MIRROR
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic\", value = \"__riscv_vwredsum_vs_i16mf2_i32m1\"/s//tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic\", value = \"__riscv_vwredsum_vs_i32m1_i32m1\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRIMITIVE-RESOURCE-MIRROR
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1"/accumulator_carry_boundary = "metadata-carry-boundary"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-ACC
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=DTYPE-CHAIN
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /route_id = "rvv-i32m1", policy = /' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH

// Pre-realized selected-body input for the bounded Stage 2 signed i8 product
// -> i16 product -> i32 reduction -> runtime-scale f32 dequantization chain.
// The RVV plugin must materialize the realized tcrv_rvv body before route
// planning; route ids, mirrors, artifact names, and ABI strings are not
// realization authority.

module {
  tcrv.exec.kernel @pre_realized_body_product_reduce_dequantize_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// Stage 3 single-scope grouped flip: the realized body is one tcrv_rvv.with_vl
// scope carrying a plain typed tcrv_rvv.widening_product head + the inline dequant
// chain, with NO tcrv_rvv.gearbox_cross_region_handoff carrier and NO
// tcrv_rvv.vsetvl_region_marker placeholders. The structural unroll_factor (=2) the
// conversion reads is stamped on with_vl (the conversion expands the ONE typed
// product/reduce slice twice into the main loop + synthesizes the scalar tail loop);
// the low_precision_resource.* facts below survive on the single scope (same selected
// Gearbox candidate source -> fact VALUES unchanged). Numerics HW-validated on ssh
// rvv (tolerance=1e-05).
// REALIZED-DAG: selected_variant = @pre_realized_body_rvv_product_reduce_dequantize
// REALIZED-DAG: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.selected_candidate_index = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.candidate_count = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.legal_candidate_count = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.operand_form = "unpacked-byte-elements"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.packing_layout = "one-element-per-byte"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.unpack_intent = "none-direct-widening-product"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.storage_element_width = 8 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.effective_element_width = 8 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.dequant_phase = "dequant-store"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.dequant_region_index = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.product_phase = "tail-product-reduce"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.product_region_index = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.primitive_widening_product_intrinsic = "__riscv_vwmul_vv_i16mf2"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_unroll_factor = 2 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64
// REALIZED-DAG: tcrv_rvv.low_precision_resource.planning_contract = "rvv-low-precision-production-resource-planning-contract.v1"
// REALIZED-DAG: tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64
// REALIZED-DAG: tcrv_rvv.gearbox.producer_scope = "gearbox-scope:product-reduction"
// REALIZED-DAG: tcrv_rvv.gearbox.consumer_scope = "gearbox-scope:dequant-store"
// The structural unroll_factor (=2) the conversion reads is stamped on with_vl,
// and the typed body is a single product/reduce slice + inline dequant + store.
// REALIZED-DAG: unroll_factor = 2 : i64
// REALIZED-DAG: tcrv_rvv.widening_product
// REALIZED-DAG: kind = "signed_widening_product"
// REALIZED-DAG: tcrv_rvv.standalone_reduce
// REALIZED-DAG: tcrv_rvv.dequantize
// REALIZED-DAG: tcrv_rvv.store
// The deleted two-scope carrier/markers are genuinely absent across the WHOLE
// realized body (the REALIZED RUN line's --implicit-check-not enforces this
// globally, fail-closed: a stray handoff or marker anywhere = incomplete flip).
// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body

// GEARBOX-SCHEDULE-PRIMITIVE: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_chain_contract = "rvv-low-precision-widening-reduction-primitive-facts.v1"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_source_extension = "sign-extend-i8-to-i16-product"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_source_load = "unit-stride-byte-load"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.widening_product_extension_policy = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2"
// GEARBOX-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.widening_product_multiplicand_roles = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.dequantize"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float"}
// PLAN-SAME: {key = "tcrv_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.candidate_count", value = "3"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.legal_candidate_count", value = "3"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate_index", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.operand_form", value = "unpacked-byte-elements"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.source_signedness", value = "signed"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.storage_element_width", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.effective_element_width", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.packing_layout", value = "one-element-per-byte"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.unpack_intent", value = "none-direct-widening-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.memory_form", value = "unit-stride-widening-product-reduce-dequantize-f32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.producer_scope", value = "gearbox-scope:product-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.consumer_scope", value = "gearbox-scope:dequant-store"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_chain_contract", value = "rvv-low-precision-widening-reduction-primitive-facts.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.widening_product_multiplicand_roles", value = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.widening_product_extension_policy", value = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_source_extension", value = "sign-extend-i8-to-i16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic", value = "__riscv_vwredsum_vs_i16mf2_i32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_dequantize

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequantize
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,out,n
// HEADER: tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: tianchenrv.rvv.dequant_scale_role: dequant-scale-value
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]
// HEADER: tianchenrv.rvv.low_precision_resource.candidate_count: 3
// HEADER: tianchenrv.rvv.low_precision_resource.legal_candidate_count: 3
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate_index: 2
// HEADER: tianchenrv.rvv.low_precision_resource.operand_form: unpacked-byte-elements
// HEADER: tianchenrv.rvv.low_precision_resource.source_signedness: signed
// HEADER: tianchenrv.rvv.low_precision_resource.storage_element_width: 8
// HEADER: tianchenrv.rvv.low_precision_resource.effective_element_width: 8
// HEADER: tianchenrv.rvv.low_precision_resource.packing_layout: one-element-per-byte
// HEADER: tianchenrv.rvv.low_precision_resource.unpack_intent: none-direct-widening-product
// HEADER: tianchenrv.rvv.low_precision_resource.memory_form: unit-stride-widening-product-reduce-dequantize-f32
// HEADER: tianchenrv.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: tianchenrv.rvv.low_precision_resource.primitive_chain_contract: rvv-low-precision-widening-reduction-primitive-facts.v1
// HEADER: tianchenrv.rvv.low_precision_resource.widening_product_multiplicand_roles: lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4
// HEADER: tianchenrv.rvv.low_precision_resource.widening_product_extension_policy: source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2
// HEADER: tianchenrv.rvv.low_precision_resource.primitive_source_load: unit-stride-byte-load
// HEADER: tianchenrv.rvv.low_precision_resource.primitive_source_extension: sign-extend-i8-to-i16-product
// HEADER: tianchenrv.rvv.low_precision_resource.primitive_reduction_intrinsic: __riscv_vwredsum_vs_i16mf2_i32m1
// HEADER: tianchenrv.rvv.gearbox_producer_scope: gearbox-scope:product-reduction
// HEADER: tianchenrv.rvv.gearbox_consumer_scope: gearbox-scope:dequant-store
// HEADER: tianchenrv.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// MISSING-RESOURCE-PASS: requires pass-produced low-precision direct-contraction resource fact
// MISSING-RESOURCE-PASS-SAME: tcrv_rvv.low_precision_resource.candidate_set

// STALE-PROVIDER-RESOURCE: low-precision direct-contraction resource selection requires a selected product-reduction candidate
// STALE-PROVIDER-RESOURCE-SAME: artifact-name-derived-resource-candidate

// STALE-PROVIDER-PRIMITIVE-RESOURCE: low-precision direct-contraction resource selection requires primitive reduction intrinsic
// STALE-PROVIDER-PRIMITIVE-RESOURCE-SAME: __riscv_vwredsum_vs_i16mf2_i32m1
// STALE-PROVIDER-PRIMITIVE-RESOURCE-SAME: __riscv_vwredsum_vs_i32m1_i32m1

// STALE-SCHEDULE-PRIMITIVE: selected-body realization
// STALE-SCHEDULE-PRIMITIVE-SAME: low-precision direct-contraction resource fact
// STALE-SCHEDULE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic
// STALE-SCHEDULE-PRIMITIVE-SAME: __riscv_vwredsum_vs_i16mf2_i32m1
// STALE-SCHEDULE-PRIMITIVE-SAME: __riscv_vwredsum_vs_i32m1_i32m1

// STALE-REALIZATION-RESOURCE: low-precision direct-contraction resource selection requires realized vsetvl region count 3
// STALE-REALIZATION-RESOURCE-SAME: but found 1

// STALE-REALIZATION-PRODUCT-PHASE: low-precision direct-contraction resource selection requires product phase 'tail-product-reduce'
// STALE-REALIZATION-PRODUCT-PHASE-SAME: artifact-metadata-region

// STALE-REALIZATION-DEQUANT-REGION: low-precision direct-contraction resource selection requires dequant region index 3
// STALE-REALIZATION-DEQUANT-REGION-SAME: but found 1

// PRUNED-RESOURCE-BUDGET: requires at least two legal provider-built low-precision resource candidates

// STALE-DEQUANT-PHASE: requires dequant phase 'dequant-store'
// STALE-DEQUANT-PHASE-SAME: but found 'artifact-metadata-region'

// STALE-REALIZATION-DECISION: requires realization decision 'consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1'
// STALE-REALIZATION-DECISION-SAME: but found 'artifact-name-derived-resource-decision'

// STALE-PLANNING: requires planning contract 'rvv-low-precision-production-resource-planning-contract.v1'
// STALE-PLANNING-SAME: but found 'metadata-derived-resource-planning-contract'

// STALE-PEAK-LIVE: requires realized peak live vector-group estimate 7
// STALE-PEAK-LIVE-SAME: but found 99

// STALE-UNPACK-POLICY: requires unpack intent 'none-direct-widening-product'
// STALE-UNPACK-POLICY-SAME: but found 'metadata-only-unpack-plan'

// STALE-PACKED-MIRROR: metadata key '{{.*}}low_precision_resource.operand_form'{{.*}}'unpacked-byte-elements' but was 'packed-i4-nibbles'

// STALE-PRIMITIVE-RESOURCE-MIRROR: metadata key '{{.*}}low_precision_resource.primitive_reduction_intrinsic'{{.*}}'__riscv_vwredsum_vs_i16mf2_i32m1' but was '__riscv_vwredsum_vs_i32m1_i32m1'

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-ACC: accumulator_carry_boundary
// MISSING-ACC-SAME: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1

// DTYPE-CHAIN: requires typed product-reduction-dequantization config

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id
