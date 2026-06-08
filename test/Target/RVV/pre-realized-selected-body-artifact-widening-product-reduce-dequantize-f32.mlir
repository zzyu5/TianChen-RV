// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-RESOURCE-PASS
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.selected_candidate = "[^"]*"/tcrv_rvv.low_precision_resource.selected_candidate = "artifact-name-derived-resource-candidate"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64/tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 1 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZATION-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64/s//tcrv_rvv.low_precision_resource.vector_register_budget = 3 : i64/' | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=PRUNED-RESOURCE-BUDGET
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/phase = "dequant-store"/phase = "artifact-metadata-region"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-REALIZED-REGION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/tcrv_rvv.dequantize %[^,]*/tcrv_rvv.dequantize %10/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-HANDOFF-CONSUMER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | sed 's/consumer_scope = "gearbox-scope:dequant-store"/consumer_scope = "gearbox-scope:product-reduction"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-GEARBOX-SCOPE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/accumulator_carry_boundary = "scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1"/accumulator_carry_boundary = "metadata-carry-boundary"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-ACC
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
      tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
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
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_product_reduce_dequantize
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 4 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_unroll_factor = 1 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u1]"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64
// REALIZED: tcrv_rvv.vsetvl_region_marker %[[VL]]
// REALIZED-SAME: phase = "load-product-reduce"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 1 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[PRODUCT:.*]] = tcrv_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "signed_widening_product"
// REALIZED-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
// REALIZED-SAME: -> !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.standalone_reduce %[[PRODUCT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "signed_widening_reduce_add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[HANDOFF:.*]] = tcrv_rvv.gearbox_cross_region_handoff %[[REDUCED]], %[[VL]], %{{[^ ]+}}
// REALIZED-SAME: consumer_scope = "gearbox-scope:dequant-store"
// REALIZED-SAME: contract = "gearbox-product-reduce-to-dequant-cross-region-handoff.v1"
// REALIZED-SAME: from_phase = "load-product-reduce"
// REALIZED-SAME: producer_scope = "gearbox-scope:product-reduction"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
// REALIZED-SAME: runtime_avl_source = "runtime_abi:n"
// REALIZED-SAME: to_phase = "dequant-store"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_product_reduce_dequantize
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 4 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_unroll_factor = 1 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u1]"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64
// REALIZED: tcrv_rvv.vsetvl_region_marker %[[VL]]
// REALIZED-SAME: phase = "dequant-store"
// REALIZED-SAME: region_count = 2 : i64
// REALIZED-SAME: region_index = 2 : i64
// REALIZED-SAME: resource_decision = "consume-low-precision-u1-two-vsetvl-region-budget-4of32.v1"
// REALIZED: %[[DEQUANT:.*]] = tcrv_rvv.dequantize %[[HANDOFF]], %{{.*}}, %[[VL]]
// REALIZED-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
// REALIZED-SAME: kind = "i32_to_f32_scaled"
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: tcrv_rvv.store %{{.*}}, %[[DEQUANT]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float"}
// PLAN-SAME: {key = "tcrv_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u1]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.memory_form", value = "unit-stride-widening-product-reduce-dequantize-f32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.producer_scope", value = "gearbox-scope:product-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.consumer_scope", value = "gearbox-scope:dequant-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_dequantize

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequantize
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,out,n
// HEADER: tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: tianchenrv.rvv.dequant_scale_role: dequant-scale-value
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u1]
// HEADER: tianchenrv.rvv.low_precision_resource.memory_form: unit-stride-widening-product-reduce-dequantize-f32
// HEADER: tianchenrv.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: tianchenrv.rvv.gearbox_producer_scope: gearbox-scope:product-reduction
// HEADER: tianchenrv.rvv.gearbox_consumer_scope: gearbox-scope:dequant-store
// HEADER: tianchenrv.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// MISSING-RESOURCE-PASS: requires pass-produced low-precision direct-contraction resource fact
// MISSING-RESOURCE-PASS-SAME: tcrv_rvv.low_precision_resource.candidate_set

// STALE-PROVIDER-RESOURCE: low-precision direct-contraction resource selection requires selected candidate
// STALE-PROVIDER-RESOURCE-SAME: artifact-name-derived-resource-candidate

// STALE-REALIZATION-RESOURCE: selected-body realization low-precision direct-contraction resource fact
// STALE-REALIZATION-RESOURCE-SAME: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count
// STALE-REALIZATION-RESOURCE-SAME: requires realized vsetvl region count 2

// PRUNED-RESOURCE-BUDGET: pruned every low-precision direct-contraction resource candidate
// PRUNED-RESOURCE-BUDGET-SAME: peak-live-vector-groups-exceed-vector-register-budget
// PRUNED-RESOURCE-BUDGET-SAME: vector register budget 3

// STALE-REALIZED-REGION: 'tcrv_rvv.gearbox_cross_region_handoff' op requires to_phase 'dequant-store'

// STALE-HANDOFF-CONSUMER: requires a preceding load-product-reduce tcrv_rvv.vsetvl_region_marker in the producer scope
// STALE-HANDOFF-CONSUMER-SAME: handoff-consuming dequant/store chain in the consumer tcrv_rvv.with_vl scope

// STALE-GEARBOX-SCOPE: requires consumer_scope 'gearbox-scope:dequant-store'

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-ACC: accumulator_carry_boundary
// MISSING-ACC-SAME: scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1

// DTYPE-CHAIN: requires typed product-reduction-dequantization config

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id
