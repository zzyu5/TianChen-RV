// RUN: not tcrv-opt %s --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-RESOURCE-PASS
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed 's/tcrv_rvv.low_precision_resource.selected_candidate = "[^"]*"/tcrv_rvv.low_precision_resource.selected_candidate = "artifact-name-derived-resource-candidate"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER-RESOURCE
// RUN: sed '/gearbox_cross_region_handoff/d;s/tcrv_rvv.dequantize %handoff/tcrv_rvv.dequantize %reduced/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-HANDOFF
// RUN: sed 's/tcrv_rvv.dequantize %handoff/tcrv_rvv.dequantize %reduced/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-HANDOFF-CONSUMER
// RUN: sed 's/producer_scope = "gearbox-scope:product-reduction"/producer_scope = "gearbox-scope:artifact-name"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=STALE-GEARBOX-SCOPE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"/s//tcrv_rvv.dequant_scale_role", value = "output-buffer"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"/s//tcrv_rvv.product_reduction_chain_relation", value = "route-string-derived-product-reduction"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"/s//tcrv_rvv.dequantization_relation", value = "script-derived-dequant"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed 's/{key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "[^"]*"}/{key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "artifact-name-derived-resource-candidate"}/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,out,n"/s//tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed 's/!tcrv_rvv.vector<f32, "m1">/!tcrv_rvv.vector<i32, "m1">/g;/c_name = "out"/s/c_type = "float \*"/c_type = "int32_t *"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=DTYPE-MISMATCH
// RUN: sed '/c_name = "acc"/s/c_type = "const int32_t \*"/c_type = "int32_t *"/;/c_name = "acc"/s/role = "accumulator-input-buffer"/role = "output-buffer"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-ACC
// RUN: sed 's/lmul = "m1"/lmul = "m2"/g' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-CONFIG

// Explicit selected-body input for one bounded Stage 2 signed low-precision
// contraction-to-dequant chain. The typed tcrv_rvv body carries i8 source
// loads, an i16 product, an i16-to-i32 standalone reduction boundary, a
// structural Gearbox cross-region handoff carrying runtime AVL/VL facts, and
// a nested consumer tcrv_rvv.with_vl region carrying runtime scale, f32
// dequantization, and the f32 store boundary.

module {
  tcrv.exec.kernel @explicit_selected_body_product_reduce_dequantize_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_product_reduce_dequantize, sew = 32 : i64, source_kernel = "explicit_selected_body_product_reduce_dequantize_kernel", status = "selected-lowering-boundary", tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1", tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1", tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64, tcrv_rvv.low_precision_resource.realized_unroll_factor = 2 : i64, tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64} {
        tcrv_rvv.vsetvl_region_marker %vl {phase = "grouped-product-reduce-main", region_count = 3 : i64, region_index = 1 : i64, resource_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1"} : !tcrv_rvv.vl
        tcrv_rvv.vsetvl_region_marker %vl {phase = "tail-product-reduce", region_count = 3 : i64, region_index = 2 : i64, resource_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1"} : !tcrv_rvv.vl
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %reduced = tcrv_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %handoff = tcrv_rvv.gearbox_cross_region_handoff %reduced, %vl, %n {consumer_scope = "gearbox-scope:dequant-store", contract = "gearbox-product-reduce-to-dequant-cross-region-handoff.v1", from_phase = "tail-product-reduce", primitive_accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", primitive_chain_contract = "rvv-low-precision-widening-reduction-primitive-facts.v1", primitive_chain_kind = "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1", primitive_product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1", primitive_reduction_store_vl = "1", primitive_result_layout = "store-standalone-reduction-lane0-to-output-scalar", primitive_scalar_seed_splat_intrinsic = "__riscv_vmv_v_x_i32m1", primitive_source_signedness = "signed", primitive_widening_product_intrinsic = "__riscv_vwmul_vv_i16mf2", primitive_widening_product_relation = "signed-i8mf4xi8mf4-to-i16mf2", producer_scope = "gearbox-scope:product-reduction", region_count = 3 : i64, resource_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1", runtime_avl_source = "runtime_abi:n", to_phase = "dequant-store"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl, index -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_product_reduce_dequantize, sew = 32 : i64, source_kernel = "explicit_selected_body_product_reduce_dequantize_kernel", status = "selected-lowering-boundary", tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1", tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1", tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64, tcrv_rvv.low_precision_resource.realized_unroll_factor = 2 : i64, tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64} {
          tcrv_rvv.vsetvl_region_marker %vl {phase = "dequant-store", region_count = 3 : i64, region_index = 3 : i64, resource_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1"} : !tcrv_rvv.vl
          %dequantized = tcrv_rvv.dequantize %handoff, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
          tcrv_rvv.store %out, %dequantized, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
        } : !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "explicit-selected-body-widening-product-reduce-dequantize-f32-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float"}
// PLAN-SAME: {key = "tcrv_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.product_emul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.accumulator_emul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.peak_live_vector_groups", value = "7"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.producer_scope", value = "gearbox-scope:product-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.consumer_scope", value = "gearbox-scope:dequant-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_product_reduce_dequantize

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_product_reduce_dequantize
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequantize-f32-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,out,n
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.source_sew: 8
// HEADER: tianchenrv.rvv.source_lmul: mf4
// HEADER: tianchenrv.rvv.product_sew: 16
// HEADER: tianchenrv.rvv.product_lmul: mf2
// HEADER: tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: tianchenrv.rvv.dequant_scale_role: dequant-scale-value
// HEADER: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]
// HEADER: tianchenrv.rvv.low_precision_resource.product_emul: mf2
// HEADER: tianchenrv.rvv.low_precision_resource.accumulator_emul: m1
// HEADER: tianchenrv.rvv.low_precision_resource.peak_live_vector_groups: 7
// HEADER: tianchenrv.rvv.low_precision_resource.vector_register_budget: 32
// HEADER: tianchenrv.rvv.gearbox_producer_scope: gearbox-scope:product-reduction
// HEADER: tianchenrv.rvv.gearbox_consumer_scope: gearbox-scope:dequant-store
// HEADER: tianchenrv.rvv.accumulator_sew: 32
// HEADER: tianchenrv.rvv.result_sew: 32
// HEADER: tianchenrv.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_dequantize_f32.v1
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat:float-e32m1,scale:float
// HEADER: void tcrv_emitc_explicit_selected_body_product_reduce_dequantize_kernel_explicit_selected_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// MISSING-RESOURCE-PASS: requires pass-produced low-precision direct-contraction resource fact
// MISSING-RESOURCE-PASS-SAME: tcrv_rvv.low_precision_resource.candidate_set

// STALE-PROVIDER-RESOURCE: low-precision direct-contraction resource selection requires a selected product-reduction candidate
// STALE-PROVIDER-RESOURCE-SAME: artifact-name-derived-resource-candidate

// MISSING-HANDOFF: requires source-producing product-reduction chain to be in the same tcrv_rvv.with_vl body as tcrv_rvv.dequantize

// STALE-HANDOFF-CONSUMER: requires a preceding tail-product-reduce tcrv_rvv.vsetvl_region_marker in the producer scope
// STALE-HANDOFF-CONSUMER-SAME: handoff-consuming dequant/store chain in the consumer tcrv_rvv.with_vl scope

// STALE-GEARBOX-SCOPE: requires producer_scope 'gearbox-scope:product-reduction'

// STALE-SCALE: target artifact candidate validation failed
// STALE-SCALE-SAME: dequant

// STALE-PRODUCT: target artifact candidate validation failed
// STALE-PRODUCT-SAME: product-reduction

// STALE-DEQUANT: target artifact candidate validation failed
// STALE-DEQUANT-SAME: dequantization

// STALE-RESOURCE: target artifact candidate validation failed
// STALE-RESOURCE-SAME: low_precision_resource.selected_candidate
// STALE-RESOURCE-SAME: provider-selected low-precision direct-contraction resource selected candidate

// STALE-ABI: target artifact candidate validation failed
// STALE-ABI-SAME: runtime_abi_order

// MISSING-SCALE: requires runtime scale operand
// MISSING-SCALE-SAME: dequant-scale-value

// DTYPE-MISMATCH: requires result vector type to be !tcrv_rvv.vector<f32, "m1">

// MISSING-ACC: requires accumulator seed operand
// MISSING-ACC-SAME: accumulator-input-buffer

// UNSUPPORTED-CONFIG: requires result element width 8 to agree with enclosing tcrv_rvv.with_vl SEW32 metadata
