// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// unit-stride memory movement slice. The RVV plugin must realize compare
// lhs/rhs, active source, and old destination ABI operands into explicit
// compare loads/old-destination passthrough load/compare/masked_load/store
// typed structure. The mask is produced
// inside the selected RVV body, not supplied by runtime ABI or inferred from
// route names.

module {
  tcrv.exec.kernel @pre_realized_body_computed_masked_unit_load_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_computed_masked_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-memory:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-memory:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-memory:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-memory:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-memory:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_memory_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %dst, %n {inactive_lane_policy = "preserve-old-destination", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-store", op_kind = "computed_masked_unit_load_store", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_computed_masked_unit_load_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-memory-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-memory-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_masked_unit_load_store
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[OLD_DST:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !tcrv_rvv.mask<i32, "m1">
// REALIZED: %[[LOADED:.*]] = tcrv_rvv.masked_load %{{.*}}, %[[MASK]], %[[OLD_DST]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-load"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.indexed_store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_unit_load_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_load"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-unit-load-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_unit_load_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_unit_load_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi-mirror|cmp-lhs-load|compare-lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi-mirror|cmp-rhs-load|compare-rhs-call;src=source-input-buffer:src:abi-mirror|materialized-masked-load-base|masked-load-source-call;dst=output-buffer:dst:abi-mirror|old-dst-load|masked-load-passthrough-call|materialized-store-base|header-mirror;n=runtime-element-count:n:abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.mask_tail_policy_owner", value = "computed-mask memory mask/tail policy"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-load-store"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-compare-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_masked_unit_load_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_computed_masked_unit_load_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-unit-load-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,dst,n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-unit-load-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-unit-load-store-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_unit_load_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_unit_load_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi-mirror|cmp-lhs-load|compare-lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi-mirror|cmp-rhs-load|compare-rhs-call;src=source-input-buffer:src:abi-mirror|materialized-masked-load-base|masked-load-source-call;dst=output-buffer:dst:abi-mirror|old-dst-load|masked-load-passthrough-call|materialized-store-base|header-mirror;n=runtime-element-count:n:abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: tianchenrv.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1
// HEADER: tianchenrv.rvv.mask_tail_policy_owner: computed-mask memory mask/tail policy
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-load-store
// HEADER: void tcrv_emitc_pre_realized_body_computed_masked_unit_load_store_kernel_pre_realized_body_rvv_computed_masked_unit_load_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *dst, size_t n);
