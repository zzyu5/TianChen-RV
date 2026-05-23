// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_runtime_scalar_cmp_masked_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_runtime_scalar_cmp_masked_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body %lhs, %rhs_scalar, %src, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-store", op_kind = "runtime_scalar_cmp_masked_store", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_runtime_scalar_cmp_masked_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_runtime_scalar_cmp_masked_store
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED: %[[SRC:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: tcrv_rvv.masked_store %{{.*}}, %[[MASK]], %[[SRC]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-store"
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_store"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;src=source-input-buffer:src:abi|src-load|mstore-src|hdr;dst=output-buffer:dst:abi|mstore-base|mstore-dst|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-runtime-scalar-cmp-masked-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-source-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "masked-store-has-no-passthrough-load"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_runtime_scalar_cmp_masked_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_runtime_scalar_cmp_masked_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,src,dst,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;src=source-input-buffer:src:abi|src-load|mstore-src|hdr;dst=output-buffer:dst:abi|mstore-base|mstore-dst|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER: tianchenrv.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: void tcrv_emitc_pre_realized_body_runtime_scalar_cmp_masked_store_kernel_pre_realized_body_rvv_runtime_scalar_cmp_masked_store(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, int32_t *dst, size_t n);
