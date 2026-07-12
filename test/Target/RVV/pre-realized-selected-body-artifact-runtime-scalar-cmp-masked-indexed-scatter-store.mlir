// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 runtime-scalar
// computed-mask indexed scatter-store slice. The RVV plugin must realize the
// runtime scalar compare producer, payload load, index load, and masked indexed
// store before provider-owned EmitC route construction.

module {
  weft.exec.kernel @pre_realized_body_rt_scalar_cmidx_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_rt_scalar_cmidx_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_indexed_scatter_pre_realized_body %lhs, %rhs_scalar, %src, %index, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-indexed-scatter-store", offset_unit = "element", op_kind = "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_rt_scalar_cmidx_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-indexed-scatter-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_indexed_scatter_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_rt_scalar_cmidx_store
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[PAYLOAD:.*]] = weft_rvv.load
// REALIZED: %[[INDICES:.*]] = weft_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: weft_rvv.masked_indexed_store %{{.*}}, %[[INDICES]], %[[MASK]], %[[PAYLOAD]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: index_uniqueness = "unique"
// REALIZED-SAME: memory_form = "masked-indexed-store"
// REALIZED-NOT: weft_rvv.masked_indexed_load
// REALIZED-NOT: weft_rvv.mask_load

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_indexed_store"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,index,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_scatter_store_unit_load.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-scatter-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-scatter-store-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/source:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,dst:runtime-scalar-masked-indexed-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-source-indexed-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.indexed_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-source-indexed-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.indexed_write_side_contract", value = "source-before-active-indexed-write;destination-before-inactive-tail-preserve"}
// PLAN-SAME: {key = "weft_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_rt_scalar_cmidx_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_rt_scalar_cmidx_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-indexed-scatter-store-unit-load-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,src,index,dst,n
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.indexed_memory_layout: unit-stride-lhs-runtime-scalar-threshold-source-indexed-masked-destination-runtime-abi
// HEADER: weft.rvv.indexed_write_side_contract: source-before-active-indexed-write;destination-before-inactive-tail-preserve
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-scatter-store-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_scatter_store_unit_load.v1
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: void weft_emitc_pre_realized_body_rt_scalar_cmidx_store_kernel_pre_realized_body_rvv_rt_scalar_cmidx_store(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
