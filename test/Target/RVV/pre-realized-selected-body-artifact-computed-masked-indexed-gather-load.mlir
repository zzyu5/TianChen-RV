// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// indexed/gather masked source-load slice. The RVV plugin must realize compare
// lhs/rhs, explicit index vector, masked indexed source load, old-destination
// passthrough, loaded vector result, and unit-stride destination store into
// explicit load/load/load/index_load/compare/masked_indexed_load/store typed
// structure before the provider may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_cmidx_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_cmidx_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-gather-load:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_indexed_gather_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %index, %dst, %n {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-indexed-gather-load-unit-store", offset_unit = "element", op_kind = "computed_masked_indexed_gather_load_unit_store", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_cmidx_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-gather-load-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-gather-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_indexed_gather_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmidx_load
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[OLD_DST:.*]] = tcrv_rvv.load
// REALIZED: %[[INDICES:.*]] = tcrv_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[LOADED:.*]] = tcrv_rvv.masked_indexed_load %{{.*}}, %[[INDICES]], %[[MASK]], %[[OLD_DST]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED-SAME: memory_form = "masked-indexed-load"
// REALIZED-SAME: offset_unit = "element"
// REALIZED: tcrv_rvv.store %{{.*}}, %[[LOADED]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.indexed_load
// REALIZED-NOT: tcrv_rvv.strided_load
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.binary

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_indexed_gather_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_indexed_load"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-indexed-gather-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,index,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;src=source-input-buffer:src:abi|midx-base|midx-load-call;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr-mirror;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr-mirror;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-compare-indexed-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "unit-stride-compare-indexed-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "tcrv_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_data_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmidx_load

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_cmidx_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,index,dst,n
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;src=source-input-buffer:src:abi|midx-base|midx-load-call;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr-mirror;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr-mirror;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror
// HEADER: void tcrv_emitc_pre_realized_body_cmidx_load_kernel_pre_realized_body_rvv_cmidx_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
