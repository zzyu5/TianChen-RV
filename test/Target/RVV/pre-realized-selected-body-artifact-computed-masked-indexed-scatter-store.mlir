// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// indexed/scatter masked destination-store slice. The RVV plugin must realize
// compare lhs/rhs, unit-stride payload source, explicit index vector, and
// masked indexed no-write destination store into explicit typed structure before
// the provider may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_cmidx_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_cmidx_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-indexed-scatter-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %index, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-indexed-scatter-store", offset_unit = "element", op_kind = "computed_masked_indexed_scatter_store_unit_load", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_cmidx_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-scatter-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-indexed-scatter-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmidx_store
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[PAYLOAD:.*]] = tcrv_rvv.load
// REALIZED: %[[INDICES:.*]] = tcrv_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: tcrv_rvv.masked_indexed_store %{{.*}}, %[[INDICES]], %[[MASK]], %[[PAYLOAD]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED-SAME: index_uniqueness = "unique"
// REALIZED-SAME: memory_form = "masked-indexed-store"
// REALIZED-SAME: offset_unit = "element"
// REALIZED-NOT: tcrv_rvv.masked_indexed_load
// REALIZED-NOT: tcrv_rvv.indexed_store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.binary

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_indexed_scatter_store_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_indexed_store"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-unit-load-indexed-scatter-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,index,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|src-load|mistore-src-call|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|mistore-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-indexed-scatter-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-compare-source-indexed-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-indexed-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "masked-indexed-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "unit-stride-compare-source-indexed-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "tcrv_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "tcrv_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmidx_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_cmidx_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-indexed-scatter-store-unit-load-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,index,dst,n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-indexed-scatter-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-indexed-scatter-store-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_indexed_scatter_store_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|src-load|mistore-src-call|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|mistore-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: tianchenrv.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,compare/source:signed-e32m1,index:u32m1,mask:b32,dst:masked-indexed-store
// HEADER: void tcrv_emitc_pre_realized_body_cmidx_store_kernel_pre_realized_body_rvv_cmidx_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
