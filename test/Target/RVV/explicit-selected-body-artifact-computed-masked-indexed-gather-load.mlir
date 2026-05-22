// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 computed
// mask plus indexed/gather masked source-load slice. The selected RVV body
// structurally carries compare-produced mask, source memory window, explicit
// runtime index vector, old-destination passthrough for inactive lanes, loaded
// result, and unit-stride destination store through tcrv_rvv.masked_indexed_load.

module {
  tcrv.exec.kernel @explicit_selected_body_cmidx_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_cmidx_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-indexed-gather-load:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_cmidx_load, sew = 32 : i64, source_kernel = "explicit_selected_body_cmidx_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old_dst = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %loaded = tcrv_rvv.masked_indexed_load %src, %indices, %mask, %old_dst, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, memory_form = "masked-indexed-load", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %dst, %loaded, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_cmidx_load {origin = "rvv-plugin", policy = "explicit-selected-body-computed-mask-indexed-gather-load-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-mask-indexed-gather-load-fallback-envelope"}
    }
  }
}

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
// PLAN-SAME: target = @explicit_selected_body_rvv_cmidx_load

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_cmidx_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-indexed-gather-load-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,index,dst,n
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;src=source-input-buffer:src:abi|midx-base|midx-load-call;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr-mirror;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr-mirror;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr-mirror
// HEADER: void tcrv_emitc_explicit_selected_body_cmidx_load_kernel_explicit_selected_body_rvv_cmidx_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
