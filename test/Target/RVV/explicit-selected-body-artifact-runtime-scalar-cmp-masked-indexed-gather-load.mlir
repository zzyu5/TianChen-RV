// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/runtime-scalar-splat-compare-rhs/vector-compare-rhs-load/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCER

// Hand-authored explicit selected-body input for one bounded Stage2 runtime
// scalar compare plus masked indexed gather-load slice. The selected RVV body
// structurally carries rhs_scalar through weft_rvv.splat, produces the mask
// from lhs <= rhs_scalar, gathers active lanes from src[index[i]], preserves
// old destination on inactive lanes, and stores the result through the target
// artifact ABI boundary.

module {
  weft.exec.kernel @explicit_selected_body_rt_scalar_cmidx_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_rt_scalar_cmidx_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_rt_scalar_cmidx_load, sew = 32 : i64, source_kernel = "explicit_selected_body_rt_scalar_cmidx_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old_dst = weft_rvv.load %dst, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %indices = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %loaded = weft_rvv.masked_indexed_load %src, %indices, %mask, %old_dst, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, memory_form = "masked-indexed-load", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %dst, %loaded, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_rt_scalar_cmidx_load {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-indexed-gather-load-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_indexed_gather_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_indexed_load"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-indexed-gather-load-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,index,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_gather_load_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_gather_load_unit_store.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src=source-input-buffer:src:abi|midx-base|midx-load-call|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-load-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-load-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/source/passthrough:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,result:runtime-scalar-masked-indexed-load-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.indexed_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "weft_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "weft_rvv.indexed_data_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_rt_scalar_cmidx_load

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_rt_scalar_cmidx_load
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-indexed-gather-load-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,src,index,dst,n
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.indexed_memory_layout: unit-stride-lhs-runtime-scalar-threshold-indexed-masked-source-old-destination-runtime-abi
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-load-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-load-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_indexed_gather_load_unit_store.v1
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: void weft_emitc_explicit_selected_body_rt_scalar_cmidx_load_kernel_explicit_selected_body_rvv_rt_scalar_cmidx_load(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);

// STALE-PRODUCER: metadata key '{{.*}}computed_mask_memory_mask_producer_source'{{.*}}'runtime-scalar-splat-compare-rhs' but was 'vector-compare-rhs-load'
