// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 indexed scatter
// memory movement slice. The RVV plugin must realize source/index/destination
// ABI operands into explicit load/index_load/move/indexed_store typed
// structure before the provider may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_indexed_scatter_unit_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_indexed_scatter_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      weft_rvv.typed_indexed_scatter_memory_pre_realized_body %src, %index, %dst, %n {index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", memory_form = "unit-load-indexed-store", offset_unit = "element", op_kind = "indexed_scatter_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_indexed_scatter_unit_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-indexed-scatter-unit-load-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-indexed-scatter-unit-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_indexed_scatter_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_indexed_scatter_unit_load
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: weft_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: weft_rvv.indexed_store
// REALIZED-SAME: index_uniqueness = "unique"
// REALIZED-SAME: offset_unit = "element"
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_indexed_scatter_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "indexed_scatter_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.move"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-load-indexed-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "src,index,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:indexed_scatter_unit_load.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:indexed_scatter_unit_load.v1;src=lhs-input-buffer:src:abi|materialized-load-base|move-source|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|materialized-indexed-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.base_memory_movement_route_family_plan", value = "rvv-base-memory-movement-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1"}
// PLAN-SAME: {key = "weft_rvv.indexed_memory_layout", value = "unit-stride-source-indexed-destination-index-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "weft_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "weft_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.indexed_destination_memory_form", value = "indexed-store"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "indexed-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_indexed_scatter_unit_load

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_indexed_scatter_unit_load
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: src,index,dst,n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:indexed_scatter_unit_load.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:indexed_scatter_unit_load.v1;src=lhs-input-buffer:src:abi|materialized-load-base|move-source|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|materialized-indexed-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1
// HEADER: void weft_emitc_pre_realized_body_indexed_scatter_unit_load_kernel_pre_realized_body_rvv_indexed_scatter_unit_load(const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
