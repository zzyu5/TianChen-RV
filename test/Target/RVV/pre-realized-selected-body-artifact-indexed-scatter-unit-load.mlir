// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 indexed scatter
// memory movement slice. The RVV plugin must realize source/index/destination
// ABI operands into explicit load/index_load/move/indexed_store typed
// structure before the provider may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_indexed_scatter_unit_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_indexed_scatter_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body %src, %index, %dst, %n {index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", memory_form = "unit-load-indexed-store", offset_unit = "element", op_kind = "indexed_scatter_unit_load", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_indexed_scatter_unit_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-indexed-scatter-unit-load-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-indexed-scatter-unit-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_indexed_scatter_unit_load
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: tcrv_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: tcrv_rvv.indexed_store
// REALIZED-SAME: index_uniqueness = "unique"
// REALIZED-SAME: offset_unit = "element"
// REALIZED-NOT: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "indexed_scatter_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-load-indexed-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,index,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:indexed_scatter_unit_load.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:indexed_scatter_unit_load.v1;src=lhs-input-buffer:src:abi|materialized-load-base|move-source|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|materialized-indexed-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.base_memory_movement_route_family_plan", value = "rvv-base-memory-movement-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "unit-stride-source-indexed-destination-index-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "tcrv_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "tcrv_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_destination_memory_form", value = "indexed-store"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "indexed-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_indexed_scatter_unit_load

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_indexed_scatter_unit_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-indexed-scatter-unit-load-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,index,dst,n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-indexed-scatter-unit-load-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-indexed-scatter-unit-load-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:indexed_scatter_unit_load.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:indexed_scatter_unit_load.v1;src=lhs-input-buffer:src:abi|materialized-load-base|move-source|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;dst=output-buffer:dst:abi|materialized-indexed-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: tianchenrv.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,index:u32m1,destination:indexed-e32m1
// HEADER: void tcrv_emitc_pre_realized_body_indexed_scatter_unit_load_kernel_pre_realized_body_rvv_indexed_scatter_unit_load(const int32_t *src, const uint32_t *index, int32_t *dst, size_t n);
