// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 indexed
// gather memory movement slice. The selected RVV body structurally carries
// data/index/output runtime ABI roles, an explicit index vector load, an
// indexed data load with element offsets, a generic move, and a unit-stride
// destination store.

module {
  weft.exec.kernel @explicit_selected_body_indexed_gather_unit_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_indexed_gather_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %data = weft_rvv.runtime_abi_value {c_name = "data", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:data", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_indexed_gather_unit_store, sew = 32 : i64, source_kernel = "explicit_selected_body_indexed_gather_unit_store_kernel", status = "selected-lowering-boundary"} {
        %indices = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %loaded = weft_rvv.indexed_load %data, %indices, %vl {index_eew = 32 : i64, offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %moved = weft_rvv.move %loaded, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %moved, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_indexed_gather_unit_store {origin = "rvv-plugin", policy = "explicit-selected-body-indexed-gather-unit-store-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-indexed-gather-unit-store-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "indexed_gather_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.move"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "indexed-load-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "data,index,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:indexed_gather_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:indexed_gather_unit_store.v1;data=lhs-input-buffer:data:abi|materialized-indexed-data-base|indexed-load-base|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.base_memory_movement_route_family_plan", value = "rvv-base-memory-movement-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-indexed-gather-unit-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-indexed-gather-unit-store-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,data:signed-e32m1,index:u32m1,result:signed-e32m1"}
// PLAN-SAME: {key = "weft_rvv.indexed_memory_layout", value = "element-indexed-data-index-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "weft_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "weft_rvv.indexed_data_memory_form", value = "indexed-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-indexed-gather-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_indexed_gather_unit_store

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_indexed_gather_unit_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-indexed-gather-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: data,index,out,n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-indexed-gather-unit-store-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-indexed-gather-unit-store-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:indexed_gather_unit_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:indexed_gather_unit_store.v1;data=lhs-input-buffer:data:abi|materialized-indexed-data-base|indexed-load-base|hdr;index=index-input-buffer:index:abi|materialized-index-load-base|index-offset-scale|index-source-mirror|hdr;out=output-buffer:out:abi|materialized-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,data:signed-e32m1,index:u32m1,result:signed-e32m1
// HEADER: void weft_emitc_explicit_selected_body_indexed_gather_unit_store_kernel_explicit_selected_body_rvv_indexed_gather_unit_store(const int32_t *data, const uint32_t *index, int32_t *out, size_t n);
