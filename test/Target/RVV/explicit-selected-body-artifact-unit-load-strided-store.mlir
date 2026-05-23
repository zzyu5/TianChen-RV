// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 strided
// destination-store slice. The selected RVV body structurally carries a
// unit-stride source load, explicit destination byte-stride ABI value, generic
// move, and runtime byte-strided destination store.

module {
  tcrv.exec.kernel @explicit_selected_body_unit_load_strided_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unit_load_strided_store, sew = 32 : i64, source_kernel = "explicit_selected_body_unit_load_strided_store_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.strided_store %dst, %moved, %dst_stride_bytes, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, index, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_unit_load_strided_store {origin = "rvv-plugin", policy = "explicit-selected-body-unit-load-strided-store-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-unit-load-strided-store-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "unit_load_strided_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-load-strided-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,dst,n,dst_stride_bytes"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:unit_load_strided_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.base_memory_movement_route_family_plan", value = "rvv-base-memory-movement-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-unit-load-strided-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-unit-load-strided-store-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "unit-stride-source-byte-strided-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.destination_stride_source", value = "runtime_abi:dst_stride_bytes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "strided-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-unit-load-strided-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_unit_load_strided_store

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_unit_load_strided_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-unit-load-strided-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,dst,n,dst_stride_bytes
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-unit-load-strided-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-unit-load-strided-store-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:unit_load_strided_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror
// HEADER: tianchenrv.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1
// HEADER: void tcrv_emitc_explicit_selected_body_unit_load_strided_store_kernel_explicit_selected_body_rvv_unit_load_strided_store(const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);
