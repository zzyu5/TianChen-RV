// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.runtime_abi_order\", value = \"src,dst,n,dst_stride_bytes/s//weft_rvv.runtime_abi_order\", value = \"src,n,dst,dst_stride_bytes/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RUNTIME-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.route_operand_binding_operands\", value = \"rvv-route-operand-binding:unit_load_strided_store.v1/s//weft_rvv.route_operand_binding_operands\", value = \"metadata-derived-binding/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.provider_supported_mirror\", value = \"provider_supported_mirror:rvv-unit-load-strided-store-plan-validated/s//weft_rvv.provider_supported_mirror\", value = \"provider_supported_mirror:metadata-only-unit-strided/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.strided_memory_layout\", value = \"unit-stride-source-byte-strided-destination-runtime-abi/s//weft_rvv.strided_memory_layout\", value = \"metadata-derived-layout/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LAYOUT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.destination_stride_source\", value = \"runtime_abi:dst_stride_bytes/s//weft_rvv.destination_stride_source\", value = \"metadata-derived-dst-stride/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEST-STRIDE

// Pre-realized selected-body input for one bounded Stage2 strided destination
// store slice. The RVV plugin must realize destination byte-stride ABI facts
// into explicit load/move/strided_store typed structure before the provider
// may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_unit_load_strided_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_unit_load_strided_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_strided_store_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_unit_load_strided_store
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_strided_store_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "unit_load_strided_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.move"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-load-strided-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "src,dst,n,dst_stride_bytes"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:unit_load_strided_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "unit-stride-source-byte-strided-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.destination_stride_source", value = "runtime_abi:dst_stride_bytes"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "strided-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-unit-load-strided-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_unit_load_strided_store

// HEADER-DAG: weft.rvv.selected_variant: @pre_realized_body_rvv_unit_load_strided_store
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-generic-unit-load-strided-store-callable-c-abi.v1
// HEADER-DAG: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: weft.rvv.memory_form: unit-load-strided-store
// HEADER-DAG: weft.rvv.runtime_abi_order: src,dst,n,dst_stride_bytes
// HEADER-DAG: weft.rvv.target_leaf_profile: rvv-v1-e32m1-unit-load-strided-store-leaf-profile.v1
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-unit-load-strided-store-plan-validated
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:unit_load_strided_store.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror
// HEADER-DAG: weft.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1
// HEADER-DAG: weft.rvv.strided_memory_layout: unit-stride-source-byte-strided-destination-runtime-abi
// HEADER-DAG: weft.rvv.destination_stride_source: runtime_abi:dst_stride_bytes
// HEADER-DAG: weft.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: weft.rvv.destination_memory_form: strided-store
// HEADER: void weft_emitc_pre_realized_body_unit_load_strided_store_kernel_pre_realized_body_rvv_unit_load_strided_store(const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);

// STALE-RUNTIME-ABI: target artifact candidate validation failed
// STALE-RUNTIME-ABI-SAME: runtime_abi_order
// STALE-RUNTIME-ABI-SAME: src,dst,n,dst_stride_bytes
// STALE-RUNTIME-ABI-SAME: src,n,dst,dst_stride_bytes

// STALE-BINDING: target artifact candidate validation failed
// STALE-BINDING-SAME: route_operand_binding_operands
// STALE-BINDING-SAME: rvv-route-operand-binding:unit_load_strided_store.v1
// STALE-BINDING-SAME: metadata-derived-binding

// STALE-PROVIDER: target artifact candidate validation failed
// STALE-PROVIDER-SAME: provider_supported_mirror
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-unit-load-strided-store-plan-validated
// STALE-PROVIDER-SAME: provider_supported_mirror:metadata-only-unit-strided

// STALE-LAYOUT: target artifact candidate validation failed
// STALE-LAYOUT-SAME: strided_memory_layout
// STALE-LAYOUT-SAME: unit-stride-source-byte-strided-destination-runtime-abi
// STALE-LAYOUT-SAME: metadata-derived-layout

// STALE-DEST-STRIDE: target artifact candidate validation failed
// STALE-DEST-STRIDE-SAME: destination_stride_source
// STALE-DEST-STRIDE-SAME: runtime_abi:dst_stride_bytes
// STALE-DEST-STRIDE-SAME: metadata-derived-dst-stride
