// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided destination
// store slice. The RVV plugin must realize destination byte-stride ABI facts
// into explicit load/move/strided_store typed structure before the provider
// may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_unit_load_strided_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      tcrv_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_unit_load_strided_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_strided_store_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_unit_load_strided_store
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.strided_load
// REALIZED-NOT: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_strided_store_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "unit_load_strided_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-load-strided-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,dst,n,dst_stride_bytes"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:unit_load_strided_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror"}
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
// PLAN-SAME: target = @pre_realized_body_rvv_unit_load_strided_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_unit_load_strided_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-unit-load-strided-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,dst,n,dst_stride_bytes
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:unit_load_strided_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:unit_load_strided_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|move-source;dst=output-buffer:dst:runtime-abi-mirror|materialized-strided-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:runtime-abi-mirror|materialized-strided-store-stride|materialized-byte-address|header-mirror
// HEADER: tianchenrv.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,destination:byte-strided-e32m1
// HEADER: tianchenrv.rvv.strided_memory_layout: unit-stride-source-byte-strided-destination-runtime-abi
// HEADER: tianchenrv.rvv.destination_stride_source: runtime_abi:dst_stride_bytes
// HEADER: tianchenrv.rvv.source_memory_form: unit-stride-load
// HEADER: tianchenrv.rvv.destination_memory_form: strided-store
// HEADER: void tcrv_emitc_pre_realized_body_unit_load_strided_store_kernel_pre_realized_body_rvv_unit_load_strided_store(const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);
