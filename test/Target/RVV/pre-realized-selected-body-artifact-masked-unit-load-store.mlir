// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 masked unit-stride
// memory movement slice. The RVV plugin must realize source/mask/destination
// ABI operands into explicit mask_load/load/masked_load/store typed
// structure before the provider may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_masked_unit_load_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_masked_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-load-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %mask = weft_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-load-store:mask", role = "mask-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-load-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-load-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_masked_memory_pre_realized_body %src, %mask, %dst, %n {inactive_lane_policy = "preserve-old-destination", lmul = "m1", mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer", memory_form = "masked-unit-load-store", op_kind = "masked_unit_load_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_masked_unit_load_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-unit-load-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-unit-load-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_masked_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_unit_load_store
// REALIZED: weft_rvv.mask_load
// REALIZED-SAME: mask_memory_form = "unit-stride-mask-load"
// REALIZED-SAME: mask_role = "predicate-mask-input-buffer"
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.masked_load
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-load"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.indexed_store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_masked_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_unit_load_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_load"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "masked-unit-load-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "src,mask,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_unit_load_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_unit_load_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-masked-load-base|masked-load-source-call|header-mirror;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-load-mask-call|header-mirror;dst=output-buffer:dst:runtime-abi-mirror|materialized-old-destination-load-base|masked-load-passthrough-call|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-source-mask-old-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-input-buffer"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "runtime_abi:mask"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "unit-stride-mask-load"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-unit-load-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_unit_load_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_masked_unit_load_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-masked-unit-load-store-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: src,mask,dst,n
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_unit_load_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_unit_load_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-masked-load-base|masked-load-source-call|header-mirror;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-load-mask-call|header-mirror;dst=output-buffer:dst:runtime-abi-mirror|materialized-old-destination-load-base|masked-load-passthrough-call|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void weft_emitc_pre_realized_body_masked_unit_load_store_kernel_pre_realized_body_rvv_masked_unit_load_store(const int32_t *src, const int32_t *mask, int32_t *dst, size_t n);
