// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_masked_unit_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_masked_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %mask = weft_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:mask", role = "mask-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_masked_memory_pre_realized_body %src, %mask, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer", memory_form = "masked-unit-store", op_kind = "masked_unit_store", policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_masked_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-unit-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_masked_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_unit_store
// REALIZED: weft_rvv.mask_load
// REALIZED-SAME: mask_memory_form = "unit-stride-mask-load"
// REALIZED-SAME: mask_role = "predicate-mask-input-buffer"
// REALIZED: weft_rvv.load
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED: weft_rvv.masked_store
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-store"
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_masked_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_store"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1"}
// PLAN-SAME: {key = "weft_rvv.tail_policy", value = "undisturbed"}
// PLAN-SAME: {key = "weft_rvv.mask_policy", value = "undisturbed"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "masked-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "src,mask,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_unit_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-store-source-call|header-mirror;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-store-mask-call|header-mirror;dst=output-buffer:dst:runtime-abi-mirror|materialized-masked-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "weft_rvv.base_memory_movement_route_family_plan", value = "rvv-base-memory-movement-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-masked-unit-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-masked-unit-store-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,mask:b32,destination:masked-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-source-mask-destination-masked-store-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "masked-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "masked-unit-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_unit_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_masked_unit_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-masked-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1
// HEADER: weft.rvv.tail_policy: undisturbed
// HEADER: weft.rvv.mask_policy: undisturbed
// HEADER: weft.rvv.runtime_abi_order: src,mask,dst,n
// HEADER: weft.rvv.memory_form: masked-unit-store
// HEADER: weft.rvv.destination_memory_form: masked-unit-store
// HEADER: weft.rvv.mask_role: predicate-mask-input-buffer
// HEADER: weft.rvv.mask_source: runtime_abi:mask
// HEADER: weft.rvv.mask_memory_form: unit-stride-mask-load
// HEADER: weft.rvv.inactive_lane_contract: masked-store-false-lanes-preserve-output-buffer
// HEADER: weft.rvv.masked_passthrough_layout: masked-store-has-no-passthrough-load
// HEADER: weft.rvv.masked_memory_layout: unit-stride-source-mask-destination-masked-store-runtime-abi
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-masked-unit-store-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-masked-unit-store-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_unit_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_unit_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-store-source-call|header-mirror;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-store-mask-call|header-mirror;dst=output-buffer:dst:runtime-abi-mirror|materialized-masked-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: weft.rvv.base_memory_movement_route_family_plan: rvv-base-memory-movement-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,mask:b32,destination:masked-store
// HEADER: void weft_emitc_pre_realized_body_masked_unit_store_kernel_pre_realized_body_rvv_masked_unit_store(const int32_t *src, const int32_t *mask, int32_t *dst, size_t n);
