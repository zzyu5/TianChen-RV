// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_masked_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_masked_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:mask", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-unit-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_masked_memory_pre_realized_body %src, %mask, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer", memory_form = "masked-unit-store", op_kind = "masked_unit_store", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_masked_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-unit-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_masked_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_unit_store
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.mask_load
// REALIZED-SAME: mask_memory_form = "unit-stride-mask-load"
// REALIZED-SAME: mask_role = "predicate-mask-input-buffer"
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED: tcrv_rvv.masked_store
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-store"
// REALIZED-NOT: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_masked_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_store"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1"}
// PLAN-SAME: {key = "tcrv_rvv.tail_policy", value = "undisturbed"}
// PLAN-SAME: {key = "tcrv_rvv.mask_policy", value = "undisturbed"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "masked-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,mask,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_unit_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-store-source-call;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-store-mask-call;dst=output-buffer:dst:runtime-abi-mirror|materialized-masked-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-source-mask-destination-masked-store-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "masked-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "masked-unit-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_unit_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_masked_unit_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-masked-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: src,mask,dst,n
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_unit_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-store-source-call;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|masked-store-mask-call;dst=output-buffer:dst:runtime-abi-mirror|materialized-masked-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void tcrv_emitc_pre_realized_body_masked_unit_store_kernel_pre_realized_body_rvv_masked_unit_store(const int32_t *src, const int32_t *mask, int32_t *dst, size_t n);
