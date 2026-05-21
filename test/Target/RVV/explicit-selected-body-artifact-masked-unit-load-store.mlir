// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 masked
// unit-stride memory movement slice. The selected RVV body structurally carries
// source/mask/destination runtime ABI roles, a source load, mask load,
// old-destination preserving load, masked move, and unit-stride store.

module {
  tcrv.exec.kernel @explicit_selected_body_masked_unit_load_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_masked_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-unit-load-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-unit-load-store:mask", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-unit-load-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-unit-load-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_masked_unit_load_store, sew = 32 : i64, source_kernel = "explicit_selected_body_masked_unit_load_store_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %old = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %moved = tcrv_rvv.masked_move %predicate, %loaded, %old, %vl {kind = "active-source-preserve-old-destination"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %dst, %moved, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_masked_unit_load_store {origin = "rvv-plugin", policy = "explicit-selected-body-masked-unit-load-store-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-masked-unit-load-store-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_unit_load_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_move"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "masked-unit-load-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,mask,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_unit_load_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_unit_load_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-move-source-call;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|mask-compare-call;dst=output-buffer:dst:runtime-abi-mirror|materialized-old-destination-load-base|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-source-mask-old-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-input-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "runtime_abi:mask"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "unit-stride-mask-load"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-unit-load-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_masked_unit_load_store

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_masked_unit_load_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-masked-unit-load-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,mask,dst,n
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_unit_load_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_unit_load_store.v1;src=lhs-input-buffer:src:runtime-abi-mirror|materialized-load-base|masked-move-source-call;mask=mask-input-buffer:mask:runtime-abi-mirror|materialized-mask-load-base|mask-compare-call;dst=output-buffer:dst:runtime-abi-mirror|materialized-old-destination-load-base|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void tcrv_emitc_explicit_selected_body_masked_unit_load_store_kernel_explicit_selected_body_rvv_masked_unit_load_store(const int32_t *src, const int32_t *mask, int32_t *dst, size_t n);
