// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-generic-scalar-broadcast-macc-add-emitc-route/s//rvv-script-derived-scalar-broadcast-macc-route/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-ROUTE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-scalar-broadcast-macc-route-family-plan.v1/s//rvv-script-derived-scalar-broadcast-macc-plan.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lhs,rhs_scalar,acc,out,n/s//lhs,acc,rhs_scalar,out,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-ABI
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-TYPE

// Pre-realized scalar-broadcast macc selected-body input. The RVV plugin must
// consume explicit typed operation/config/runtime facts into load/splat/load/
// macc/store structure before the provider route/common EmitC/target path can
// construct scalar_broadcast_macc_add route facts.

module {
  tcrv.exec.kernel @pre_realized_body_scalar_broadcast_macc_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @pre_realized_body_rvv_scalar_broadcast_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_macc_pre_realized_body %lhs, %rhs_scalar, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", memory_form = "rhs-scalar-broadcast-macc", op_kind = "scalar_broadcast_macc_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_scalar_broadcast_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_scalar_broadcast_macc_add
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED: %[[ACC:.*]] = tcrv_rvv.load
// REALIZED: %[[SUM:.*]] = tcrv_rvv.macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_macc_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "scalar_broadcast_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.macc"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "rhs-scalar-broadcast-macc"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "tcrv_rvv.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "tcrv_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_scalar_broadcast_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-case"}
// PLAN-SAME: {key = "tcrv_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:scalar_broadcast_macc_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:scalar_broadcast_macc_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|macc-lhs-call;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|macc-rhs-call;acc=accumulator-input-buffer:acc:runtime-abi-mirror|materialized-accumulator-load-base|macc-accumulator-call;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.exec_abi_bindings", value = "lhs=lhs-input-buffer->@abi_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;out=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count"}
// PLAN-SAME: {key = "tcrv_rvv.scalar_broadcast_macc_route_family_plan", value = "rvv-scalar-broadcast-macc-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-scalar-broadcast-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-scalar-broadcast-macc-add-composition-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/acc:signed-e32m1,rhs_scalar:i32,result:signed-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_scalar_broadcast_macc_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_scalar_broadcast_macc_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,acc,out,n
// HEADER: tianchenrv.rvv.memory_form: rhs-scalar-broadcast-macc
// HEADER: tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER: tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic
// HEADER: tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_scalar_broadcast_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-case
// HEADER: tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_macc_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_macc_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|macc-lhs-call;rhs_scalar=rhs-scalar-value:rhs_scalar:runtime-abi-mirror|scalar-broadcast-rhs-call|macc-rhs-call;acc=accumulator-input-buffer:acc:runtime-abi-mirror|materialized-accumulator-load-base|macc-accumulator-call;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: tianchenrv.rvv.exec_abi_bindings: lhs=lhs-input-buffer->@abi_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;out=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count
// HEADER: tianchenrv.rvv.scalar_broadcast_macc_route_family_plan: rvv-scalar-broadcast-macc-route-family-plan.v1
// HEADER: void tcrv_emitc_pre_realized_body_scalar_broadcast_macc_add_kernel_pre_realized_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);

// STALE-SBMACC-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-ROUTE: candidate rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route
// STALE-SBMACC-ROUTE-SAME: rvv-script-derived-scalar-broadcast-macc-route

// STALE-SBMACC-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-PLAN: tcrv_rvv.scalar_broadcast_macc_route_family_plan
// STALE-SBMACC-PLAN-SAME: must mirror
// STALE-SBMACC-PLAN-SAME: rvv-script-derived-scalar-broadcast-macc-plan.v1

// STALE-SBMACC-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-ABI: tcrv_rvv.runtime_abi_order
// STALE-SBMACC-ABI-SAME: must mirror
// STALE-SBMACC-ABI-SAME: lhs,acc,rhs_scalar,out,n

// STALE-SBMACC-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-TYPE: tcrv_rvv.c_type_mapping
// STALE-SBMACC-TYPE-SAME: must mirror
// STALE-SBMACC-TYPE-SAME: vl:uint64_t
