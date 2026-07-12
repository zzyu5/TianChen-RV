// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-generic-scalar-broadcast-macc-add-emitc-route/s//rvv-script-derived-scalar-broadcast-macc-route/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-ROUTE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-scalar-broadcast-macc-route-family-plan.v1/s//rvv-script-derived-scalar-broadcast-macc-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs,rhs_scalar,acc,out,n/s//lhs,acc,rhs_scalar,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr/s//rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SBMACC-TYPE

// Pre-realized scalar-broadcast macc selected-body input. The RVV plugin must
// consume explicit typed operation/config/runtime facts into load/splat/load/
// macc/store structure before the provider route/common EmitC/target path can
// construct scalar_broadcast_macc_add route facts.

module {
  weft.exec.kernel @pre_realized_body_scalar_broadcast_macc_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.variant @pre_realized_body_rvv_scalar_broadcast_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, weft_rvv.require_exec_abi_bindings = true} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:n", role = "runtime-element-count"} : index
      weft_rvv.typed_macc_pre_realized_body %lhs, %rhs_scalar, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", memory_form = "rhs-scalar-broadcast-macc", op_kind = "scalar_broadcast_macc_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_scalar_broadcast_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_scalar_broadcast_macc_add
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[SUM:.*]] = weft_rvv.macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "scalar_broadcast_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.macc"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "rhs-scalar-broadcast-macc"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "weft_rvv.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "weft_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_scalar_broadcast_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-case"}
// PLAN-SAME: {key = "weft_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:scalar_broadcast_macc_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:scalar_broadcast_macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.exec_abi_bindings", value = "lhs=lhs-input-buffer->@abi_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;out=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count"}
// PLAN-SAME: {key = "weft_rvv.scalar_broadcast_macc_route_family_plan", value = "rvv-scalar-broadcast-macc-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-scalar-broadcast-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-scalar-broadcast-macc-add-composition-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/acc:typed-vector,rhs_scalar:typed-scalar,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_scalar_broadcast_macc_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_scalar_broadcast_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,acc,out,n
// HEADER: weft.rvv.memory_form: rhs-scalar-broadcast-macc
// HEADER: weft.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER: weft.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic
// HEADER: weft.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_scalar_broadcast_macc_add;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-case
// HEADER: weft.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:scalar_broadcast_macc_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:scalar_broadcast_macc_add.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.exec_abi_bindings: lhs=lhs-input-buffer->@abi_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;out=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count
// HEADER: weft.rvv.scalar_broadcast_macc_route_family_plan: rvv-scalar-broadcast-macc-route-family-plan.v1
// HEADER: void weft_emitc_pre_realized_body_scalar_broadcast_macc_add_kernel_pre_realized_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);

// STALE-SBMACC-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-ROUTE: candidate rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route
// STALE-SBMACC-ROUTE-SAME: rvv-script-derived-scalar-broadcast-macc-route

// STALE-SBMACC-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-PLAN: weft_rvv.scalar_broadcast_macc_route_family_plan
// STALE-SBMACC-PLAN-SAME: must mirror
// STALE-SBMACC-PLAN-SAME: rvv-script-derived-scalar-broadcast-macc-plan.v1

// STALE-SBMACC-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-ABI: weft_rvv.runtime_abi_order
// STALE-SBMACC-ABI-SAME: must mirror
// STALE-SBMACC-ABI-SAME: lhs,acc,rhs_scalar,out,n

// STALE-SBMACC-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-BINDING: weft_rvv.route_operand_binding_operands
// STALE-SBMACC-BINDING-SAME: must mirror
// STALE-SBMACC-BINDING-SAME: rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs

// STALE-SBMACC-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-SBMACC-TYPE: weft_rvv.c_type_mapping
// STALE-SBMACC-TYPE-SAME: must mirror
// STALE-SBMACC-TYPE-SAME: vl:uint64_t
