// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated/s//provider_supported_mirror:rvv-script-derived-plain-elementwise/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:add.v1/s//rvv-route-operand-binding:script-derived-add.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs,rhs,out,n/s//lhs,out,rhs,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-TYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-elementwise-arithmetic-route-family-plan.v1/s//rvv-script-derived-elementwise-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.memory_form", value = "vector-rhs-load"/s//weft_rvv.memory_form", value = "script-derived-memory-form"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-MEMORY

// Pre-realized selected-body input. The RVV plugin must realize this bounded
// typed body before the provider route/common EmitC/target path can
// consume it.

module {
  weft.exec.kernel @pre_realized_body_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:n", role = "runtime-element-count"} : index
      weft_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_i32_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_i32_add
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "weft_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs:typed-vector,rhs:typed-vector,result:typed-vector"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_i32_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_i32_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: weft.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: weft.rvv.target_leaf_profile: rvv-v1-typed-plain-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated
// HEADER-DAG: weft.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: weft.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,lhs:typed-vector,rhs:typed-vector,result:typed-vector
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:add.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void weft_emitc_pre_realized_body_add_kernel_pre_realized_body_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// STALE-ELEM-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-PROVIDER: weft_rvv.provider_supported_mirror
// STALE-ELEM-PROVIDER-SAME: must mirror
// STALE-ELEM-PROVIDER-SAME: rvv-script-derived-plain-elementwise

// STALE-ELEM-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-BINDING: weft_rvv.route_operand_binding_plan
// STALE-ELEM-BINDING-SAME: must mirror
// STALE-ELEM-BINDING-SAME: rvv-route-operand-binding:script-derived-add.v1

// STALE-ELEM-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-ABI: weft_rvv.runtime_abi_order
// STALE-ELEM-ABI-SAME: must mirror
// STALE-ELEM-ABI-SAME: lhs,out,rhs,n

// STALE-ELEM-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-HEADER: weft_rvv.required_header_declarations
// STALE-ELEM-HEADER-SAME: must mirror
// STALE-ELEM-HEADER-SAME: stddef.h,stdint.h

// STALE-ELEM-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-TYPE: weft_rvv.c_type_mapping
// STALE-ELEM-TYPE-SAME: must mirror
// STALE-ELEM-TYPE-SAME: vl:uint64_t

// STALE-ELEM-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-PLAN: weft_rvv.elementwise_arithmetic_route_family_plan
// STALE-ELEM-PLAN-SAME: must mirror
// STALE-ELEM-PLAN-SAME: rvv-script-derived-elementwise-plan.v1

// STALE-ELEM-MEMORY: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-MEMORY: weft_rvv.memory_form
// STALE-ELEM-MEMORY-SAME: must mirror
// STALE-ELEM-MEMORY-SAME: script-derived-memory-form
