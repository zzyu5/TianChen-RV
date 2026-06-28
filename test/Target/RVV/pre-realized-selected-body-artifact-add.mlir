// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated/s//provider_supported_mirror:rvv-script-derived-plain-elementwise/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-route-operand-binding:add.v1/s//rvv-route-operand-binding:script-derived-add.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lhs,rhs,out,n/s//lhs,out,rhs,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-ABI
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-TYPE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-elementwise-arithmetic-route-family-plan.v1/s//rvv-script-derived-elementwise-plan.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.memory_form", value = "vector-rhs-load"/s//tcrv_rvv.memory_form", value = "script-derived-memory-form"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEM-MEMORY

// Pre-realized selected-body input. The RVV plugin must realize this bounded
// typed body before the provider route/common EmitC/target path can
// consume it.

module {
  tcrv.exec.kernel @pre_realized_body_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_i32_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_i32_add
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "tcrv_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-plain-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs:typed-vector,rhs:typed-vector,result:typed-vector"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_i32_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_i32_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-DAG: tianchenrv.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-plain-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated
// HEADER-DAG: tianchenrv.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs:typed-vector,rhs:typed-vector,result:typed-vector
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:add.v1
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void tcrv_emitc_pre_realized_body_add_kernel_pre_realized_body_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// STALE-ELEM-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-PROVIDER: tcrv_rvv.provider_supported_mirror
// STALE-ELEM-PROVIDER-SAME: must mirror
// STALE-ELEM-PROVIDER-SAME: rvv-script-derived-plain-elementwise

// STALE-ELEM-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-BINDING: tcrv_rvv.route_operand_binding_plan
// STALE-ELEM-BINDING-SAME: must mirror
// STALE-ELEM-BINDING-SAME: rvv-route-operand-binding:script-derived-add.v1

// STALE-ELEM-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-ABI: tcrv_rvv.runtime_abi_order
// STALE-ELEM-ABI-SAME: must mirror
// STALE-ELEM-ABI-SAME: lhs,out,rhs,n

// STALE-ELEM-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-HEADER: tcrv_rvv.required_header_declarations
// STALE-ELEM-HEADER-SAME: must mirror
// STALE-ELEM-HEADER-SAME: stddef.h,stdint.h

// STALE-ELEM-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-TYPE: tcrv_rvv.c_type_mapping
// STALE-ELEM-TYPE-SAME: must mirror
// STALE-ELEM-TYPE-SAME: vl:uint64_t

// STALE-ELEM-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-PLAN: tcrv_rvv.elementwise_arithmetic_route_family_plan
// STALE-ELEM-PLAN-SAME: must mirror
// STALE-ELEM-PLAN-SAME: rvv-script-derived-elementwise-plan.v1

// STALE-ELEM-MEMORY: RVV materialized EmitC target artifact bridge failed
// STALE-ELEM-MEMORY: tcrv_rvv.memory_form
// STALE-ELEM-MEMORY-SAME: must mirror
// STALE-ELEM-MEMORY-SAME: script-derived-memory-form
