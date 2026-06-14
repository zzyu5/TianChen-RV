// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.reduction_kind", value = "add"/s//tcrv_rvv.reduction_kind", value = "max"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-REDUCTION-KIND
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"/s//tcrv_rvv.reduction_accumulator_layout", value = "scalar-i64-seed-lane0-from-accumulator-input"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-REDUCTION-ACC
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/rvv-route-operand-binding:standalone_reduce_add.v1/s//rvv-route-operand-binding:script-derived-standalone-reduce.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-REDUCTION-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/input:typed-source-vector/s//input:artifact-name-derived-vector/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-REDUCTION-TYPE

// Explicit standalone reduction selected-body input. The RVV plugin must derive
// source, scalar seed, scalar result, and runtime n/AVL facts from the typed
// tcrv_rvv body plus RouteOperandBindingPlan closure.

module {
  tcrv.exec.kernel @explicit_selected_body_standalone_reduce_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_standalone_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:input", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_standalone_reduce_add, sew = 32 : i64, source_kernel = "explicit_selected_body_standalone_reduce_add_kernel", status = "selected-lowering-boundary"} {
        %input = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %reduced = tcrv_rvv.standalone_reduce %input, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_standalone_reduce_add {origin = "rvv-plugin", policy = "explicit-selected-body-standalone-reduce-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-standalone-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "standalone_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-standalone-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:standalone_reduce_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:standalone_reduce_add.v1;lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;out=output-buffer:out:abi|acc-state|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_route_family_plan", value = "rvv-standalone-reduction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary", value = "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-standalone-reduction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-standalone-reduction-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_kind", value = "add"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-standalone-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_standalone_reduce_add

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_standalone_reduce_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-standalone-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.memory_form: unit-stride-standalone-reduction
// HEADER: tianchenrv.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input
// HEADER: tianchenrv.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar
// HEADER: tianchenrv.rvv.reduction_kind: add
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:standalone_reduce_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:standalone_reduce_add.v1;lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;out=output-buffer:out:abi|acc-state|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.standalone_reduction_route_family_plan: rvv-standalone-reduction-route-family-plan.v1
// HEADER: tianchenrv.rvv.standalone_reduction_scalar_result_runtime_boundary: scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1
// HEADER: void tcrv_emitc_explicit_selected_body_standalone_reduce_add_kernel_explicit_selected_body_rvv_standalone_reduce_add(const int32_t *lhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-REDUCTION-KIND: RVV materialized EmitC target artifact bridge failed
// STALE-REDUCTION-KIND: tcrv_rvv.reduction_kind
// STALE-REDUCTION-KIND-SAME: must mirror
// STALE-REDUCTION-KIND-SAME: max

// STALE-REDUCTION-ACC: RVV materialized EmitC target artifact bridge failed
// STALE-REDUCTION-ACC: tcrv_rvv.reduction_accumulator_layout
// STALE-REDUCTION-ACC-SAME: must mirror
// STALE-REDUCTION-ACC-SAME: scalar-i64-seed-lane0-from-accumulator-input

// STALE-REDUCTION-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-REDUCTION-BINDING: tcrv_rvv.route_operand_binding_plan
// STALE-REDUCTION-BINDING-SAME: must mirror
// STALE-REDUCTION-BINDING-SAME: rvv-route-operand-binding:script-derived-standalone-reduce.v1

// STALE-REDUCTION-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-REDUCTION-TYPE: tcrv_rvv.c_type_mapping
// STALE-REDUCTION-TYPE-SAME: must mirror
// STALE-REDUCTION-TYPE-SAME: input:artifact-name-derived-vector
