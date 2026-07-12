// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Explicit signed min standalone reduction selected-body input. The RVV plugin
// must derive reduction kind, source, scalar seed/output, and runtime n/AVL
// facts from the typed weft_rvv body plus RouteOperandBindingPlan closure.

module {
  weft.exec.kernel @explicit_selected_body_standalone_reduce_min_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_standalone_reduce_min attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-min:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-min:seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-min:scalar-output", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-min:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_standalone_reduce_min, sew = 32 : i64, source_kernel = "explicit_selected_body_standalone_reduce_min_kernel", status = "selected-lowering-boundary"} {
        %input = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %reduced = weft_rvv.standalone_reduce %input, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "min", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_standalone_reduce_min {origin = "rvv-plugin", policy = "explicit-selected-body-standalone-reduce-min-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-standalone-reduce-min-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "standalone_reduce_min"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.standalone_reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-standalone-reduction"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:standalone_reduce_min.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:standalone_reduce_min.v1;lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;out=output-buffer:out:abi|acc-state|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-standalone-reduction-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-standalone-reduction-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector"}
// PLAN-SAME: {key = "weft_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "weft_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-standalone-reduce-min-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_standalone_reduce_min

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_standalone_reduce_min
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-standalone-reduce-min-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.memory_form: unit-stride-standalone-reduction
// HEADER: weft.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input
// HEADER: weft.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:standalone_reduce_min.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:standalone_reduce_min.v1;lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;out=output-buffer:out:abi|acc-state|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: void weft_emitc_explicit_selected_body_standalone_reduce_min_kernel_explicit_selected_body_rvv_standalone_reduce_min(const int32_t *lhs, const int32_t *acc, int32_t *out, size_t n);
