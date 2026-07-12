// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Stage2 reduction/accumulation selected-body input. Reduction dataflow is
// carried by generic weft_rvv.reduce, including accumulator/result layout
// facts; this fixture proves route/header materialization only and does not
// claim ssh rvv correctness.

module {
  weft.exec.kernel @explicit_selected_body_reduce_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:accumulator", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_reduce_add, sew = 32 : i64, source_kernel = "explicit_selected_body_reduce_add_kernel", status = "selected-lowering-boundary"} {
        %input = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %acc = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %reduced = weft_rvv.reduce %input, %acc, %vl {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", kind = "add", result_layout = "store-reduction-lane0-to-output-chunk-base"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_reduce_add {origin = "rvv-plugin", policy = "explicit-selected-body-reduce-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.reduce"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:reduce_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:reduce_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|reduction-input-call;rhs=rhs-input-buffer:rhs:runtime-abi-mirror|materialized-accumulator-load-base|reduction-accumulator-call;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|reduction-result-store|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror"}
// PLAN-SAME: {key = "weft_rvv.reduction_accumulator_layout", value = "rhs-vector-seed-lane0-per-vl-chunk"}
// PLAN-SAME: {key = "weft_rvv.reduction_result_layout", value = "store-reduction-lane0-to-output-chunk-base"}
// PLAN-SAME: {key = "weft_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: runtime_abi_name = "rvv-generic-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_reduce_add

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_reduce_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:reduce_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:reduce_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|reduction-input-call;rhs=rhs-input-buffer:rhs:runtime-abi-mirror|materialized-accumulator-load-base|reduction-accumulator-call;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|reduction-result-store|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void weft_emitc_explicit_selected_body_reduce_add_kernel_explicit_selected_body_rvv_reduce_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
