// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized reduce_add selected-body input. The RVV plugin must consume
// explicit accumulator/result layout facts into typed weft_rvv.reduce
// structure before the existing provider/common EmitC/target path can consume it.

module {
  weft.exec.kernel @pre_realized_body_reduce_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-reduce-add:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-reduce-add:accumulator-seed", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-reduce-add:result-output", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_reduce_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-reduce-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_reduce_add
// REALIZED: %[[INPUT:.*]] = weft_rvv.load
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[REDUCED:.*]] = weft_rvv.reduce %[[INPUT]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-reduction-lane0-to-output-chunk-base"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_reduce_pre_realized_body

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
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_reduce_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_reduce_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:reduce_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:reduce_add.v1;lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base|reduction-input-call;rhs=rhs-input-buffer:rhs:runtime-abi-mirror|materialized-accumulator-load-base|reduction-accumulator-call;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|reduction-result-store|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror
// HEADER: void weft_emitc_pre_realized_body_reduce_add_kernel_pre_realized_body_rvv_reduce_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
