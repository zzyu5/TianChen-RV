// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_rt_scalar_cm_standalone_reduce_max_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_pre_rt_scalar_cm_standalone_reduce_max_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:scalar-output", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_max", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_pre_rt_scalar_cm_standalone_reduce_max_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @rvv_pre_rt_scalar_cm_standalone_reduce_max_m2
// REALIZED-SAME: sew = 32
// REALIZED: weft_rvv.splat
// REALIZED: weft_rvv.compare
// REALIZED-SAME: kind = "sle"
// REALIZED: weft_rvv.masked_standalone_reduce
// REALIZED-SAME: kind = "max"
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.standalone_reduce
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_standalone_reduce_max"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|masked-reduce-input|neutral-inactive|hdr;acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc|hdr;out=output-buffer:out:abi|acc-state|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.standalone_reduction_source_vector_c_type", value = "vint32m2_t"}
// PLAN-SAME: {key = "weft_rvv.standalone_reduction_scalar_result_vector_c_type", value = "vint32m1_t"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_zeroing_requirement", value = "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_rt_scalar_cm_standalone_reduce_max_m2

// HEADER-DAG: weft.rvv.selected_variant: @rvv_pre_rt_scalar_cm_standalone_reduce_max_m2
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1
// HEADER-DAG: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER-DAG: weft.rvv.standalone_reduction_source_vector_c_type: vint32m2_t
// HEADER-DAG: weft.rvv.standalone_reduction_scalar_result_vector_c_type: vint32m1_t
// HEADER: void weft_emitc_pre_rt_scalar_cm_standalone_reduce_max_m2_kernel_rvv_pre_rt_scalar_cm_standalone_reduce_max_m2(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);
