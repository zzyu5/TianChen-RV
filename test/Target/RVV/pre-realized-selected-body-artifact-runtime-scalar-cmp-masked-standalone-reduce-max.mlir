// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_rt_scalar_cm_standalone_reduce_max_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_pre_rt_scalar_cm_standalone_reduce_max attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_max", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_pre_rt_scalar_cm_standalone_reduce_max {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-max-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @rvv_pre_rt_scalar_cm_standalone_reduce_max
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED: %[[SRC:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.masked_standalone_reduce %[[MASK]], %[[SRC]], %{{.*}}, %[[VL]]
// REALIZED-SAME: kind = "max"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.standalone_reduce
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.masked_store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_standalone_reduce_max"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-unit-stride-standalone-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,src,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|masked-reduce-input|neutral-inactive|hdr;acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc|hdr;out=output-buffer:out:abi|acc-state|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_zeroing_requirement", value = "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_rt_scalar_cm_standalone_reduce_max

// HEADER: tianchenrv.rvv.selected_variant: @rvv_pre_rt_scalar_cm_standalone_reduce_max
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-max-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,src,acc,out,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1
// HEADER: void tcrv_emitc_pre_rt_scalar_cm_standalone_reduce_max_kernel_rvv_pre_rt_scalar_cm_standalone_reduce_max(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);
