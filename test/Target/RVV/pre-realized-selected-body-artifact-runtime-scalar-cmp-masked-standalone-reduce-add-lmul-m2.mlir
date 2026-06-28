// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_rt_scalar_cm_standalone_reduce_m2_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_pre_rt_scalar_cm_standalone_reduce_m2 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_pre_rt_scalar_cm_standalone_reduce_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @rvv_pre_rt_scalar_cm_standalone_reduce_m2
// REALIZED-SAME: sew = 32
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED: %[[SRC:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.masked_standalone_reduce %[[MASK]], %[[SRC]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.standalone_reduce
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.masked_store
// REALIZED-NOT: tcrv_rvv.masked_macc
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_standalone_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,src,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_route_family_plan", value = "rvv-standalone-reduction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_source_vector_type", value = "!tcrv_rvv.vector<i32, \22m2\22>"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_source_vector_c_type", value = "vint32m2_t"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_scalar_result_vector_type", value = "!tcrv_rvv.vector<i32, \22m1\22>"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type", value = "vint32m1_t"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/source:typed-source-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_zeroing_requirement", value = "masked-standalone-reduction-zero-inactive-lanes-before-reduction"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_rt_scalar_cm_standalone_reduce_m2

// HEADER-DAG: tianchenrv.rvv.selected_variant: @rvv_pre_rt_scalar_cm_standalone_reduce_m2
// HEADER-DAG: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1
// HEADER-DAG: tianchenrv.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,src,acc,out,n
// HEADER-DAG: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_route_family_plan: rvv-standalone-reduction-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_source_vector_type: !tcrv_rvv.vector<i32, "m2">
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_source_vector_c_type: vint32m2_t
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_scalar_result_vector_type: !tcrv_rvv.vector<i32, "m1">
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_scalar_result_vector_c_type: vint32m1_t
// HEADER: void tcrv_emitc_pre_rt_scalar_cm_standalone_reduce_m2_kernel_rvv_pre_rt_scalar_cm_standalone_reduce_m2(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);
