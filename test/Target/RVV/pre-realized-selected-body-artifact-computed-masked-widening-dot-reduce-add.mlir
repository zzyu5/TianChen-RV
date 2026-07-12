// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed 's/source_lmul = "mf2"/route_id = "rvv-i32m1", source_lmul = "mf2"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH
// RUN: sed 's/mask_source = "compare-produced-mask-same-vl-scope"/mask_source = "runtime_abi:mask"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-MASK-PROVENANCE

// Pre-realized selected-body input for one bounded Stage 2 signed computed-mask
// widening dot-product reduction slice. The RVV plugin must derive compare
// mask provenance, i16mf2 dot inputs, i32 scalar seed/result boundary, policy,
// route, and ABI facts from typed body/config/runtime facts.

module {
  weft.exec.kernel @pre_realized_body_masked_widening_dot_reduce_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_masked_widening_dot_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:cmp-lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:cmp-rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:lhs", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:rhs", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_masked_widening_dot_reduce_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-masked-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_widening_dot_reduce_add
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_widening_dot_reduce %[[MASK]], %[[LHS]], %[[RHS]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"
// REALIZED-SAME: kind = "signed_masked_widening_dot_reduce_add"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-dot-reduction-lane0-to-output-scalar"
// REALIZED-SAME: !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_widening_dot_reduce"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-unit-stride-widening-dot-reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_widening_dot_reduce.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_widening_dot_reduce.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;dot_lhs=dot-lhs-input-buffer:lhs:abi|ld|mlhs|i16|hdr;dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_zeroing_requirement", value = "masked-widening-products-zero-inactive-lanes-before-reduction"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_result_layout", value = "store-dot-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "weft_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1"}
// PLAN-SAME: {key = "weft_rvv.masked_widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1_m"}
// PLAN-SAME: {key = "weft_rvv.widening_dot_reduction_store_vl", value = "1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_widening_dot_reduce_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_masked_widening_dot_reduce_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n
// HEADER: weft.rvv.compare_predicate_kind: slt
// HEADER: weft.rvv.memory_form: computed-mask-unit-stride-widening-dot-reduce
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_widening_dot_reduce.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_widening_dot_reduce.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;dot_lhs=dot-lhs-input-buffer:lhs:abi|ld|mlhs|i16|hdr;dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: weft.rvv.inactive_lane_zeroing_requirement: masked-widening-products-zero-inactive-lanes-before-reduction
// HEADER: void weft_emitc_pre_realized_body_masked_widening_dot_reduce_add_kernel_pre_realized_body_rvv_masked_widening_dot_reduce_add(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-AUTH: does not accept authority metadata attribute '"route_id"'
// MISSING-MASK-PROVENANCE: currently supports only mask_source "compare-produced-mask-same-vl-scope"
