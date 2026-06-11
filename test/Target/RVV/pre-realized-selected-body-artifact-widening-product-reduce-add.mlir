// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed 's/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE

// Pre-realized selected-body input for the bounded Stage 2 signed i8 product
// -> i16 product -> i32 widening reduction chain. The RVV plugin must consume
// typed primitive facts into an explicit tcrv_rvv body before route planning.

module {
  tcrv.exec.kernel @pre_realized_body_product_reduce_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_product_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, memory_form = "unit-stride-widening-product-reduce-add", op_kind = "widening_product_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_product_reduce_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_product_reduce_add
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[PRODUCT:.*]] = tcrv_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "signed_widening_product"
// REALIZED-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
// REALIZED-SAME: -> !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.standalone_reduce %[[PRODUCT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "signed_widening_reduce_add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store %{{.*}}, %[[REDUCED]], %[[VL]]

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.contract", value = "rvv-low-precision-widening-primitive-facts.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "signed"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_extension", value = "sign-extend-i8-to-i16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.source_signedness: signed
// HEADER: tianchenrv.rvv.low_precision_primitive.source_extension: sign-extend-i8-to-i16-product
// HEADER: tianchenrv.rvv.low_precision_primitive.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-product-reduction-contraction-leaf-profile.v1
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_add_kernel_pre_realized_body_rvv_product_reduce_add(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-SOURCE: requires typed product-reduction config to match signed source SEW8 LMUL mf4
