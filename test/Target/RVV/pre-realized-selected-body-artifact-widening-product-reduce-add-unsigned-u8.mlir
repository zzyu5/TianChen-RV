// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed 's/source_signedness = "unsigned"/source_signedness = "signed"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-SIGN

// Pre-realized selected-body input for the bounded Stage 2 unsigned u8 product
// -> u16 product -> u32 widening reduction chain. The same typed pre-realized
// op carries source signedness and must realize through the RVV plugin before
// route planning.

module {
  tcrv.exec.kernel @pre_realized_u8_product_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_u8_product_reduce_rvv attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, memory_form = "unit-stride-widening-product-reduce-add", op_kind = "widening_product_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf4", source_sew = 8 : i64, source_signedness = "unsigned"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_u8_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_u8_product_reduce_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-unsigned-widening-product-reduce-add-case"}
      tcrv.exec.fallback @pre_realized_u8_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-unsigned-widening-product-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_u8_product_reduce_rvv
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<ui8, "mf4">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<ui8, "mf4">
// REALIZED: %[[PRODUCT:.*]] = tcrv_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "unsigned_widening_product"
// REALIZED-SAME: product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"
// REALIZED-SAME: -> !tcrv_rvv.vector<ui16, "mf2">
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.standalone_reduce %[[PRODUCT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "unsigned_widening_reduce_add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: -> !tcrv_rvv.vector<ui32, "m1">
// REALIZED: tcrv_rvv.store %{{.*}}, %[[REDUCED]], %[[VL]]

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-reduction.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_extension", value = "zero-extend-u8-to-u16-product"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_dtype", value = "u32"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_u8_product_reduce_rvv

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_u8_product_reduce_rvv
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.low_precision_primitive.source_signedness: unsigned
// HEADER: tianchenrv.rvv.low_precision_primitive.source_extension: zero-extend-u8-to-u16-product
// HEADER: tianchenrv.rvv.low_precision_primitive.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-contraction-leaf-profile.v1
// HEADER: void tcrv_emitc_pre_realized_u8_product_reduce_kernel_pre_realized_u8_product_reduce_rvv(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc, uint32_t *out, size_t n);

// STALE-SIGN: requires product_relation "signed-i8mf4xi8mf4-to-i16mf2" when source_signedness is "signed"
