// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed 's/source_signedness = "unsigned"/source_signedness = "signed"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-SIGN

// Pre-realized selected-body input for the bounded Stage 2 unsigned u8 product
// -> u16 product -> u32 widening reduction chain. The same typed pre-realized
// op carries source signedness and must realize through the RVV plugin before
// route planning.

module {
  weft.exec.kernel @pre_realized_u8_product_reduce_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_u8_product_reduce_rvv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unsigned-widening-product-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, memory_form = "unit-stride-widening-product-reduce-add", op_kind = "widening_product_reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf4", source_sew = 8 : i64, source_signedness = "unsigned"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_u8_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_u8_product_reduce_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-unsigned-widening-product-reduce-add-case"}
      weft.exec.fallback @pre_realized_u8_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-unsigned-widening-product-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_u8_product_reduce_rvv
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<ui8, "mf4">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<ui8, "mf4">
// REALIZED: %[[PRODUCT:.*]] = weft_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "unsigned_widening_product"
// REALIZED-SAME: product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"
// REALIZED-SAME: -> !weft_rvv.vector<ui16, "mf2">
// REALIZED: %[[REDUCED:.*]] = weft_rvv.standalone_reduce %[[PRODUCT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "unsigned_widening_reduce_add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: -> !weft_rvv.vector<ui32, "m1">
// REALIZED: weft_rvv.store %{{.*}}, %[[REDUCED]], %[[VL]]

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product+weft_rvv.standalone_reduce"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-product-u32m1-reduction.v1"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_load", value = "unit-stride-byte-load"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.source_extension", value = "zero-extend-u8-to-u16-product"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.result_dtype", value = "u32"}
// PLAN-SAME: {key = "weft_rvv.low_precision_primitive.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_u8_product_reduce_rvv

// HEADER: weft.rvv.selected_variant: @pre_realized_u8_product_reduce_rvv
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-add-callable-c-abi.v1
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_signedness: unsigned
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.source_extension: zero-extend-u8-to-u16-product
// HEADER: weft.rvv.low_precision_primitive.payload_mirror.runtime_avl_source: runtime_abi:n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-u8mf4-u16mf2-u32m1-product-reduction-contraction-leaf-profile.v1
// HEADER: void weft_emitc_pre_realized_u8_product_reduce_kernel_pre_realized_u8_product_reduce_rvv(const uint8_t *lhs, const uint8_t *rhs, const uint32_t *acc, uint32_t *out, size_t n);

// STALE-SIGN: requires product_relation "signed-i8mf4xi8mf4-to-i16mf2" when source_signedness is "signed"
