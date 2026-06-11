// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-rvv-materialize-gearbox-schedules | FileCheck %s
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.selected_candidate = "artifact-name-derived-resource-candidate", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.planning_contract = "artifact-derived-resource-planning-contract", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE-PLANNING
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.operand_form = "packed-i4-nibbles", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-PACKED
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i32m1_i32m1", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE-PRIMITIVE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.widening_product_candidate_fact = "metadata-only-widening-product-candidate", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE-WIDENING-CANDIDATE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.reduction_candidate_fact = "metadata-only-reduction-candidate", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE-REDUCTION-CANDIDATE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/policy = /tcrv_rvv.low_precision_resource.primitive_source_extension = "metadata-only-source-extension", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE-EXTENSION
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-RESOURCE

// CHECK-LABEL: tcrv.exec.kernel @rvv_gearbox_widening_product_reduce_dequantize_f32_kernel
// CHECK: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// CHECK-SAME: tcrv_rvv.low_precision_resource.candidate_set = "rvv-low-precision-direct-contraction-resource-candidate-set.v4[i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,u2-grouped-tail-safe,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1:u1-unpack-required]"
// CHECK-SAME: tcrv_rvv.low_precision_resource.effective_element_width = 8 : i64
// CHECK-SAME: tcrv_rvv.low_precision_resource.memory_form = "unit-stride-widening-product-reduce-dequantize-f32"
// CHECK-SAME: tcrv_rvv.low_precision_resource.operand_form = "unpacked-byte-elements"
// CHECK-SAME: tcrv_rvv.low_precision_resource.packing_layout = "one-element-per-byte"
// CHECK-SAME: tcrv_rvv.low_precision_resource.planning_contract = "rvv-low-precision-production-resource-planning-contract.v1"
// CHECK-SAME: tcrv_rvv.low_precision_resource.primitive_contract = "rvv-low-precision-widening-primitive-facts.v1"
// CHECK-SAME: tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"
// CHECK-SAME: tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK-SAME: tcrv_rvv.low_precision_resource.primitive_source_extension = "sign-extend-i8-to-i16-product"
// CHECK-SAME: tcrv_rvv.low_precision_resource.primitive_source_load = "unit-stride-byte-load"
// CHECK-SAME: tcrv_rvv.low_precision_resource.product_dtype = "i16"
// CHECK-SAME: tcrv_rvv.low_precision_resource.reduction_candidate_fact = "resource-candidate-widening-reduction:signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32:__riscv_vwredsum_vs_i16mf2_i32m1:store-vl=1"
// CHECK-SAME: tcrv_rvv.low_precision_resource.result_dtype = "f32"
// CHECK-SAME: tcrv_rvv.low_precision_resource.runtime_abi_order = "lhs,rhs,acc,scale,out,n"
// CHECK-SAME: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]"
// CHECK-SAME: tcrv_rvv.low_precision_resource.selection_reason = "static-bounded-product-reduction-dequant-i8mf4-i16mf2-i32m1-f32m1-u2-grouped-tail-safe-runtime-avl"
// CHECK-SAME: tcrv_rvv.low_precision_resource.source_dtype = "i8"
// CHECK-SAME: tcrv_rvv.low_precision_resource.source_signedness = "signed"
// CHECK-SAME: tcrv_rvv.low_precision_resource.storage_element_width = 8 : i64
// CHECK-SAME: tcrv_rvv.low_precision_resource.unpack_intent = "none-direct-widening-product"
// CHECK-SAME: tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64
// CHECK-SAME: tcrv_rvv.low_precision_resource.widening_product_candidate_fact = "resource-candidate-widening-product:signed-i8mf4xi8mf4-to-i16mf2:__riscv_vwmul_vv_i16mf2"
// CHECK-SAME: tcrv_rvv.low_precision_resource.widening_product_extension_policy = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2"
// CHECK-SAME: tcrv_rvv.low_precision_resource.widening_product_multiplicand_roles = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"

// STALE: cannot match explicit selected candidate
// STALE-SAME: artifact-name-derived-resource-candidate
// STALE-SAME: provider-owned candidate set

// STALE-PLANNING: RVV Gearbox pass found stale schedule fact
// STALE-PLANNING-SAME: tcrv_rvv.low_precision_resource.planning_contract
// STALE-PLANNING-SAME: expected 'rvv-low-precision-production-resource-planning-contract.v1'
// STALE-PLANNING-SAME: artifact-derived-resource-planning-contract

// UNSUPPORTED-PACKED: RVV Gearbox pass found stale schedule fact
// UNSUPPORTED-PACKED-SAME: tcrv_rvv.low_precision_resource.operand_form
// UNSUPPORTED-PACKED-SAME: expected 'unpacked-byte-elements'
// UNSUPPORTED-PACKED-SAME: packed-i4-nibbles

// STALE-PRIMITIVE: RVV Gearbox pass found stale schedule fact
// STALE-PRIMITIVE-SAME: tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic
// STALE-PRIMITIVE-SAME: expected '__riscv_vwredsum_vs_i16mf2_i32m1'
// STALE-PRIMITIVE-SAME: __riscv_vwredsum_vs_i32m1_i32m1

// STALE-WIDENING-CANDIDATE: RVV Gearbox pass found stale schedule fact
// STALE-WIDENING-CANDIDATE-SAME: tcrv_rvv.low_precision_resource.widening_product_candidate_fact
// STALE-WIDENING-CANDIDATE-SAME: resource-candidate-widening-product:signed-i8mf4xi8mf4-to-i16mf2:__riscv_vwmul_vv_i16mf2
// STALE-WIDENING-CANDIDATE-SAME: metadata-only-widening-product-candidate

// STALE-REDUCTION-CANDIDATE: RVV Gearbox pass found stale schedule fact
// STALE-REDUCTION-CANDIDATE-SAME: tcrv_rvv.low_precision_resource.reduction_candidate_fact
// STALE-REDUCTION-CANDIDATE-SAME: resource-candidate-widening-reduction:signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32:__riscv_vwredsum_vs_i16mf2_i32m1:store-vl=1
// STALE-REDUCTION-CANDIDATE-SAME: metadata-only-reduction-candidate

// STALE-SOURCE-EXTENSION: RVV Gearbox pass found stale schedule fact
// STALE-SOURCE-EXTENSION-SAME: tcrv_rvv.low_precision_resource.primitive_source_extension
// STALE-SOURCE-EXTENSION-SAME: expected 'sign-extend-i8-to-i16-product'
// STALE-SOURCE-EXTENSION-SAME: metadata-only-source-extension

// UNSUPPORTED-RESOURCE: requires typed product-reduction-dequantization config
// UNSUPPORTED-RESOURCE-SAME: source SEW8 LMUL mf4

module {
  tcrv.exec.kernel @rvv_gearbox_widening_product_reduce_dequantize_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_gearbox_widening_product_reduce_dequantize_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
