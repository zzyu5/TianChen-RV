// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/c_name = "lower_bound"/s/role = "lower-bound-scalar-value"/role = "upper-bound-scalar-value"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-LOWER
// RUN: sed '/c_name = "lower_bound"/s/role = "lower-bound-scalar-value"/role = "upper-bound-scalar-value"/;/c_name = "upper_bound"/s/role = "upper-bound-scalar-value"/role = "lower-bound-scalar-value"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=SWAPPED-BOUNDS
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"/product_reduction_chain_relation = "metadata-product-reduction"/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-REDUCTION
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=DTYPE-CHAIN
// RUN: sed 's/#tcrv_rvv.policy<tail = agnostic, mask = agnostic>/#tcrv_rvv.policy<tail = undisturbed, mask = agnostic>/g' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-POLICY
// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-RESOURCE-FACTS
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.low_precision_resource.accumulator_dtype = "i32"/s//tcrv_rvv.low_precision_resource.accumulator_dtype = "i16"/' | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE-ACC
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.low_precision_resource.reduction_layout = "scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1"/s//tcrv_rvv.low_precision_resource.reduction_layout = "metadata-layout"/' | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-RESOURCE-REDUCTION-LAYOUT
// RUN: sed '/typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body/s/policy = /route_id = "rvv-i32m1", policy = /' %s | not tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-contraction-family-plan-validated/s//provider_supported_mirror:rvv-artifact-name-authority/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n/s//tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,lower_bound,scale,upper_bound,out,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr/s//lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '/rvv_selected_body_typed_compute_op/s/tcrv_rvv.gearbox_cross_region_handoff+//' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HANDOFF
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"/s//tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/dequant-splat\/clamped:float-e32m1/s//converted\/scaled\/clamped:float-e32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"/s//tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-product-only-leaf-profile.v1"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LEAF
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"/s//tcrv_rvv.reduction_accumulator_layout", value = "metadata-accumulator-layout"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ACC-LAYOUT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"/s//tcrv_rvv.reduction_result_layout", value = "store-vector-result-from-metadata"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESULT-LAYOUT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.product_vector_c_type", value = "vint16mf2_t"/s//tcrv_rvv.product_vector_c_type", value = "vint32m1_t"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCT-CTYPE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/__riscv_vwmul_vv_i16mf2/s//__riscv_vwmul_vv_i32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-WPROD
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/__riscv_vwredsum_vs_i16mf2_i32m1/s//__riscv_vredsum_vs_i32m1_i32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-WRED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}/s//tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}, {key = "tcrv_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}/s//tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}, {key = "tcrv_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT-SCALE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.lower_bound_c_type", value = "float"/s//tcrv_rvv.lower_bound_c_type", value = "double"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOWER-CTYPE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"/s//tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m2"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BOUND-SPLAT

// Pre-realized selected-body input for the bounded Stage 2 signed i8 product
// -> i16 product -> i32 reduction -> runtime-scale f32 dequantization
// -> runtime f32 lower/upper clamp/select -> f32 output store chain. The RVV
// plugin must realize the typed body before route planning; route ids,
// q-names, artifact names, and mirror metadata are not route authority.

module {
  tcrv.exec.kernel @pre_realized_body_product_reduce_dequant_clamp_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_product_reduce_dequant_clamp attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:lower", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:upper", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body %lhs, %rhs, %acc, %scale, %lower, %upper, %out, %n {accumulator_carry_boundary = "scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-clamped-dequantized-f32-vector-to-output-buffer", lower_predicate_kind = "slt", memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32", op_kind = "widening_product_reduce_dequant_clamp_f32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", source_lmul = "mf4", source_sew = 8 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_product_reduce_dequant_clamp {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_product_reduce_dequant_clamp
// REALIZED-SAME: tcrv_rvv.gearbox.producer_scope = "gearbox-scope:product-reduction"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1"
// REALIZED-SAME: tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 2 : i64
// REALIZED-SAME: tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,i8mf4-i16mf2-i32m1-f32m1,u1]"
// REALIZED: tcrv_rvv.vsetvl_region_marker %[[VL]]
// REALIZED-SAME: phase = "load-product-reduce"
// REALIZED-SAME: region_index = 1 : i64
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i8, "mf4">
// REALIZED: %[[PRODUCT:.*]] = tcrv_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
// REALIZED-SAME: -> !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.standalone_reduce %[[PRODUCT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: kind = "signed_widening_reduce_add"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[HANDOFF:.*]] = tcrv_rvv.gearbox_cross_region_handoff %[[REDUCED]], %[[VL]], %{{[0-9]+}}
// REALIZED-SAME: consumer_scope = "gearbox-scope:dequant-store"
// REALIZED-SAME: contract = "gearbox-product-reduce-to-dequant-cross-region-handoff.v1"
// REALIZED-SAME: runtime_avl_source = "runtime_abi:n"
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: tcrv_rvv.gearbox.consumer_scope = "gearbox-scope:dequant-store"
// REALIZED: tcrv_rvv.vsetvl_region_marker %[[VL]]
// REALIZED-SAME: phase = "dequant-store"
// REALIZED-SAME: region_index = 2 : i64
// REALIZED: %[[DEQUANT:.*]] = tcrv_rvv.dequantize %[[HANDOFF]], %{{[0-9]+}}, %[[VL]]
// REALIZED-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = tcrv_rvv.splat
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = tcrv_rvv.splat
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = tcrv_rvv.compare %[[DEQUANT]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[LOWER_CLAMPED:.*]] = tcrv_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[DEQUANT]], %[[VL]]
// REALIZED: %[[UPPER_MASK:.*]] = tcrv_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[CLAMPED:.*]] = tcrv_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED: tcrv_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_product_reduce_dequant_clamp_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce+tcrv_rvv.gearbox_cross_region_handoff+tcrv_rvv.dequantize+tcrv_rvv.compare+tcrv_rvv.select"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-widening-product-reduce-dequant-clamp-f32"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1;lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;out=output-buffer:out:abi|cdeq|store|f32m1|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-contraction-family-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf4"}
// PLAN-SAME: {key = "tcrv_rvv.product_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.product_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.product_vector_type", value = "!tcrv_rvv.vector<i16, \22mf2\22>"}
// PLAN-SAME: {key = "tcrv_rvv.product_vector_c_type", value = "vint16mf2_t"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_relation", value = "signed-i8mf4xi8mf4-to-i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.product_reduction_chain_relation", value = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_reduction_intrinsic", value = "__riscv_vwredsum_vs_i16mf2_i32m1"}
// PLAN-SAME: {key = "tcrv_rvv.scalar_seed_splat_intrinsic", value = "__riscv_vmv_v_x_i32m1"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: {key = "tcrv_rvv.scalar_result_runtime_boundary", value = "scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_broadcast_intrinsic", value = "__riscv_vfmv_v_f_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.lower_bound_role", value = "lower-bound-scalar-value"}
// PLAN-SAME: {key = "tcrv_rvv.upper_bound_role", value = "upper-bound-scalar-value"}
// PLAN-SAME: {key = "tcrv_rvv.lower_bound_c_type", value = "float"}
// PLAN-SAME: {key = "tcrv_rvv.upper_bound_c_type", value = "float"}
// PLAN-SAME: {key = "tcrv_rvv.bound_order", value = "lower-bound-before-upper-bound"}
// PLAN-SAME: {key = "tcrv_rvv.clamp_relation", value = "signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.select_layout", value = "clamp-lower-then-upper"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "tcrv_rvv.compare_intrinsic", value = "__riscv_vmflt_vv_f32m1_b32"}
// PLAN-SAME: {key = "tcrv_rvv.secondary_compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "tcrv_rvv.secondary_compare_intrinsic", value = "__riscv_vmflt_vv_f32m1_b32"}
// PLAN-SAME: {key = "tcrv_rvv.masked_merge_intrinsic", value = "__riscv_vmerge_vvm_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selected_candidate", value = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,i8mf4-i16mf2-i32m1-f32m1,u1]"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.selection_reason", value = "static-bounded-product-reduction-dequant-clamp-i8mf4-i16mf2-i32m1-f32m1-runtime-avl"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.memory_form", value = "unit-stride-widening-product-reduce-dequant-clamp-f32"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.producer_scope", value = "gearbox-scope:product-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.consumer_scope", value = "gearbox-scope:dequant-store"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_resource.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_product_reduce_dequant_clamp

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_product_reduce_dequant_clamp
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-product-reduce-dequant-clamp-f32-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1;lhs=lhs-input-buffer:lhs:abi|ld|wpl|i8mf4|hdr;rhs=rhs-input-buffer:rhs:abi|ld|wpr|i8mf4|hdr;acc=accumulator-input-buffer:acc:abi|seed|wred|i32|hdr;scale=dequant-scale-value:scale:abi|scale|f32|deq|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|up|splat|cmp|sel|hdr;out=output-buffer:out:abi|cdeq|store|f32m1|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER-DAG: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.target_leaf_profile: rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequant-clamp-leaf-profile.v1
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-contraction-family-plan-validated
// HEADER-DAG: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e8mf4,product:signed-e16mf2,seed:signed-i32,accumulator:signed-e32m1,dequant-splat/clamped:float-e32m1,scale:float,lower:float,upper:float
// HEADER-DAG: tianchenrv.rvv.source_sew: 8
// HEADER-DAG: tianchenrv.rvv.source_lmul: mf4
// HEADER-DAG: tianchenrv.rvv.product_sew: 16
// HEADER-DAG: tianchenrv.rvv.product_lmul: mf2
// HEADER-DAG: tianchenrv.rvv.product_vector_type: !tcrv_rvv.vector<i16, "mf2">
// HEADER-DAG: tianchenrv.rvv.product_vector_c_type: vint16mf2_t
// HEADER-DAG: tianchenrv.rvv.accumulator_sew: 32
// HEADER-DAG: tianchenrv.rvv.accumulator_lmul: m1
// HEADER-DAG: tianchenrv.rvv.result_sew: 32
// HEADER-DAG: tianchenrv.rvv.result_lmul: m1
// HEADER-DAG: tianchenrv.rvv.source_memory_form: unit-stride-load
// HEADER-DAG: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER-DAG: tianchenrv.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input
// HEADER-DAG: tianchenrv.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar
// HEADER-DAG: tianchenrv.rvv.widening_product_relation: signed-i8mf4xi8mf4-to-i16mf2
// HEADER-DAG: tianchenrv.rvv.product_reduction_chain_relation: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32
// HEADER-DAG: tianchenrv.rvv.widening_product_intrinsic: __riscv_vwmul_vv_i16mf2
// HEADER-DAG: tianchenrv.rvv.widening_reduction_intrinsic: __riscv_vwredsum_vs_i16mf2_i32m1
// HEADER-DAG: tianchenrv.rvv.scalar_seed_splat_intrinsic: __riscv_vmv_v_x_i32m1
// HEADER-DAG: tianchenrv.rvv.reduction_store_vl: 1
// HEADER-DAG: tianchenrv.rvv.standalone_reduction_scalar_result_runtime_boundary: scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1
// HEADER-DAG: tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER-DAG: tianchenrv.rvv.dequant_scale_role: dequant-scale-value
// HEADER-DAG: tianchenrv.rvv.dequant_scale_c_type: float
// HEADER-DAG: tianchenrv.rvv.dequant_scale_name: scale
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selected_candidate: rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequant-clamp-f32,i8mf4-i16mf2-i32m1-f32m1,u1]
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.selection_reason: static-bounded-product-reduction-dequant-clamp-i8mf4-i16mf2-i32m1-f32m1-runtime-avl
// HEADER-DAG: tianchenrv.rvv.low_precision_resource.memory_form: unit-stride-widening-product-reduce-dequant-clamp-f32
// HEADER-DAG: tianchenrv.rvv.gearbox_producer_scope: gearbox-scope:product-reduction
// HEADER-DAG: tianchenrv.rvv.gearbox_consumer_scope: gearbox-scope:dequant-store
// HEADER-DAG: tianchenrv.rvv.lower_bound_role: lower-bound-scalar-value
// HEADER-DAG: tianchenrv.rvv.upper_bound_role: upper-bound-scalar-value
// HEADER-DAG: tianchenrv.rvv.lower_bound_c_type: float
// HEADER-DAG: tianchenrv.rvv.upper_bound_c_type: float
// HEADER-DAG: tianchenrv.rvv.bound_order: lower-bound-before-upper-bound
// HEADER-DAG: tianchenrv.rvv.clamp_relation: signed-i8mf4xi8mf4-i32-reduction-scale-f32-clamp-lower-upper-to-f32m1
// HEADER-DAG: tianchenrv.rvv.select_layout: clamp-lower-then-upper
// HEADER-DAG: tianchenrv.rvv.compare_predicate_kind: slt
// HEADER-DAG: tianchenrv.rvv.secondary_compare_predicate_kind: slt
// HEADER: void tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-LOWER: lower bound scalar
// MISSING-LOWER-SAME: lower-bound-scalar-value

// SWAPPED-BOUNDS: lower bound scalar
// SWAPPED-BOUNDS-SAME: lower-bound-scalar-value

// MISSING-REDUCTION: product_reduction_chain_relation
// MISSING-REDUCTION-SAME: signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32

// DTYPE-CHAIN: requires typed product-reduction-dequant-clamp config

// UNSUPPORTED-POLICY: requires tail agnostic, mask agnostic policy

// MISSING-RESOURCE-FACTS: requires pass-produced low-precision direct-contraction resource fact
// MISSING-RESOURCE-FACTS-SAME: tcrv_rvv.low_precision_resource.candidate_set

// STALE-RESOURCE-ACC: cannot consume stale or unsupported low-precision direct-contraction resource fact
// STALE-RESOURCE-ACC-SAME: tcrv_rvv.low_precision_resource.accumulator_dtype
// STALE-RESOURCE-ACC-SAME: expected 'i32'
// STALE-RESOURCE-ACC-SAME: found 'i16'

// STALE-RESOURCE-REDUCTION-LAYOUT: cannot consume stale or unsupported low-precision direct-contraction resource fact
// STALE-RESOURCE-REDUCTION-LAYOUT-SAME: tcrv_rvv.low_precision_resource.reduction_layout
// STALE-RESOURCE-REDUCTION-LAYOUT-SAME: scalar-i32-local-carry-dot_acc_scalar-across-runtime-vl-chunks-final-f32-store.v1
// STALE-RESOURCE-REDUCTION-LAYOUT-SAME: metadata-layout

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: provider_supported_mirror
// STALE-PROVIDER-SAME: rvv-artifact-name-authority

// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: tcrv_rvv.runtime_abi_order
// STALE-ABI-SAME: lhs,rhs,acc,lower_bound,scale,upper_bound,out,n

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: tcrv_rvv.route_operand_binding_operands
// STALE-BINDING-SAME: must mirror
// STALE-BINDING-SAME: lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel

// STALE-HANDOFF: RVV construction manifest invalid
// STALE-HANDOFF: rvv_selected_body_typed_compute_op
// STALE-HANDOFF-SAME: tcrv_rvv.gearbox_cross_region_handoff

// STALE-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-HEADER: tcrv_rvv.required_header_declarations
// STALE-HEADER-SAME: must mirror
// STALE-HEADER-SAME: stddef.h,stdint.h

// STALE-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-CTYPE: tcrv_rvv.c_type_mapping
// STALE-CTYPE-SAME: must mirror
// STALE-CTYPE-SAME: converted/scaled/clamped:float-e32m1

// STALE-LEAF: RVV materialized EmitC target artifact bridge failed
// STALE-LEAF: tcrv_rvv.target_leaf_profile
// STALE-LEAF-SAME: product-only-leaf-profile

// STALE-ACC-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-ACC-LAYOUT: tcrv_rvv.reduction_accumulator_layout
// STALE-ACC-LAYOUT-SAME: metadata-accumulator-layout

// STALE-RESULT-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-RESULT-LAYOUT: tcrv_rvv.reduction_result_layout
// STALE-RESULT-LAYOUT-SAME: store-vector-result-from-metadata

// STALE-PRODUCT-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-PRODUCT-CTYPE: tcrv_rvv.product_vector_c_type
// STALE-PRODUCT-CTYPE-SAME: vint32m1_t

// STALE-WPROD: RVV materialized EmitC target artifact bridge failed
// STALE-WPROD: tcrv_rvv.widening_product_intrinsic
// STALE-WPROD-SAME: __riscv_vwmul_vv_i32m1

// STALE-WRED: RVV materialized EmitC target artifact bridge failed
// STALE-WRED: tcrv_rvv.widening_reduction_intrinsic
// STALE-WRED-SAME: __riscv_vredsum_vs_i32m1_i32m1

// STALE-DEQUANT: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT: tcrv_rvv.dequantize_convert_intrinsic
// STALE-DEQUANT-SAME: standalone vector dequant convert

// STALE-DEQUANT-SCALE: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT-SCALE: tcrv_rvv.dequantize_scale_intrinsic
// STALE-DEQUANT-SCALE-SAME: standalone vector dequant scale

// STALE-LOWER-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-LOWER-CTYPE: tcrv_rvv.lower_bound_c_type
// STALE-LOWER-CTYPE-SAME: double

// STALE-BOUND-SPLAT: RVV materialized EmitC target artifact bridge failed
// STALE-BOUND-SPLAT: tcrv_rvv.rhs_broadcast_intrinsic
// STALE-BOUND-SPLAT-SAME: __riscv_vfmv_v_f_f32m2
