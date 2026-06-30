// Production-export coverage for the NON-deferred WIDE widening-product-reduce-
// dequantize body -- the exact shape the Track-B dequant front door auto-constructs
// (lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp) at VLEN128:
//   load i8m2 x2 -> widening_product i16m4 -> per-iteration vwredsum_i16m4_i32m1
//   -> dequantize i32m1->f32m1 -> store  (NO i32m8 deferred accumulate).
//
// Before this coverage the export layer (`--tcrv-materialize-emission-plans`)
// hardcoded the NARROW i8mf4 -> i16mf2 product-reduction triple and REJECTED this
// wide body with "product-reduction lhs source vector LMUL 'm2' to match selected
// config LMUL 'mf4'". The front-door lit only ran `--tcrv-rvv-lower-to-emitc`, so
// this body's production export was previously untested. The export path is now
// capability-driven: it derives the strip LMUL STRUCTURALLY from the realized body
// (I5) and admits the wide i8m2 -> i16m4 -> i32m1 chain as a parallel config,
// mirroring the proven deferred-wide path.
//
// DESIGN SPLIT (narrow ROUTE IDENTITY, wide REALIZED BODY) -- identical to the
// deferred-wide route: the route IDENTITY stays narrow (target_leaf_profile,
// c-type mapping, candidate_set / selected_candidate, multiplicand roles all
// i8mf4-i16mf2), while the realized PRIMITIVE facts + emitted intrinsics carry the
// wide i8m2/i16m4 strip. The low_precision_resource.* facts on the with_vl are the
// N3 evidence the non-deferred dequant op-kind requires; this hand-written body
// carries the wide-primitive/narrow-identity fact set (the dequant front door does
// NOT yet stamp these facts -- a separate front-door completeness gap).
//
// HOST-ONLY: correctness of the i8m2 -> i16m4 -> i32m1 chain is covered by the
// existing host scalar oracle; this fixture asserts only the PLAN facts + emitted
// intrinsics. NO board / NO perf claim.

// The PLAN carries the WIDE primitive LMUL (source m2 / product m4 / accumulator
// m1) while the route identity (leaf profile) stays NARROW i8mf4-i16mf2.
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN

// The EmitC emits the WIDE intrinsics (vsetvl_e8m2 / vle8_v_i8m2 / vwmul_vv_i16m4 /
// vwredsum_vs_i16m4_i32m1), NOT the narrow i8mf4/i16mf2 forms.
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// PLAN-DAG: "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.source_lmul", value = "m2"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.product_lmul", value = "m4"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.accumulator_lmul", value = "m1"
// The route IDENTITY stays narrow (the wide strip is internal to the realized body).
// PLAN-DAG: "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"

// EMITC: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
// EMITC: call_opaque "__riscv_vsetvl_e8m2"
// EMITC-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMITC-NOT: call_opaque "__riscv_vsetvl_e8mf4"
// EMITC: call_opaque "__riscv_vle8_v_i8m2"
// EMITC-NOT: call_opaque "__riscv_vle8_v_i8mf4"
// EMITC: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC-NOT: call_opaque "__riscv_vwmul_vv_i16mf2"
// EMITC: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC-NOT: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// EMITC: call_opaque "__riscv_vfmv_v_f_f32m1"
// EMITC: call_opaque "__riscv_vse32_v_f32m1"
// EMITC: return

module {
  tcrv.exec.kernel @rvv_widening_dot_reduce_dequantize_i8_from_vector_source {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_widening_dot_reduce_dequantize_i8 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.gearbox_selected_integer_core_lmul = "m2", tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %1 = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %5 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:n", role = "runtime-element-count"} : index
      %6 = tcrv_rvv.setvl %5 {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "dispatch case", selected_variant = @rvv_widening_dot_reduce_dequantize_i8, sew = 8 : i64, source_kernel = "rvv_widening_dot_reduce_dequantize_i8_from_vector_source", status = "selected-lowering-boundary", tcrv_rvv.gearbox.consumer_scope = "gearbox-scope:dequant-store", tcrv_rvv.gearbox.producer_scope = "gearbox-scope:product-reduction", tcrv_rvv.low_precision_resource.accumulator_count = 2 : i64, tcrv_rvv.low_precision_resource.accumulator_dtype = "i32", tcrv_rvv.low_precision_resource.accumulator_emul = "m1", tcrv_rvv.low_precision_resource.accumulator_lmul = "m1", tcrv_rvv.low_precision_resource.accumulator_sew = 32 : i64, tcrv_rvv.low_precision_resource.candidate_count = 3 : i64, tcrv_rvv.low_precision_resource.candidate_set = "rvv-low-precision-direct-contraction-resource-candidate-set.v4[i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,u2-grouped-tail-safe,signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1:u1-unpack-required]", tcrv_rvv.low_precision_resource.dequant_phase = "dequant-store", tcrv_rvv.low_precision_resource.dequant_region_index = 3 : i64, tcrv_rvv.low_precision_resource.effective_element_width = 8 : i64, tcrv_rvv.low_precision_resource.legal_candidate_count = 3 : i64, tcrv_rvv.low_precision_resource.legality = "legal", tcrv_rvv.low_precision_resource.legality_scope = "typed-low-precision-product-reduction-dequant-resource-legality.v1", tcrv_rvv.low_precision_resource.mask_policy = "agnostic", tcrv_rvv.low_precision_resource.memory_form = "unit-stride-widening-product-reduce-dequantize-f32", tcrv_rvv.low_precision_resource.operand_form = "unpacked-byte-elements", tcrv_rvv.low_precision_resource.packing_layout = "one-element-per-byte", tcrv_rvv.low_precision_resource.peak_live_vector_groups = 7 : i64, tcrv_rvv.low_precision_resource.planning_contract = "rvv-low-precision-production-resource-planning-contract.v1", tcrv_rvv.low_precision_resource.primitive_accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", tcrv_rvv.low_precision_resource.primitive_chain_contract = "rvv-low-precision-widening-reduction-primitive-facts.v1", tcrv_rvv.low_precision_resource.primitive_chain_kind = "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1", tcrv_rvv.low_precision_resource.primitive_contract = "rvv-low-precision-widening-primitive-facts.v1", tcrv_rvv.low_precision_resource.primitive_kind = "signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-reduction-f32m1-dequant.v1", tcrv_rvv.low_precision_resource.primitive_product_reduction_chain_relation = "signed-i8m2xi8m2-to-i16m4-reduce-plus-i32-scalar-to-i32", tcrv_rvv.low_precision_resource.primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16m4_i32m1", tcrv_rvv.low_precision_resource.primitive_reduction_store_vl = "1", tcrv_rvv.low_precision_resource.primitive_result_layout = "store-standalone-reduction-lane0-to-output-scalar", tcrv_rvv.low_precision_resource.primitive_scalar_seed_splat_intrinsic = "__riscv_vmv_v_x_i32m1", tcrv_rvv.low_precision_resource.primitive_source_extension = "sign-extend-i8-to-i16-product", tcrv_rvv.low_precision_resource.primitive_source_load = "unit-stride-byte-load", tcrv_rvv.low_precision_resource.primitive_widening_product_intrinsic = "__riscv_vwmul_vv_i16m4", tcrv_rvv.low_precision_resource.primitive_widening_product_relation = "signed-i8m2xi8m2-to-i16m4", tcrv_rvv.low_precision_resource.product_dtype = "i16", tcrv_rvv.low_precision_resource.product_emul = "mf2", tcrv_rvv.low_precision_resource.product_lmul = "m4", tcrv_rvv.low_precision_resource.product_phase = "tail-product-reduce", tcrv_rvv.low_precision_resource.product_region_index = 2 : i64, tcrv_rvv.low_precision_resource.product_sew = 16 : i64, tcrv_rvv.low_precision_resource.realization_decision = "consume-low-precision-u2-three-vsetvl-region-budget-7of32.v1", tcrv_rvv.low_precision_resource.realization_producer = "rvv-plugin-local-selected-body-realization-resource-consumer.v1", tcrv_rvv.low_precision_resource.realized_peak_live_vector_groups = 7 : i64, tcrv_rvv.low_precision_resource.realized_unroll_factor = 2 : i64, tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 3 : i64, tcrv_rvv.low_precision_resource.reduction_candidate_fact = "resource-candidate-widening-reduction:signed-i8m2xi8m2-to-i16m4-reduce-plus-i32-scalar-to-i32:__riscv_vwredsum_vs_i16m4_i32m1:store-vl=1", tcrv_rvv.low_precision_resource.reduction_layout = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", tcrv_rvv.low_precision_resource.rejection_reason = "none", tcrv_rvv.low_precision_resource.result_dtype = "f32", tcrv_rvv.low_precision_resource.result_lmul = "m1", tcrv_rvv.low_precision_resource.result_sew = 32 : i64, tcrv_rvv.low_precision_resource.runtime_abi_order = "lhs,rhs,acc,scale,out,n", tcrv_rvv.low_precision_resource.runtime_avl_source = "runtime_abi:n", tcrv_rvv.low_precision_resource.selected_candidate = "rvv-low-precision-direct-contraction-resource-candidate.v1[product-reduction-dequantize-f32,i8mf4-i16mf2-i32m1-f32m1,u2-grouped]", tcrv_rvv.low_precision_resource.selected_candidate_index = 2 : i64, tcrv_rvv.low_precision_resource.selection_reason = "static-bounded-product-reduction-dequant-i8mf4-i16mf2-i32m1-f32m1-u2-grouped-tail-safe-runtime-avl", tcrv_rvv.low_precision_resource.source_dtype = "i8", tcrv_rvv.low_precision_resource.source_lmul = "m2", tcrv_rvv.low_precision_resource.source_sew = 8 : i64, tcrv_rvv.low_precision_resource.source_signedness = "signed", tcrv_rvv.low_precision_resource.storage_element_width = 8 : i64, tcrv_rvv.low_precision_resource.tail_policy = "agnostic", tcrv_rvv.low_precision_resource.unpack_intent = "none-direct-widening-product", tcrv_rvv.low_precision_resource.unroll_factor = 2 : i64, tcrv_rvv.low_precision_resource.vector_register_budget = 32 : i64, tcrv_rvv.low_precision_resource.vsetvl_region_count = 3 : i64, tcrv_rvv.low_precision_resource.widening_product_candidate_fact = "resource-candidate-widening-product:signed-i8m2xi8m2-to-i16m4:__riscv_vwmul_vv_i16m4", tcrv_rvv.low_precision_resource.widening_product_extension_policy = "source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2", tcrv_rvv.low_precision_resource.widening_product_multiplicand_roles = "lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %8 = tcrv_rvv.load %1, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %9 = tcrv_rvv.widening_product %7, %8, %6 {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %10 = tcrv_rvv.standalone_reduce %9, %2, %6 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %11 = tcrv_rvv.dequantize %10, %3, %6 {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %4, %11, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_widening_dot_reduce_dequantize_i8 {origin = "rvv-plugin", policy = "rvv-widening-dot-reduce-dequantize-source-front-door-case"}
      tcrv.exec.fallback @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}
