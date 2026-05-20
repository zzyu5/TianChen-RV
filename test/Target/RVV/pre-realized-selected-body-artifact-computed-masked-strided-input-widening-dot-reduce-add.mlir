// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed computed-mask
// runtime-strided-input widening dot-product reduction slice. The RVV plugin
// must carry compare mask provenance, i16mf2 source element strides, i32 scalar
// seed/result boundary, policy, route, and ABI facts from typed body/config/
// runtime SSA facts.

module {
  tcrv.exec.kernel @pre_realized_masked_strided_dot_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_computed_mask_strided_input_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:lhs", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:rhs", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:n", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-computed-mask-strided-input-dot:rhs-stride", role = "rhs-input-stride"} : index
      tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-strided-input-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_computed_mask_strided_input_dot {origin = "rvv-plugin", policy = "pre-realized-computed-mask-strided-input-dot-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-computed-mask-strided-input-dot-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @rvv_computed_mask_strided_input_dot
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[LHS:.*]] = tcrv_rvv.strided_load
// REALIZED-SAME: !tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.strided_load
// REALIZED-SAME: !tcrv_rvv.vector<i16, "mf2">
// REALIZED-NOT: tcrv_rvv.load {{.*}}!tcrv_rvv.vector<i16, "mf2">
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[SUM:.*]] = tcrv_rvv.masked_widening_dot_reduce %[[MASK]], %[[LHS]], %[[RHS]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"
// REALIZED-SAME: kind = "signed_masked_widening_dot_reduce_add"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-dot-reduction-lane0-to-output-scalar"
// REALIZED-SAME: !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_strided_input_widening_dot_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_widening_dot_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-strided-input-widening-dot-reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.lhs_stride_source", value = "runtime_abi:lhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.rhs_stride_source", value = "runtime_abi:rhs_stride"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_result_layout", value = "store-dot-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_relation", value = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1"}
// PLAN-SAME: {key = "tcrv_rvv.masked_widening_product_intrinsic", value = "__riscv_vwmul_vv_i32m1_m"}
// PLAN-SAME: {key = "tcrv_rvv.strided_load_intrinsic", value = "__riscv_vlse16_v_i16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_dot_reduction_store_vl", value = "1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_computed_mask_strided_input_dot

// HEADER: tianchenrv.rvv.selected_variant: @rvv_computed_mask_strided_input_dot
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-strided-input-widening-dot-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride
// HEADER: tianchenrv.rvv.memory_form: computed-mask-strided-input-widening-dot-reduce
// HEADER: tianchenrv.rvv.strided_memory_layout: unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi
// HEADER: tianchenrv.rvv.lhs_stride_source: runtime_abi:lhs_stride
// HEADER: tianchenrv.rvv.rhs_stride_source: runtime_abi:rhs_stride
// HEADER: tianchenrv.rvv.source_memory_form: strided-load
// HEADER: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER: tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: tianchenrv.rvv.widening_dot_relation: signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32
// HEADER: void tcrv_emitc_pre_realized_masked_strided_dot_kernel_rvv_computed_mask_strided_input_dot(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);
