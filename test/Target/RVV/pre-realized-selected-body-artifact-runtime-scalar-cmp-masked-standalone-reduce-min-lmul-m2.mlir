// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_rt_scalar_cm_standalone_reduce_min_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_pre_rt_scalar_cm_standalone_reduce_min_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:scalar-output", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_min", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_pre_rt_scalar_cm_standalone_reduce_min_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-standalone-reduce-min-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @rvv_pre_rt_scalar_cm_standalone_reduce_min_m2
// REALIZED-SAME: sew = 32
// REALIZED: weft_rvv.splat
// REALIZED: weft_rvv.compare
// REALIZED-SAME: kind = "sle"
// REALIZED: weft_rvv.masked_standalone_reduce
// REALIZED-SAME: kind = "min"
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.standalone_reduce
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_rt_scalar_cm_standalone_reduce_min_m2

// HEADER-DAG: weft.rvv.selected_variant: @rvv_pre_rt_scalar_cm_standalone_reduce_min_m2
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_rt_scalar_cm_standalone_reduce_min_m2_kernel_rvv_pre_rt_scalar_cm_standalone_reduce_min_m2(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);
