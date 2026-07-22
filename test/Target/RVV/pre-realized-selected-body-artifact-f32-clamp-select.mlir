// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_f32_clamp_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_rvv_f32_clamp_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_rvv_f32_clamp_select {origin = "rvv-plugin", policy = "pre-realized-f32-clamp-select-case"}
      weft.exec.fallback @pre_realized_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-f32-clamp-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_f32_clamp_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_rvv_f32_clamp_select
// REALIZED: %[[INPUT:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = weft_rvv.compare %[[INPUT]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<f32, "m1">
// REALIZED: %[[LOWER_CLAMPED:.*]] = weft_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[INPUT]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER_MASK:.*]] = weft_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<f32, "m1">
// REALIZED: %[[CLAMPED:.*]] = weft_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: weft_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: weft_rvv.typed_f32_clamp_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: target = @pre_realized_rvv_f32_clamp_select

// HEADER: weft.rvv.selected_variant: @pre_realized_rvv_f32_clamp_select
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float *input, float lower_bound, float upper_bound, float *out, size_t n);









