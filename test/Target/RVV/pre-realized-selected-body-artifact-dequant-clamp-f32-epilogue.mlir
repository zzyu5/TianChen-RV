// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_dequant_clamp_f32_epilogue_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_rvv_dequant_clamp_f32_epilogue attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:n", role = "runtime-element-count"} : index
      weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_rvv_dequant_clamp_f32_epilogue {origin = "rvv-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-case"}
      weft.exec.fallback @pre_realized_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_rvv_dequant_clamp_f32_epilogue
// REALIZED: %[[SRC:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// REALIZED: %[[DEQ:.*]] = weft_rvv.dequantize %[[SRC]]
// REALIZED-SAME: kind = "i32_to_f32_scaled"
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = weft_rvv.compare %[[DEQ]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[LOWER_CLAMPED:.*]] = weft_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[DEQ]], %[[VL]]
// REALIZED: %[[UPPER_MASK:.*]] = weft_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[CLAMPED:.*]] = weft_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED: weft_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: target = @pre_realized_rvv_dequant_clamp_f32_epilogue

// HEADER: weft.rvv.selected_variant: @pre_realized_rvv_dequant_clamp_f32_epilogue
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_dequant_clamp_f32_epilogue_kernel_pre_realized_rvv_dequant_clamp_f32_epilogue(const int32_t *lhs, float scale, float lower_bound, float upper_bound, float *out, size_t n);













