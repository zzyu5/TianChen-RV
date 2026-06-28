// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @missing_scale_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires runtime scale operand to bind runtime ABI role 'dequant-scale-value'}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @swapped_lower_bound_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires lower bound scalar operand to bind runtime ABI role 'lower-bound-scalar-value'}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @ambiguous_bound_roles {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires upper bound scalar operand to bind runtime ABI role 'upper-bound-scalar-value'}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @source_dtype_mismatch {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires lhs operand C type 'const int32_t *' for the bounded dequant-clamp epilogue route}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @unsupported_lmul {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized dequant-clamp epilogue data config to be SEW32 LMUL m1}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m2", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @unsupported_policy {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body dequant-clamp epilogue hook}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @route_id_authority {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "rvv-generic-dequant-clamp-f32-epilogue-emitc-route", scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @artifact_name_authority {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{only accepts pre-realization attributes}}
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {artifact_name = "dequant_clamp_f32_epilogue.c", bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
