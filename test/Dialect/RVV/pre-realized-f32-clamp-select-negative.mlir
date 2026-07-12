// RUN: weft-opt %s --split-input-file --verify-diagnostics

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_lower_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_lower_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires lower bound scalar operand to bind runtime ABI role 'lower-bound-scalar-value'}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_upper_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_upper_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires upper bound scalar operand to bind runtime ABI role 'upper-bound-scalar-value'}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_input_dtype {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_input_dtype attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires input operand C type 'const float *' for the bounded f32 clamp/select route}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_bound_order {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_bound_order attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only bound_order "lower-bound-before-upper-bound"}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "upper-bound-before-lower-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_config {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized f32 clamp/select data config to be SEW32 LMUL m1}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m2", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_policy {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body f32 clamp/select hook}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = undisturbed, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_authority_attr {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_f32_clamp_select_bad_authority_attr attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "script-derived-f32-clamp-select", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}
