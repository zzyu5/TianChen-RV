// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_lower_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_lower_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires lower bound scalar operand to bind runtime ABI role 'lower-bound-scalar-value'}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_upper_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_upper_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires upper bound scalar operand to bind runtime ABI role 'upper-bound-scalar-value'}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_input_dtype {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_input_dtype attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires input operand C type 'const float *' for the bounded f32 clamp/select route}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_bound_order {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_bound_order attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only bound_order "lower-bound-before-upper-bound"}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "upper-bound-before-lower-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_config {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized f32 clamp/select data config to be SEW32 LMUL m1}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m2", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_policy {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body f32 clamp/select hook}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_f32_clamp_select_reject_authority_attr {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_f32_clamp_select_bad_authority_attr attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %input = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "script-derived-f32-clamp-select", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
