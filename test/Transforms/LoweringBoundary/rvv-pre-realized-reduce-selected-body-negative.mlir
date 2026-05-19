// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_missing_operation_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_missing_operation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires attribute 'op_kind'}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_missing_operation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_missing_operation {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_missing_operation_scalar {origin = "scalar-plugin"}
    }
  }
}
// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_operation_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_operation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only op_kind "reduce_add"}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_operation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_operation {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_operation_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_memory_form_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_memory_form attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only memory_form "vector-rhs-load"}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "strided-load-store", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_memory_form_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_memory_form {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_memory_form_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_accumulator_role_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_accumulator_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only accumulator_role "rhs-input-buffer"}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "output-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_accumulator_role_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_accumulator_role {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_accumulator_role_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_accumulator_layout_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_accumulator_layout attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only accumulator_layout "rhs-vector-seed-lane0-per-vl-chunk"}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "output-buffer-vector-accumulator-input", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_accumulator_layout_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_accumulator_layout {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_accumulator_layout_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_result_layout_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_result_layout attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only result_layout "store-reduction-lane0-to-output-chunk-base"}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-full-vector-to-output-buffer", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_result_layout_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_result_layout {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_result_layout_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_out_role_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_out_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires result output operand to bind runtime ABI role 'output-buffer'}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_out_role_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_out_role {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_out_role_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_config_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized reduce config to be SEW32 LMUL m1}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m2", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_config_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_config {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_config_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_policy_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body reduce realization hook}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_policy_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_policy {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_policy_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_reduce_reject_runtime_n_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_runtime_n attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      // expected-error@+1 {{requires runtime n/AVL operand to be defined by tcrv_rvv.runtime_abi_value}}
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_runtime_n_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_runtime_n {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_runtime_n_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  // expected-error@+1 {{pre-realized RVV selected body must not be mixed with an already realized setvl/with_vl body before route construction}}
  tcrv.exec.kernel @pre_realized_reduce_reject_mixed_realized_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_reduce_reject_mixed_realized_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @pre_realized_reduce_reject_mixed_realized_body, sew = 32 : i64, source_kernel = "pre_realized_reduce_reject_mixed_realized_body_kernel", status = "selected-lowering-boundary"} {
      } : !tcrv_rvv.vl
      tcrv_rvv.typed_reduce_pre_realized_body %lhs, %rhs, %out, %n {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", accumulator_role = "rhs-input-buffer", lmul = "m1", memory_form = "vector-rhs-load", op_kind = "reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-reduction-lane0-to-output-chunk-base", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_reduce_reject_mixed_realized_body_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_reduce_reject_mixed_realized_body {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_reduce_reject_mixed_realized_body_scalar {origin = "scalar-plugin"}
    }
  }
}
