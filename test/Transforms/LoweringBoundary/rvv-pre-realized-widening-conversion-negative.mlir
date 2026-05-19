// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries

module {
  tcrv.exec.kernel @pre_realized_widen_reject_operation_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_operation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only op_kind "widen_i32_to_i64"}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "extend_i16_to_i32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_operation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_operation {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_operation_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_memory_form_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_memory_form attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only memory_form "unit-stride-conversion"}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "vector-rhs-load", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_memory_form_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_memory_form {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_memory_form_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_source_config_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_source_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires source config to be SEW32 LMUL m1}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m2", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_source_config_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_source_config {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_source_config_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_dest_config_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_dest_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires destination config to be SEW64 LMUL m2}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m1", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_dest_config_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_dest_config {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_dest_config_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_relation_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_relation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only conversion_relation "signed-i32m1-to-i64m2"}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "unsigned-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_relation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_relation {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_relation_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_policy_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_policy_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_policy {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_policy_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_n_role_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_n_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      // expected-error@+1 {{requires runtime n/AVL operand to bind runtime ABI role 'runtime-element-count'}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_n_role_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_n_role {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_n_role_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_widen_reject_authority_metadata_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_widen_reject_authority_metadata attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "rvv-i32m1", source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_widen_reject_authority_metadata_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_widen_reject_authority_metadata {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_widen_reject_authority_metadata_scalar {origin = "scalar-plugin"}
    }
  }
}
