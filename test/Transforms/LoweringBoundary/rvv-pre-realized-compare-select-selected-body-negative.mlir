// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_predicate_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_predicate attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only predicate_kind "eq", "slt", or "sle" for the bounded selected-body compare/select realization hook}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "ne", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_predicate_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_predicate {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_predicate_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_layout_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_layout attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only select_layout "select-lhs-when-mask-else-rhs" for the bounded selected-body compare/select realization hook}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-rhs-when-mask-else-lhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_layout_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_layout {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_layout_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_mask_source_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_mask_source attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only mask_source "compare-produced-mask-same-vl-scope" for the bounded selected-body compare/select realization hook}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "metadata-mask", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_mask_source_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_mask_source {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_mask_source_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_memory_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_memory attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only memory_form "vector-rhs-load" for the bounded selected-body compare/select realization hook}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "rhs-broadcast-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_memory_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_memory {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_memory_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_config_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized compare/select config to be SEW32 LMUL m1}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m2", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_config_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_config {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_config_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_policy_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body compare/select realization hook}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = undisturbed>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_policy_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_policy {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_policy_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_cmp_select_reject_rhs_role_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_rhs_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires rhs operand to bind runtime ABI role 'rhs-input-buffer'}}
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_rhs_role_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_rhs_role {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_rhs_role_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  // expected-error@+1 {{pre-realized RVV selected body must not be mixed with an already realized setvl/with_vl body before route construction}}
  tcrv.exec.kernel @pre_realized_cmp_select_reject_mixed_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_cmp_select_reject_mixed attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @pre_realized_cmp_select_reject_mixed, sew = 32 : i64, source_kernel = "pre_realized_cmp_select_reject_mixed_kernel", status = "selected-lowering-boundary"} {
      } : !tcrv_rvv.vl
      tcrv_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_cmp_select_reject_mixed_scalar attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_cmp_select_reject_mixed {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_cmp_select_reject_mixed_scalar {origin = "scalar-plugin"}
    }
  }
}
