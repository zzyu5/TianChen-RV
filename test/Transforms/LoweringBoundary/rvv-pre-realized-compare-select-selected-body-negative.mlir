// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-selected-lowering-boundaries

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_predicate_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_predicate attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only predicate_kind "eq", "slt", or "sle" for the bounded selected-body compare/select realization hook}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "ne", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_predicate_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_predicate {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_predicate_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_layout_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_layout attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only select_layout "select-lhs-when-mask-else-rhs" for the bounded selected-body compare/select realization hook}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-rhs-when-mask-else-lhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_layout_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_layout {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_layout_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_mask_source_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_mask_source attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only mask_source "compare-produced-mask-same-vl-scope" for the bounded selected-body compare/select realization hook}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "metadata-mask", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_mask_source_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_mask_source {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_mask_source_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_memory_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_memory attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only memory_form "vector-rhs-load" for the bounded selected-body compare/select realization hook}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "rhs-broadcast-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_memory_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_memory {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_memory_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_config_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized compare/select config to be SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m2", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 64 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_config_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_config {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_config_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_policy_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_policy attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires tail agnostic, mask agnostic policy for the bounded selected-body compare/select realization hook}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = undisturbed>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_policy_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_policy {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_policy_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_cmp_select_reject_rhs_role_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_rhs_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires rhs operand to bind runtime ABI role 'rhs-input-buffer'}}
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_rhs_role_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_rhs_role {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_rhs_role_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  // expected-error@+1 {{family construction found both pre-realized and final RVV bodies}}
  weft.exec.kernel @pre_realized_cmp_select_reject_mixed_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_cmp_select_reject_mixed attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @pre_realized_cmp_select_reject_mixed, sew = 32 : i64, source_kernel = "pre_realized_cmp_select_reject_mixed_kernel", status = "selected-lowering-boundary"} {
      } : !weft_rvv.vl
      weft_rvv.typed_compare_select_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "vector-rhs-load", op_kind = "cmp_select", predicate_kind = "eq", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-lhs-when-mask-else-rhs", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_cmp_select_reject_mixed_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_cmp_select_reject_mixed {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_cmp_select_reject_mixed_scalar {origin = "scalar-plugin"}
    }
  }
}
