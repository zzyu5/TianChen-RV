// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @rvv_exec_binding_reject_wrong_exec_op_kind {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [@rvv]} {
      // expected-error@+1 {{exec_binding @abi_runtime_element_count for buffer ABI role 'lhs-input-buffer' must reference a direct same-kernel tcrv.exec.mem_window}}
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_exec_binding_reject_mem_window_abi_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [@rvv]} {
      // expected-error@+1 {{exec_binding @abi_lhs_input_buffer must reference a tcrv.exec.mem_window with attribute 'abi_role' = "lhs-input-buffer"}}
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_exec_binding_reject_runtime_param_c_name {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_value", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [@rvv]} {
      // expected-error@+1 {{exec_binding @abi_rhs_scalar_value must reference a tcrv.exec.runtime_param with attribute 'c_name' = "rhs_scalar"}}
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_exec_binding_reject_runtime_param_ownership {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "caller-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [@rvv]} {
      // expected-error@+1 {{exec_binding @abi_runtime_element_count must reference a tcrv.exec.runtime_param with attribute 'ownership' = "target-export-abi-owned"}}
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_exec_binding_reject_mem_window_c_type {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int64_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = [@rvv]} {
      // expected-error@+1 {{exec_binding @abi_lhs_input_buffer must reference a tcrv.exec.mem_window with attribute 'c_type' = "const int32_t *"}}
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    }
  }
}
