// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-emission-plans

module {
  // expected-error@+1 {{selected dispatch case declares runtime_guard_required=true but does not link runtime_guard to a same-kernel dispatch-availability-guard runtime_param before RVV route construction}}
  tcrv.exec.kernel @rvv_dispatch_case_missing_runtime_guard {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_scalar_broadcast_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_scalar_broadcast_macc, sew = 32 : i64, source_kernel = "rvv_dispatch_case_missing_runtime_guard", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.macc %a, %b, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @scalar_fallback_path attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_scalar_broadcast_macc {origin = "rvv-plugin", policy = "dispatch-envelope-negative-case", runtime_guard_required = true}
      tcrv.exec.fallback @scalar_fallback_path {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  // expected-error@+1 {{selected dispatch fallback target @bad_scalar_fallback must be a fallback-eligible tcrv.exec.variant with fallback_role='conservative'}}
  tcrv.exec.kernel @rvv_dispatch_fallback_target_not_eligible {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_scalar_broadcast_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_scalar_broadcast_macc, sew = 32 : i64, source_kernel = "rvv_dispatch_fallback_target_not_eligible", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.macc %a, %b, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @bad_scalar_fallback attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_scalar_broadcast_macc {origin = "rvv-plugin", policy = "dispatch-envelope-negative-case"}
      tcrv.exec.fallback @bad_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dispatch_case_wrong_runtime_guard_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_scalar_broadcast_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_scalar_broadcast_macc, sew = 32 : i64, source_kernel = "rvv_dispatch_case_wrong_runtime_guard_role", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.macc %a, %b, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @scalar_fallback_path attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      // expected-error@+1 {{runtime_guard @abi_runtime_element_count must reference a tcrv.exec.runtime_param with ABI role 'dispatch-availability-guard'}}
      tcrv.exec.case @rvv_scalar_broadcast_macc {origin = "rvv-plugin", policy = "dispatch-envelope-negative-case", runtime_guard = @abi_runtime_element_count, runtime_guard_required = true}
      tcrv.exec.fallback @scalar_fallback_path {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dispatch_fallback_target_missing {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_scalar_broadcast_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "dispatch-envelope-negative:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_scalar_broadcast_macc, sew = 32 : i64, source_kernel = "rvv_dispatch_fallback_target_missing", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.macc %a, %b, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_scalar_broadcast_macc {origin = "rvv-plugin", policy = "dispatch-envelope-negative-case"}
      // expected-error@+1 {{references unknown fallback variant @missing_scalar_fallback in enclosing tcrv.exec.kernel}}
      tcrv.exec.fallback @missing_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}
