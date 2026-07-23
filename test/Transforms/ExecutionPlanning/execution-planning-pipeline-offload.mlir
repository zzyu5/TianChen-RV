// RUN: not weft-opt %s --split-input-file --weft-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not='status = "supported"'

// FAIL-DAG: Weft-RV selected lowering-boundary materialization failed for kernel @pipeline_offload_plus_scalar: selected owner did not construct an executable final body before boundary exposure: offload delegation plan has no executable implementation
// FAIL-DAG: Weft-RV selected lowering-boundary materialization failed for kernel @pipeline_profile_offload_plus_scalar: selected owner did not construct an executable final body before boundary exposure: offload delegation plan has no executable implementation
// FAIL-DAG: Weft-RV plugin variant materialization for kernel @pipeline_vendor_string_no_offload collected no viable plugin proposals
// FAIL-DAG: Weft-RV plugin variant materialization for kernel @pipeline_malformed_offload_declines_to_scalar collected no viable plugin proposals

module {
  weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
  }
  weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
  }
  weft.exec.target @offload_scalar_profile {id = "offload.scalar.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@offload_runtime, @scalar_fallback]}
  weft.exec.kernel @pipeline_offload_plus_scalar attributes {target = @offload_scalar_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "n",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }

  }
}

// -----

module {
  weft.exec.target @module_offload_scalar_profile {
    id = "profile.offload.scalar",
    target_kind = "profile",
    construction_domain = "riscv-execution",
    relations = #weft.capability_relations<provides = ["offload.runtime", "scalar.fallback"]>,
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }

  weft.exec.kernel @pipeline_profile_offload_plus_scalar attributes {construction_domain = "riscv-execution", problem = @canonical_problem, target = @module_offload_scalar_profile} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    weft.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "n",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }

  }
}

// -----

module {
  weft.exec.capability @vendor_runtime {
    id = "sophgo.runtime",
    kind = "runtime-offload",
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }
  weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
  weft.exec.target @vendor_only_profile {id = "vendor.only.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@vendor_runtime, @scalar_fallback]}
  weft.exec.kernel @pipeline_vendor_string_no_offload attributes {
    target = @vendor_only_profile,
    problem = @canonical_problem,
    vendor_hint = "sophgo"
  } {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// -----

module {
  weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "sophgo-vendor-runtime",
      handoff_kind = "runtime-offload"
  }
  weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
  }
  weft.exec.target @malformed_offload_profile {id = "malformed.offload.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@offload_runtime, @scalar_fallback]}
  weft.exec.kernel @pipeline_malformed_offload_declines_to_scalar attributes {target = @malformed_offload_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}
