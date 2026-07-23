// RUN: weft-opt %s --split-input-file --weft-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE

module {
  // PIPE-LABEL: weft.exec.kernel @pipeline_offload_plus_scalar
  // PIPE: weft_offload.lowering_boundary
  // PIPE-SAME: selected_variant = @offload_runtime_first_slice
  // PIPE-SAME: status = "no-active-route"
  // PIPE: weft.exec.diagnostic
  // PIPE-SAME: message = "the Offload extension currently has no active executable lowering or target artifact route"
  // PIPE-SAME: status = "unsupported"
  weft.exec.kernel @pipeline_offload_plus_scalar attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
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
  // PIPE-LABEL: weft.exec.kernel @pipeline_profile_offload_plus_scalar
  // PIPE: weft_offload.lowering_boundary
  // PIPE-SAME: selected_variant = @offload_runtime_first_slice
  // PIPE-SAME: status = "no-active-route"
  // PIPE: weft.exec.diagnostic
  // PIPE-SAME: message = "the Offload extension currently has no active executable lowering or target artifact route"
  // PIPE-SAME: status = "unsupported"
  weft.exec.target @module_offload_scalar_profile {
    id = "profile.offload.scalar",
    target_kind = "profile",
    relations = #weft.capability_relations<provides = ["offload.runtime", "scalar.fallback"]>,
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }

  weft.exec.kernel @pipeline_profile_offload_plus_scalar attributes {problem = @canonical_problem, target = @module_offload_scalar_profile} {
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
  // PIPE-LABEL: weft.exec.kernel @pipeline_vendor_string_no_offload
  weft.exec.kernel @pipeline_vendor_string_no_offload attributes {
    problem = @canonical_problem,
    vendor_hint = "sophgo"
  } {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @vendor_runtime {
      id = "sophgo.runtime",
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

    // PIPE-NOT: weft.exec.variant @offload_runtime_first_slice
    // PIPE: weft.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-NOT: weft_scalar.lowering_boundary
    // PIPE-NOT: weft_offload.lowering_boundary
    // PIPE: weft.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission"
    // PIPE-SAME: target = @scalar_fallback_first_slice

  }
}

// -----

module {
  // PIPE-LABEL: weft.exec.kernel @pipeline_malformed_offload_declines_to_scalar
  weft.exec.kernel @pipeline_malformed_offload_declines_to_scalar attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
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

    // PIPE-NOT: weft.exec.variant @offload_runtime_first_slice
    // PIPE: weft.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-NOT: weft_scalar.lowering_boundary
    // PIPE-NOT: weft_offload.lowering_boundary
    // PIPE: weft.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission"
    // PIPE-SAME: target = @scalar_fallback_first_slice

  }
}
