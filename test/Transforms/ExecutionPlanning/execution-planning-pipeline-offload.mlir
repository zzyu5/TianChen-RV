// RUN: not tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: TianChen-RV emission path check failed for kernel @pipeline_offload_plus_scalar
  // PIPE-SAME: selected lowering-boundary validation failed before plugin emission routing
  // PIPE-SAME: selected path @offload_runtime_first_slice as direct variant requires one materialized plugin lowering boundary before emission planning
  tcrv.exec.kernel @pipeline_offload_plus_scalar {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
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
  // PIPE: TianChen-RV emission path check failed for kernel @pipeline_profile_offload_plus_scalar
  // PIPE-SAME: selected lowering-boundary validation failed before plugin emission routing
  // PIPE-SAME: selected path @offload_runtime_first_slice as direct variant requires one materialized plugin lowering boundary before emission planning
  tcrv.exec.target @module_offload_scalar_profile {
    id = "profile.offload.scalar",
    kind = "profile",
    provides = ["offload.runtime", "scalar.fallback"],
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }

  tcrv.exec.kernel @pipeline_profile_offload_plus_scalar attributes {target = @module_offload_scalar_profile} {
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
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
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_vendor_string_no_offload
  tcrv.exec.kernel @pipeline_vendor_string_no_offload attributes {
    vendor_hint = "sophgo"
  } {
    tcrv.exec.capability @vendor_runtime {
      id = "sophgo.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE-NOT: tcrv.exec.variant @offload_runtime_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-NOT: tcrv_scalar.lowering_boundary
    // PIPE-NOT: tcrv_offload.lowering_boundary
    // PIPE: tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission"
    // PIPE-SAME: target = @scalar_fallback_first_slice

  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_malformed_offload_declines_to_scalar
  tcrv.exec.kernel @pipeline_malformed_offload_declines_to_scalar {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "sophgo-vendor-runtime",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE-NOT: tcrv.exec.variant @offload_runtime_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-NOT: tcrv_scalar.lowering_boundary
    // PIPE-NOT: tcrv_offload.lowering_boundary
    // PIPE: tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission"
    // PIPE-SAME: target = @scalar_fallback_first_slice

  }
}
