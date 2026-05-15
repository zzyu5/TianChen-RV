// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | tcrv-opt --split-input-file | FileCheck %s --check-prefix=ROUNDTRIP

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_offload_plus_scalar
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_offload_plus_scalar
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

    // PIPE: tcrv.exec.variant @offload_runtime_first_slice
    // PIPE-SAME: condition = "offload_runtime_capability_available"
    // PIPE-SAME: guard = "plugin_local_runtime_offload_handoff_metadata"
    // PIPE-SAME: origin = "offload-plugin"
    // PIPE-SAME: policy = "metadata_only_runtime_offload_first_slice"
    // PIPE-SAME: requires = [@offload_runtime]
    // PIPE-SAME: tcrv_offload.handoff_kind = "runtime-offload"
    // PIPE-SAME: tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1"
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: requires = [@scalar_fallback]
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: preference_available = true
    // PIPE-SAME: preference_rank = 0
    // PIPE-SAME: preference_score
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @offload_runtime_first_slice
    // PIPE: tcrv_offload.lowering_boundary
    // PIPE-SAME: handoff_kind = "runtime-offload"
    // PIPE-SAME: origin = "offload-plugin"
    // PIPE-SAME: required_capabilities = [@offload_runtime]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: runtime_abi = "generic-runtime-offload-c-abi-handoff.v1"
    // PIPE-SAME: selected_variant = @offload_runtime_first_slice
    // PIPE-SAME: source_kernel = "pipeline_offload_plus_scalar"
    // PIPE-SAME: status = "metadata-only"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: lowering_boundary = "tcrv_offload.lowering_boundary"
    // PIPE-SAME: message = "the Offload extension currently has no active executable lowering or target artifact route"
    // PIPE-SAME: origin = "offload-plugin"
    // PIPE-SAME: plan_kind = "plugin-emission-plan"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@offload_runtime]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: runtime_abi_kind = "unsupported-plugin-runtime-abi"
    // PIPE-SAME: runtime_abi_name = "unsupported-emission-runtime-abi"
    // PIPE-SAME: runtime_glue_role = "no-runtime-glue-unsupported"
    // PIPE-SAME: status = "unsupported"
    // PIPE-SAME: target = @offload_runtime_first_slice
    // ROUNDTRIP: tcrv_offload.lowering_boundary
    // ROUNDTRIP-SAME: selected_variant = @offload_runtime_first_slice
  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.target @module_offload_scalar_profile
  // PIPE-SAME: id = "profile.offload.scalar"
  // PIPE-SAME: kind = "profile"
  // PIPE-SAME: provides = ["offload.runtime", "scalar.fallback"]
  tcrv.exec.target @module_offload_scalar_profile {
    id = "profile.offload.scalar",
    kind = "profile",
    provides = ["offload.runtime", "scalar.fallback"],
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }

  // PIPE-LABEL: tcrv.exec.kernel @pipeline_profile_offload_plus_scalar
  // PIPE-SAME: target = @module_offload_scalar_profile
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_profile_offload_plus_scalar
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

    // PIPE: tcrv.exec.variant @offload_runtime_first_slice
    // PIPE-SAME: origin = "offload-plugin"
    // PIPE-SAME: requires = [@module_offload_scalar_profile]
    // PIPE-SAME: tcrv_offload.handoff_kind = "runtime-offload"
    // PIPE-SAME: tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1"
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: requires = [@module_offload_scalar_profile]
    // PIPE-NOT: tcrv_scalar.element_count
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @offload_runtime_first_slice
    // PIPE: tcrv_offload.lowering_boundary
    // PIPE-SAME: required_capabilities = [@module_offload_scalar_profile]
    // PIPE-SAME: selected_variant = @offload_runtime_first_slice
    // PIPE-SAME: source_kernel = "pipeline_profile_offload_plus_scalar"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: message = "the Offload extension currently has no active executable lowering or target artifact route"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@module_offload_scalar_profile]
    // PIPE-SAME: runtime_abi_kind = "unsupported-plugin-runtime-abi"
    // PIPE-SAME: status = "unsupported"
    // PIPE-SAME: target = @offload_runtime_first_slice

    // ROUNDTRIP: tcrv_offload.lowering_boundary
    // ROUNDTRIP-SAME: required_capabilities = [@module_offload_scalar_profile]
  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_vendor_string_no_offload
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_vendor_string_no_offload
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

    // ROUNDTRIP-NOT: tcrv.exec.variant @offload_runtime_first_slice
    // ROUNDTRIP: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_malformed_offload_declines_to_scalar
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_malformed_offload_declines_to_scalar
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

    // ROUNDTRIP-NOT: tcrv.exec.variant @offload_runtime_first_slice
    // ROUNDTRIP: tcrv.exec.variant @scalar_fallback_first_slice
  }
}
