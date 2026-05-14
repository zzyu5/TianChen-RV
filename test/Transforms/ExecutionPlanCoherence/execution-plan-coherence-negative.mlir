// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-check-execution-plan-coherence

// expected-error@+1 {{stale lowering-boundary 'tcrv.exec.diagnostic' selected_variant @scalar_fallback_first_slice as direct variant is not selected by the current dispatch or selected diagnostic surface}}
module {
  tcrv.exec.kernel @coherence_stale_selected_variant {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.variant @old_scalar attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "stale selected target",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @old_scalar
    }
    tcrv.exec.diagnostic {
      message = "boundary for previous selected target",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "coherence_stale_selected_variant",
      status = "metadata-only"
    }
  }
}

// -----

// expected-error@+1 {{selected-path diagnostic origin 'rvv-plugin' does not match selected variant @scalar_fallback_first_slice origin 'scalar-plugin'}}
module {
  tcrv.exec.kernel @coherence_selected_origin_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "origin mismatch selected target",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// -----

// expected-error@+1 {{source_kernel 'other_kernel' does not match selected kernel @coherence_boundary_kernel_mismatch}}
module {
  tcrv.exec.kernel @coherence_boundary_kernel_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected scalar",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock boundary with stale source kernel",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "other_kernel",
      status = "metadata-only"
    }
  }
}

// -----

// expected-error@+1 {{stale lowering-boundary 'tcrv.exec.diagnostic' selected_variant @other_scalar as direct variant is not selected by the current dispatch or selected diagnostic surface}}
module {
  tcrv.exec.kernel @coherence_boundary_variant_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.variant @other_scalar attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected scalar",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock stale boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @other_scalar,
      source_kernel = "coherence_boundary_variant_mismatch",
      status = "metadata-only"
    }
  }
}

// -----

// expected-error@+1 {{lowering-boundary 'tcrv.exec.diagnostic' origin 'rvv-plugin' does not match selected variant @scalar_fallback_first_slice origin 'scalar-plugin'}}
module {
  tcrv.exec.kernel @coherence_boundary_origin_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected scalar",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock boundary with origin mismatch",
      origin = "rvv-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "coherence_boundary_origin_mismatch",
      status = "metadata-only"
    }
  }
}

// -----

// expected-error@+1 {{emission-plan origin 'rvv-plugin' does not match selected variant @scalar_fallback_first_slice origin 'scalar-plugin'}}
module {
  tcrv.exec.kernel @coherence_emission_origin_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected scalar",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "coherence_emission_origin_mismatch",
      status = "metadata-only"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "runtime-callable-c-source",
      emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source",
      lowering_boundary = "tcrv_scalar.lowering_boundary",
      lowering_pipeline = "tcrv-export-scalar-microkernel-c",
      message = "spoofed scalar source route",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      runtime_abi = "scalar-i32-vadd-runtime-callable-c-abi.v1",
      runtime_abi_kind = "scalar-runtime-callable-c-abi",
      runtime_abi_name = "scalar-i32-vadd-runtime-callable-c-function.v1",
      runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}, {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}, {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"}, {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}],
      runtime_glue_role = "runtime-callable-i32-vadd-fallback-function",
      status = "supported",
      target = @scalar_fallback_first_slice
    }
  }
}

// -----

// expected-error@+1 {{selected variant @ghost_variant names unregistered origin plugin 'ghost-plugin'}}
module {
  tcrv.exec.kernel @coherence_unregistered_origin {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @ghost_variant attributes {
      origin = "ghost-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected unregistered origin",
      origin = "ghost-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @ghost_variant
    }
  }
}
