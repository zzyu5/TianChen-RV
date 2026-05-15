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
      selection_kind = "static-variant",
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
      status = "no-active-route"
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
      selection_kind = "static-variant",
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
      selection_kind = "static-variant",
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
      status = "no-active-route"
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
      selection_kind = "static-variant",
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
      status = "no-active-route"
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
      selection_kind = "static-variant",
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
      status = "no-active-route"
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
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock scalar lowering boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "coherence_emission_origin_mismatch",
      status = "no-active-route"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "scalar-fallback-unsupported-emission",
      lowering_pipeline = "scalar-fallback-no-materialized-emitc-route",
      message = "spoofed scalar unsupported route",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      runtime_abi = "scalar-fallback-no-runtime-abi",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      status = "unsupported",
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
