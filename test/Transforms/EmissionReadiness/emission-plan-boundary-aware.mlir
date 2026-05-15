// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-emission-plans

module {
  tcrv.exec.kernel @fallback_only_scalar_without_boundary {
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
      message = "selected scalar fallback envelope",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// -----

module {
  // expected-error@+1 {{selected path @rvv_deleted_route_path as direct variant requires one materialized plugin lowering boundary before emission planning}}
  tcrv.exec.kernel @missing_rvv_boundary {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_deleted_route_path attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected deleted RVV route path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @rvv_deleted_route_path
    }
  }
}

// -----

module {
  // expected-error@+1 {{origin 'other-plugin' does not match selected variant @scalar_fallback_first_slice origin 'scalar-plugin'}}
  tcrv.exec.kernel @boundary_origin_mismatch {
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
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock boundary with wrong origin",
      origin = "other-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "boundary_origin_mismatch",
      status = "no-active-route"
    }
  }
}

// -----

module {
  // expected-error@+1 {{stale lowering boundary 'tcrv.exec.diagnostic' selected_variant @other_scalar as direct variant is not selected by the current dispatch or selected diagnostic surface}}
  tcrv.exec.kernel @boundary_selected_variant_mismatch {
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
      message = "selected scalar fallback path changed",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "stale boundary for old selected variant",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @other_scalar,
      source_kernel = "boundary_selected_variant_mismatch",
      status = "no-active-route"
    }
  }
}

// -----

module {
  // expected-error@+1 {{duplicate competing lowering boundaries for selected path @scalar_fallback_first_slice as direct variant}}
  tcrv.exec.kernel @duplicate_competing_boundaries {
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
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "first mock boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "duplicate_competing_boundaries",
      status = "no-active-route"
    }
    tcrv.exec.diagnostic {
      message = "second mock boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "duplicate_competing_boundaries",
      status = "no-active-route"
    }
  }
}

// -----

module {
  // expected-error@+1 {{required_capabilities must be a safe subset of selected variant @scalar_fallback_first_slice requires metadata}}
  tcrv.exec.kernel @boundary_required_capabilities_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.capability @portable {
      id = "portable",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      message = "mock boundary with mismatched requires",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@portable],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "boundary_required_capabilities_mismatch",
      status = "no-active-route"
    }
  }
}
