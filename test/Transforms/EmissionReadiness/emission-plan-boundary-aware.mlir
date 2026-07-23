// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-emission-plans | FileCheck %s

module {
  weft.exec.kernel @fallback_only_scalar_without_boundary {
    // CHECK-LABEL: weft.exec.kernel @fallback_only_scalar_without_boundary
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.diagnostic {
      message = "selected scalar fallback envelope",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    // CHECK: lowering_pipeline = "scalar-no-constructed-body-route"
    // CHECK-SAME: status = "unsupported"
  }
}

// -----

module {
  // expected-error@+1 {{bound family construction for origin 'rvv-plugin' rejected selected variant legality: Weft-RV RVV extension plugin first slice failed: materialized RVV variant requires explicit typed RVV extension-family body}}
  weft.exec.kernel @missing_rvv_boundary {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @rvv_missing_boundary_path attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    weft.exec.diagnostic {
      message = "selected RVV path without materialized boundary",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @rvv_missing_boundary_path
    }
  }
}

// -----

module {
  weft.exec.kernel @boundary_origin_mismatch {
    // CHECK-LABEL: weft.exec.kernel @boundary_origin_mismatch
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.diagnostic {
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    weft.exec.diagnostic {
      message = "mock boundary with wrong origin",
      origin = "other-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "boundary_origin_mismatch",
      status = "no-active-route"
    }
    // A legacy diagnostic-like boundary is not construction authority.
    // CHECK: lowering_pipeline = "scalar-no-constructed-body-route"
    // CHECK-SAME: status = "unsupported"
  }
}

// -----

module {
  weft.exec.kernel @boundary_selected_variant_mismatch {
    // CHECK-LABEL: weft.exec.kernel @boundary_selected_variant_mismatch
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.variant @other_scalar attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.diagnostic {
      message = "selected scalar fallback path changed",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    weft.exec.diagnostic {
      message = "stale boundary for old selected variant",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @other_scalar,
      source_kernel = "boundary_selected_variant_mismatch",
      status = "no-active-route"
    }
    // CHECK: lowering_pipeline = "scalar-no-constructed-body-route"
    // CHECK-SAME: status = "unsupported"
  }
}

// -----

module {
  weft.exec.kernel @duplicate_competing_boundaries {
    // CHECK-LABEL: weft.exec.kernel @duplicate_competing_boundaries
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.diagnostic {
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    weft.exec.diagnostic {
      message = "first mock boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "duplicate_competing_boundaries",
      status = "no-active-route"
    }
    weft.exec.diagnostic {
      message = "second mock boundary",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "duplicate_competing_boundaries",
      status = "no-active-route"
    }
    // CHECK: lowering_pipeline = "scalar-no-constructed-body-route"
    // CHECK-SAME: status = "unsupported"
  }
}

// -----

module {
  weft.exec.kernel @boundary_required_capabilities_mismatch {
    // CHECK-LABEL: weft.exec.kernel @boundary_required_capabilities_mismatch
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.capability @portable {
      id = "portable",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.diagnostic {
      message = "selected scalar fallback path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    weft.exec.diagnostic {
      message = "mock boundary with mismatched requires",
      origin = "scalar-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@portable],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "boundary_required_capabilities_mismatch",
      status = "no-active-route"
    }
    // CHECK: lowering_pipeline = "scalar-no-constructed-body-route"
    // CHECK-SAME: status = "unsupported"
  }
}
