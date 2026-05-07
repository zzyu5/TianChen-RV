// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @scalar_boundary_valid
  tcrv.exec.kernel @scalar_boundary_valid {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK: tcrv_scalar.lowering_boundary
    // CHECK-SAME: fallback_reason = "scalar fallback selected boundary is metadata-only"
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "scalar_boundary_valid"
    // CHECK-SAME: status = "metadata-only"
    tcrv_scalar.lowering_boundary {
      fallback_reason = "scalar fallback selected boundary is metadata-only",
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_valid",
      status = "metadata-only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_boundary_missing_selected_variant {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{requires attribute 'selected_variant'}}
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      source_kernel = "scalar_boundary_missing_selected_variant",
      status = "metadata-only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_boundary_malformed_required_capabilities {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{attribute 'required_capabilities' must contain only capability symbol references}}
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = ["scalar.fallback"],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_malformed_required_capabilities",
      status = "metadata-only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_boundary_supported_status {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{status must be 'metadata-only'}}
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_supported_status",
      status = "supported"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_boundary_wrong_required_capabilities {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.capability @portable {id = "portable", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{required_capabilities must match selected variant requires metadata}}
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@portable],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_wrong_required_capabilities",
      status = "metadata-only"
    }
  }
}
