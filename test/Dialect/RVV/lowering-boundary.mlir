// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_boundary_valid
  tcrv.exec.kernel @rvv_boundary_valid {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: capability_summary = "rvv"
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "rvv_boundary_valid"
    // CHECK-SAME: status = "unsupported"
    tcrv_rvv.lowering_boundary {
      capability_summary = "rvv",
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "dispatch case",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_boundary_valid",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_boundary_empty_source {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // expected-error@+1 {{requires non-empty string attribute 'source_kernel'}}
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_boundary_missing_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // expected-error@+1 {{requires attribute 'role'}}
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_boundary_missing_role",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_boundary_supported_status {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // expected-error@+1 {{status must be 'unsupported'}}
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_boundary_supported_status",
      status = "supported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_boundary_missing_status {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // expected-error@+1 {{requires attribute 'status'}}
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_boundary_missing_status",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_boundary_executable_claim {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // expected-error@+1 {{unsupported_reason must not claim executable RVV emission}}
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_boundary_executable_claim",
      status = "unsupported",
      unsupported_reason = "executable emission supported by RVV first slice"
    }
  }
}
