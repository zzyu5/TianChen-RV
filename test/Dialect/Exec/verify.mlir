// RUN: tcrv-opt %s --verify-diagnostics -split-input-file

tcrv.exec.kernel @ok attributes {} {
  tcrv.exec.target @rvv_main {arch = "riscv64"}
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @toolchain {id = "llvm-rvv", kind = "toolchain"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.mem_window @inputs {purpose = "dispatch-guard", binding = "args"}
  tcrv.exec.variant @rvv_variant attributes {
    origin = "rvv-plugin",
    requires = [@rvv, @toolchain]
  } {
    tcrv.exec.hart_parallel attributes {harts = 64 : i64, policy = "static"} {
      tcrv.exec.region attributes {kind = "extension-resource", purpose = "extension-owned-body"} {
        tcrv.exec.diagnostic {reason = "accepted", message = "variant metadata is well formed", severity = "note", status = "selected"}
      }
    }
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available", guard = "shape_guard_passed", policy = "prefer_accelerated"}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @missing_requires attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires structured array attribute 'requires' containing capability symbol references}}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin"} {
  }
}

// -----

tcrv.exec.kernel @unknown_capability attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires unknown capability @zvfh in enclosing tcrv.exec.kernel}}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv, @zvfh]} {
  }
}

// -----

tcrv.exec.kernel @missing_origin attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'origin'}}
  tcrv.exec.variant @rvv_variant attributes {requires = [@rvv]} {
  }
}

// -----

tcrv.exec.kernel @empty_origin attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'origin'}}
  tcrv.exec.variant @rvv_variant attributes {origin = "", requires = [@rvv]} {
  }
}

// -----

tcrv.exec.kernel @empty_variant_condition attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'condition' when present}}
  tcrv.exec.variant @rvv_variant attributes {
    condition = "",
    origin = "rvv-plugin",
    requires = [@rvv]
  } {
  }
}

// -----

tcrv.exec.kernel @empty_variant_guard attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'guard' when present}}
  tcrv.exec.variant @rvv_variant attributes {
    guard = "",
    origin = "rvv-plugin",
    requires = [@rvv]
  } {
  }
}

// -----

tcrv.exec.kernel @empty_variant_policy attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
  tcrv.exec.variant @rvv_variant attributes {
    origin = "rvv-plugin",
    policy = "",
    requires = [@rvv]
  } {
  }
}

// -----

tcrv.exec.kernel @missing_capability_id attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'id'}}
  tcrv.exec.capability @rvv {kind = "isa-vector"}
}

// -----

tcrv.exec.kernel @empty_capability_kind attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'kind'}}
  tcrv.exec.capability @rvv {id = "rvv", kind = ""}
}

// -----

tcrv.exec.kernel @malformed_requires attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{attribute 'requires' must contain only capability symbol references}}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = ["rvv"]} {
  }
}

// -----

tcrv.exec.kernel @bad_fallback attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available"}
    // expected-error @+1 {{references unknown fallback variant @missing_variant in enclosing tcrv.exec.kernel}}
    tcrv.exec.fallback @missing_variant
  }
}

// -----

tcrv.exec.kernel @missing_mem_window_purpose attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'purpose'}}
  tcrv.exec.mem_window @inputs {binding = "args"}
}

// -----

tcrv.exec.mem_window @top_level_window {purpose = "dispatch-guard"}

// expected-error @-2 {{must be nested in a tcrv.exec.kernel or tcrv.exec.variant}}

// -----

tcrv.exec.kernel @bad_hart_count attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires positive integer attribute 'harts' when present}}
    tcrv.exec.hart_parallel attributes {harts = 0 : i64, policy = "static"} {
    }
  }
}

// -----

tcrv.exec.kernel @empty_hart_policy attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
    tcrv.exec.hart_parallel attributes {harts = 1 : i64, policy = ""} {
    }
  }
}

// -----

tcrv.exec.kernel @missing_region_kind attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires non-empty string attribute 'kind'}}
    tcrv.exec.region attributes {purpose = "extension-owned-body"} {
    }
  }
}

// -----

tcrv.exec.kernel @bad_region_nesting attributes {} {
  // expected-error @+1 {{must be nested in a tcrv.exec.variant}}
  tcrv.exec.region attributes {kind = "extension-resource"} {
  }
}

// -----

tcrv.exec.kernel @missing_diagnostic_message attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'message'}}
  tcrv.exec.diagnostic {reason = "fallback-selected"}
}

// -----

tcrv.exec.kernel @empty_diagnostic_severity attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'severity' when present}}
  tcrv.exec.diagnostic {reason = "fallback-selected", message = "using fallback", severity = ""}
}

// -----

tcrv.exec.kernel @unknown_dispatch_case attributes {} {
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    // expected-error @+1 {{references unknown dispatch case variant @missing_variant in enclosing tcrv.exec.kernel}}
    tcrv.exec.case @missing_variant {condition = "preferred_capability_available"}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @case_outside_dispatch attributes {} {
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{must be nested directly in a tcrv.exec.dispatch}}
  tcrv.exec.case @portable_variant {condition = "preferred_capability_available"}
}

// -----

tcrv.exec.kernel @dispatch_without_fallback attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  // expected-error @+1 {{requires exactly one tcrv.exec.fallback}}
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available"}
  }
}

// -----

tcrv.exec.kernel @dispatch_with_duplicate_fallback attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires exactly one tcrv.exec.fallback}}
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available"}
    tcrv.exec.fallback @portable_variant
    tcrv.exec.fallback @rvv_variant
  }
}

// -----

tcrv.exec.kernel @dispatch_without_case attributes {} {
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires at least one tcrv.exec.case}}
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @empty_dispatch_condition attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'condition' when present}}
    tcrv.exec.case @rvv_variant {condition = ""}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @empty_dispatch_guard attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'guard' when present}}
    tcrv.exec.case @rvv_variant {guard = ""}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @empty_dispatch_policy attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
    tcrv.exec.case @rvv_variant {policy = ""}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @fallback_inside_variant attributes {} {
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
    // expected-error @+1 {{must be nested directly in a tcrv.exec.dispatch}}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @dispatch_inside_variant attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{must be nested directly in a tcrv.exec.kernel}}
    tcrv.exec.dispatch attributes {} {
      tcrv.exec.case @rvv_variant {condition = "preferred_capability_available"}
      tcrv.exec.fallback @portable_variant
    }
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
}

// -----

tcrv.exec.kernel @duplicate_dispatch_case attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "first_generic_guard"}
    // expected-error @+1 {{duplicates dispatch case target @rvv_variant in the same tcrv.exec.dispatch}}
    tcrv.exec.case @rvv_variant {condition = "second_generic_guard"}
    tcrv.exec.fallback @portable_variant
  }
}

// -----

tcrv.exec.kernel @illegal_dispatch_body_op attributes {} {
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available"}
    // expected-error @+1 {{is not allowed in tcrv.exec.dispatch; expected only tcrv.exec.case or tcrv.exec.fallback}}
    tcrv.exec.diagnostic {reason = "dispatch-note", message = "diagnostics stay outside dispatch body"}
    tcrv.exec.fallback @portable_variant
  }
}
