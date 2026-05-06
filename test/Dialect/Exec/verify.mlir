// RUN: tcrv-opt %s --verify-diagnostics -split-input-file

tcrv.exec.kernel @ok attributes {} {
  tcrv.exec.target @rvv_main {arch = "riscv64"}
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  tcrv.exec.capability @toolchain {id = "llvm-rvv", kind = "toolchain"}
  tcrv.exec.variant @rvv_variant attributes {
    origin = "rvv-plugin",
    requires = [@rvv, @toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.fallback @rvv_variant
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
    // expected-error @+1 {{references unknown fallback variant @missing_variant in enclosing tcrv.exec.kernel}}
    tcrv.exec.fallback @missing_variant
  }
}
