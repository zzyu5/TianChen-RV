// RUN: tcrv-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @saxpy
tcrv.exec.kernel @saxpy attributes {} {
  // CHECK: tcrv.exec.target @rvv_main
  tcrv.exec.target @rvv_main {arch = "riscv64", cores = 64 : i64, isa = ["rv64", "rvv"], vlen = 256 : i64}

  // CHECK: tcrv.exec.capability @rvv
  // CHECK-SAME: id = "rvv"
  // CHECK-SAME: kind = "isa-vector"
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", version = "1.0"}

  // CHECK: tcrv.exec.capability @portable
  // CHECK-SAME: id = "portable"
  // CHECK-SAME: kind = "toolchain"
  tcrv.exec.capability @portable {id = "portable", kind = "toolchain"}

  // CHECK: tcrv.exec.mem_window @input_window
  // CHECK-SAME: binding = "args"
  // CHECK-SAME: memory_space = "host"
  // CHECK-SAME: purpose = "variant-dispatch-guard"
  tcrv.exec.mem_window @input_window {binding = "args", memory_space = "host", purpose = "variant-dispatch-guard"}

  // CHECK: tcrv.exec.variant @rvv_variant
  // CHECK-SAME: condition = "preferred_capability_available"
  // CHECK-SAME: guard = "shape_guard_passed"
  // CHECK-SAME: origin = "rvv-plugin"
  // CHECK-SAME: policy = "prefer_accelerated"
  // CHECK-SAME: requires = [@rvv]
  tcrv.exec.variant @rvv_variant attributes {
    condition = "preferred_capability_available",
    guard = "shape_guard_passed",
    origin = "rvv-plugin",
    policy = "prefer_accelerated",
    requires = [@rvv]
  } {
    // CHECK: tcrv.exec.hart_parallel
    // CHECK-SAME: harts = 64
    // CHECK-SAME: policy = "static"
    tcrv.exec.hart_parallel attributes {harts = 64 : i64, policy = "static"} {
      // CHECK: tcrv.exec.region
      // CHECK-SAME: kind = "extension-resource"
      // CHECK-SAME: name = "rvv-resource"
      // CHECK-SAME: purpose = "extension-owned-body"
      tcrv.exec.region attributes {kind = "extension-resource", name = "rvv-resource", purpose = "extension-owned-body"} {
        // CHECK: tcrv.exec.diagnostic
        // CHECK-SAME: message = "rvv variant selected by capability guard"
        // CHECK-SAME: reason = "variant-selected"
        // CHECK-SAME: severity = "note"
        // CHECK-SAME: status = "accepted"
        tcrv.exec.diagnostic {message = "rvv variant selected by capability guard", reason = "variant-selected", severity = "note", status = "accepted"}
      }
    }
  }

  // CHECK: tcrv.exec.variant @portable_variant
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@portable]
  tcrv.exec.variant @portable_variant attributes {origin = "portable-plugin", requires = [@portable]} {
  }

  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @rvv_variant
    // CHECK-SAME: condition = "preferred_capability_available"
    // CHECK-SAME: guard = "shape_guard_passed"
    // CHECK-SAME: policy = "prefer_accelerated"
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available", guard = "shape_guard_passed", policy = "prefer_accelerated"}
    // CHECK: tcrv.exec.fallback @portable_variant
    tcrv.exec.fallback @portable_variant
  }
}
