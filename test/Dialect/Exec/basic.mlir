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
  // CHECK-SAME: abi_role = "lhs-input-buffer"
  // CHECK-SAME: access = "read"
  // CHECK-SAME: binding = "args"
  // CHECK-SAME: c_type = "const int32_t *"
  // CHECK-SAME: memory_space = "host"
  // CHECK-SAME: ownership = "target-export-abi-owned"
  // CHECK-SAME: purpose = "variant-dispatch-guard"
  tcrv.exec.mem_window @input_window {abi_role = "lhs-input-buffer", access = "read", binding = "args", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "variant-dispatch-guard"}

  // CHECK: tcrv.exec.runtime_param @runtime_n
  // CHECK-SAME: abi_role = "runtime-element-count"
  // CHECK-SAME: c_name = "n"
  // CHECK-SAME: c_type = "size_t"
  // CHECK-SAME: ownership = "target-export-abi-owned"
  // CHECK-SAME: purpose = "runtime-abi-scalar"
  tcrv.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}

  // CHECK: tcrv.exec.runtime_param @runtime_guard
  // CHECK-SAME: abi_role = "dispatch-availability-guard"
  // CHECK-SAME: c_name = "rvv_available"
  // CHECK-SAME: c_type = "int"
  // CHECK-SAME: ownership = "target-export-abi-owned"
  // CHECK-SAME: purpose = "runtime-abi-scalar"
  tcrv.exec.runtime_param @runtime_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}

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
    // CHECK-SAME: runtime_guard = @runtime_guard
    // CHECK-SAME: runtime_guard_required = true
    tcrv.exec.case @rvv_variant {condition = "preferred_capability_available", guard = "shape_guard_passed", policy = "prefer_accelerated", runtime_guard = @runtime_guard, runtime_guard_required = true}
    // CHECK: tcrv.exec.fallback @portable_variant
    tcrv.exec.fallback @portable_variant
  }
}
