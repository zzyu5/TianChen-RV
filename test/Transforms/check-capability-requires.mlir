// RUN: tcrv-opt %s --tcrv-check-capability-requires -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @all_available
tcrv.exec.kernel @all_available attributes {} {
  // CHECK: tcrv.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  tcrv.exec.capability @generic_toolchain {id = "generic.toolchain", kind = "toolchain"}

  // CHECK: tcrv.exec.capability @portable_runtime
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "available"
  tcrv.exec.capability @portable_runtime {id = "portable.runtime", kind = "runtime-offload", status = "available"}

  // CHECK: tcrv.exec.variant @portable_path
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain, @portable_runtime]
  tcrv.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain, @portable_runtime]
  } {
  }
}

// -----

// CHECK-LABEL: tcrv.exec.kernel @guarded_runtime_dispatch
tcrv.exec.kernel @guarded_runtime_dispatch attributes {} {
  // CHECK: tcrv.exec.capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime.probe"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "unavailable"
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "unavailable"
  }

  // CHECK: tcrv.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }

  // CHECK: tcrv.exec.variant @runtime_offload_path
  // CHECK-SAME: origin = "runtime-offload-plugin"
  // CHECK-SAME: requires = [@runtime_probe]
  tcrv.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }

  // CHECK: tcrv.exec.variant @portable_fallback
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain]
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }

  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @runtime_offload_path
    // CHECK-SAME: policy = "runtime_probe_selects_path"
    tcrv.exec.case @runtime_offload_path {policy = "runtime_probe_selects_path"}
    // CHECK: tcrv.exec.fallback @portable_fallback
    tcrv.exec.fallback @portable_fallback
  }
}

// -----

// CHECK-LABEL: tcrv.exec.kernel @guarded_by_inherited_case_metadata
tcrv.exec.kernel @guarded_by_inherited_case_metadata attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  // CHECK: tcrv.exec.variant @runtime_offload_path
  // CHECK-SAME: condition = "runtime_probe_available"
  // CHECK-SAME: guard = "runtime_guard_passed"
  // CHECK-SAME: policy = "prefer_runtime_when_guarded"
  tcrv.exec.variant @runtime_offload_path attributes {
    condition = "runtime_probe_available",
    guard = "runtime_guard_passed",
    origin = "runtime-offload-plugin",
    policy = "prefer_runtime_when_guarded",
    requires = [@runtime_probe]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @runtime_offload_path
    // CHECK-SAME: condition = "runtime_probe_available"
    // CHECK-SAME: guard = "runtime_guard_passed"
    // CHECK-SAME: policy = "prefer_runtime_when_guarded"
    tcrv.exec.case @runtime_offload_path {
      condition = "runtime_probe_available",
      guard = "runtime_guard_passed",
      policy = "prefer_runtime_when_guarded"
    }
    tcrv.exec.fallback @portable_fallback
  }
}
