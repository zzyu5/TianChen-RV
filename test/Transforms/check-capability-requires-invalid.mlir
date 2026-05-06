// RUN: not tcrv-opt %s --tcrv-check-capability-requires 2>&1 | FileCheck %s

tcrv.exec.kernel @disabled_runtime attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime",
    kind = "runtime-offload",
    status = "disabled"
  }
  // CHECK: error: static variant @portable_path requires unavailable capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "disabled"
  // CHECK-SAME: kernel @disabled_runtime
  // CHECK-SAME: not protected by tcrv.exec.dispatch case
  tcrv.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@runtime_probe]
  } {
  }
}

tcrv.exec.kernel @unguarded_dispatch_case attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  tcrv.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    // CHECK: error: unguarded dispatch case in kernel @unguarded_dispatch_case targets variant @runtime_offload_path with unavailable required capability @runtime_probe
    // CHECK-SAME: id = "portable.runtime.probe"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: condition, guard, or policy
    tcrv.exec.case @runtime_offload_path
    tcrv.exec.fallback @portable_fallback
  }
}

tcrv.exec.kernel @unavailable_fallback attributes {} {
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  tcrv.exec.capability @portable_runtime {
    id = "portable.runtime",
    kind = "runtime-offload",
    availability = "unavailable"
  }
  tcrv.exec.variant @available_case_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "runtime-fallback-plugin",
    requires = [@portable_runtime]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @available_case_path {condition = "generic_toolchain_available"}
    // CHECK: error: dispatch fallback in kernel @unavailable_fallback targets variant @portable_fallback with unavailable required capability @portable_runtime
    // CHECK-SAME: id = "portable.runtime"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "unavailable"
    tcrv.exec.fallback @portable_fallback
  }
}
