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
