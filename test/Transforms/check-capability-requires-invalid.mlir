// RUN: not tcrv-opt %s --tcrv-check-capability-requires 2>&1 | FileCheck %s

tcrv.exec.kernel @disabled_runtime attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime",
    kind = "runtime-offload",
    status = "disabled"
  }
  // CHECK: error: variant @portable_path requires unavailable capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "disabled"
  tcrv.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@runtime_probe]
  } {
  }
}
