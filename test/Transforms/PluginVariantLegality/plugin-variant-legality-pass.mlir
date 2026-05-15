// RUN: tcrv-opt %s --split-input-file --tcrv-verify-plugin-variant-legality | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @legality_valid_scalar
  tcrv.exec.kernel @legality_valid_scalar {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: requires = [@scalar_fallback]
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback],
      policy = "portable_scalar_fallback_first_slice"
    } {
    }
  }
}
