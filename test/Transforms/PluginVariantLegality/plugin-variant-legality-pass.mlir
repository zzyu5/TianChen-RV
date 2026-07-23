// RUN: weft-opt %s --split-input-file --weft-verify-plugin-variant-legality | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @legality_valid_scalar
  weft.exec.kernel @legality_valid_scalar {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // CHECK: weft.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: requires = [@scalar_fallback]
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback],
      policy = "portable_scalar_fallback_first_slice"
    } {
      weft_scalar.immediate_call_body {
        source_kernel = "legality_valid_scalar",
        selected_variant = @scalar_fallback_first_slice,
        scalar_immediate = 7 : i64
      }
    }
  }
}
