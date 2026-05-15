// RUN: not tcrv-opt %s --split-input-file 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @scalar_boundary_deleted {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK: custom op 'tcrv_scalar.lowering_boundary' is unknown
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_deleted",
      status = "no-active-route"
    }
  }
}
