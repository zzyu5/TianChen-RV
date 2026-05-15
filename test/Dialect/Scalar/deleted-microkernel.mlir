// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'tcrv_scalar.i32_vadd_microkernel' is unknown
module {
  tcrv.exec.kernel @deleted_scalar_microkernel {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "deleted_scalar_microkernel"
    }
  }
}
