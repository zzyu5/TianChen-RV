// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_boundary_deleted {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_typed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    // CHECK: custom op 'tcrv_rvv.lowering_boundary' is unknown
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_typed_body,
      source_kernel = "rvv_boundary_deleted",
      status = "unsupported",
      unsupported_reason = "deleted RVV metadata boundary"
    }
  }
}
