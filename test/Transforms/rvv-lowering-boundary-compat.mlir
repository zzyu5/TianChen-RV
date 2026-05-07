// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s
// RUN: tcrv-opt %s --tcrv-materialize-rvv-lowering-boundary | FileCheck %s

module {
  tcrv.exec.kernel @rvv_compat_wrapper {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    tcrv.exec.diagnostic {
      message = "select RVV first-slice metadata path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
  }
}

// CHECK-LABEL: tcrv.exec.kernel @rvv_compat_wrapper
// CHECK: tcrv_rvv.lowering_boundary
// CHECK-SAME: role = "direct variant"
// CHECK-SAME: selected_variant = @rvv_first_slice
// CHECK-SAME: source_kernel = "rvv_compat_wrapper"
// CHECK-SAME: status = "unsupported"
