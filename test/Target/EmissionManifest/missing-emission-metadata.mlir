// RUN: not weft-translate --weft-export-emission-manifest %s 2>&1 | FileCheck %s --implicit-check-not=weft.emission_manifest.version

module {
  weft.exec.kernel @missing_emission_metadata {
    weft.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    weft.exec.variant @fast attributes {
      origin = "mock-plugin",
      requires = [@base]
    } {
    }
    weft.exec.diagnostic {
      message = "fast selected by generic planner",
      reason = "variant-selected",
      selection_kind = "static-variant",
      target = @fast
    }
    weft.exec.diagnostic {
      message = "mock selected lowering boundary",
      origin = "mock-plugin",
      reason = "mock-lowering-boundary",
      required_capabilities = [@base],
      role = "direct variant",
      selected_variant = @fast,
      source_kernel = "missing_emission_metadata",
      status = "no-active-route"
    }
  }
}

// CHECK: Weft-RV emission manifest export failed for kernel @missing_emission_metadata
// CHECK-SAME: selected path @fast as direct variant requires exactly one runtime ABI emission-plan diagnostic
