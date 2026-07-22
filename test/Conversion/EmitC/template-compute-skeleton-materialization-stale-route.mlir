// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s

module {
  weft.exec.kernel @template_stale_route_mapping {
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
    weft.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension],
      weft_template.integration_contract = "template-zero-core-handoff.v1",
      weft_template.handoff_kind = "template-extension-lowering-boundary",
      weft_template.emitc_route_mapping = "legacy-stale-route"
    } {
    }
    weft.exec.diagnostic {
      message = "selected stale Template route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @template_zero_core_first_slice
    }
  }
}

// CHECK: emitc.func private @weft_template_compute_skeleton
// CHECK: emitc.func @weft_emitc_template_stale_route_mapping_template_zero_core_first_slice
// CHECK: call_opaque "weft_template_compute_skeleton"
