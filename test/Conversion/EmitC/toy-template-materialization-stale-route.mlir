// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s

module {
  weft.exec.kernel @toy_stale_route_mapping attributes {problem = @canonical_problem} {
    weft.exec.template_compute_problem @canonical_problem {template_kind = "compute-skeleton"}
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      weft_toy.template_abi = "toy-metadata-boundary.v1",
      weft_toy.handoff_kind = "toy-lowering-template",
      weft_toy.emitc_route_mapping = "legacy-stale-route"
    } {
    }
    weft.exec.diagnostic {
      message = "selected stale Toy route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
  }
}

// CHECK: emitc.func private @weft_toy_template_compute
// CHECK: emitc.func @weft_emitc_toy_stale_route_mapping_toy_template_first_slice
// CHECK: call_opaque "weft_toy_template_compute"
