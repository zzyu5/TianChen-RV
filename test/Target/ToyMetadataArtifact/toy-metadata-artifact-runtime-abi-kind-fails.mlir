// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @toy_bad_runtime_abi_kind {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }

    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      tcrv_toy.handoff_kind = "toy-lowering-template",
      tcrv_toy.template_abi = "toy-metadata-boundary.v1"
    } {
    }

    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static Toy metadata route",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice,
      origin = "toy-plugin",
      selection_kind = "static-variant"
    }

    tcrv_toy.lowering_boundary {
      source_kernel = "toy_bad_runtime_abi_kind",
      selected_variant = @toy_template_first_slice,
      origin = "toy-plugin",
      role = "direct variant",
      status = "metadata-only",
      required_capabilities = [@toy_template],
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template",
      template_reason = "Toy metadata boundary only"
    }

    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "Toy metadata artifact candidate with stale runtime ABI kind",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice,
      origin = "toy-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "toy-template-metadata-route",
      lowering_pipeline = "none-executable-toy-template-metadata",
      lowering_boundary = "tcrv_toy.lowering_boundary",
      runtime_abi = "toy-metadata-boundary.v1",
      runtime_abi_kind = "wrong-toy-template-metadata",
      runtime_abi_name = "toy-metadata-boundary.v1",
      runtime_glue_role = "metadata-only-toy-template-boundary",
      required_capabilities = [@toy_template],
      artifact_kind = "metadata-diagnostic",
      selected_plan_metadata = [
        {name = "toy_template_capability_id", value = "toy.template", role = "capability-requirement", note = "records Toy capability id"},
        {name = "toy_template_abi", value = "toy-metadata-boundary.v1", role = "template-abi", note = "mirrors Toy template ABI"},
        {name = "toy_template_scope", value = "metadata-only", role = "evidence-scope", note = "non-executable Toy route"}
      ]
    }
  }
}

// CHECK: target artifact candidate validation failed
// CHECK-SAME: Toy metadata artifact export failed
// CHECK-SAME: runtime ABI kind 'wrong-toy-template-metadata'
// CHECK-SAME: expected 'toy-template-metadata'
