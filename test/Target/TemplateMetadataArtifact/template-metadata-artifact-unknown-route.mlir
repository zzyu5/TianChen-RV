// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @template_unknown_route {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }

    tcrv.exec.variant @template_zero_core_first_slice attributes {
      origin = "template-plugin",
      requires = [@template_extension],
      tcrv_template.handoff_kind = "template-extension-lowering-boundary",
      tcrv_template.integration_contract = "template-zero-core-handoff.v1"
    } {
    }

    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static Template metadata route",
      severity = "note",
      status = "selected",
      target = @template_zero_core_first_slice,
      origin = "template-plugin",
      selection_kind = "static-variant"
    }

    tcrv_template.lowering_boundary {
      source_kernel = "template_unknown_route",
      selected_variant = @template_zero_core_first_slice,
      origin = "template-plugin",
      role = "direct variant",
      status = "metadata-only",
      required_capabilities = [@template_extension],
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary",
      template_reason = "Template metadata boundary only"
    }

    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "Template metadata artifact candidate with unknown route",
      severity = "info",
      status = "supported",
      target = @template_zero_core_first_slice,
      origin = "template-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "template-extension-manifest-route",
      lowering_pipeline = "unknown-template-route",
      lowering_boundary = "tcrv_template.lowering_boundary",
      runtime_abi = "template-zero-core-handoff.v1",
      runtime_abi_kind = "template-extension-handoff",
      runtime_abi_name = "template-zero-core-handoff.v1",
      runtime_glue_role = "metadata-only-template-extension-handoff",
      required_capabilities = [@template_extension],
      artifact_kind = "template-extension-handoff-manifest",
      selected_plan_metadata = [
        {name = "template_extension_capability_id", value = "template.extension", role = "capability-requirement", note = "records Template capability id"},
        {name = "template_extension_integration_contract", value = "template-zero-core-handoff.v1", role = "integration-contract", note = "mirrors Template extension contract"},
        {name = "template_extension_scope", value = "zero-core-integration", role = "evidence-scope", note = "non-executable Template route"}
      ]
    }
  }
}

// CHECK: names unknown target artifact export route id 'unknown-template-route'
