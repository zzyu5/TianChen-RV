// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @template_bad_handoff_kind {
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
      source_kernel = "template_bad_handoff_kind",
      selected_variant = @template_zero_core_first_slice,
      origin = "template-plugin",
      role = "direct variant",
      status = "metadata-only",
      required_capabilities = [@template_extension],
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "stale-template-handoff",
      template_reason = "Template metadata boundary only"
    }
  }
}

// CHECK: handoff_kind must be 'template-extension-lowering-boundary'
