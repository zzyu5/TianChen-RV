// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @template_missing_compute_role_op {
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
      message = "selected static Template generated route",
      severity = "note",
      status = "selected",
      target = @template_zero_core_first_slice,
      origin = "template-plugin",
      selection_kind = "static-variant"
    }

    tcrv_template.lowering_boundary {
      source_kernel = "template_missing_compute_role_op",
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
      message = "Template generated route candidate without ODS compute role op",
      severity = "info",
      status = "supported",
      target = @template_zero_core_first_slice,
      origin = "template-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "template-extension-manifest-route",
      lowering_pipeline = "template-extension-zero-core-manifest",
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
        {name = "template_extension_scope", value = "zero-core-integration", role = "evidence-scope", note = "non-executable Template route"},
        {name = "template_construction_protocol", value = "extension-family-construction-protocol.v1", role = "construction-protocol", note = "records construction protocol"},
        {name = "template_extension_archetype", value = "custom-riscv-extension-minimal", role = "extension-archetype", note = "records Template archetype"},
        {name = "template_semantic_role_graph", value = "configure->load->compute->store", role = "semantic-role-graph", note = "records Template role graph"},
        {name = "template_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface", role = "common-interface-realization", note = "records Template interfaces"},
        {name = "template_typed_role_realization", value = "configure:template.role.configure.config_skeleton:tcrv_template.config_skeleton:TCRVConfigOpInterface:__tcrv_template_config;load:template.role.load.load_skeleton:tcrv_template.load_skeleton:TCRVMemoryOpInterface:__tcrv_template_load;compute:template.role.compute.compute_skeleton:tcrv_template.compute_skeleton:TCRVComputeOpInterface:__tcrv_template_compute;store:template.role.store.store_skeleton:tcrv_template.store_skeleton:TCRVMemoryOpInterface:__tcrv_template_store", role = "typed-role-interface-realization", note = "records Template typed roles"},
        {name = "template_emitc_route_mapping", value = "template-extension-zero-core-manifest", role = "emitc-route-mapping", note = "records Template EmitC route"},
        {name = "template_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output", role = "evidence-profile", note = "records Template evidence profile"}
      ]
    }
  }
}

// CHECK: selected Template candidate requires one tcrv_template.compute_skeleton
