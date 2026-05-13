// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @toy_missing_compute_role {
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
      tcrv_toy.template_abi = "toy-metadata-boundary.v1",
      tcrv_toy.handoff_kind = "toy-lowering-template",
      tcrv_toy.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_toy.archetype = "custom-riscv-extension-minimal",
      tcrv_toy.semantic_role_graph = "configure->load->compute->store",
      tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:__tcrv_toy_config;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:__tcrv_toy_load;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:__tcrv_toy_compute;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:__tcrv_toy_store",
      tcrv_toy.emitc_route_mapping = "none-executable-toy-template-metadata",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
    } {
    }

    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static Toy generated route",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice,
      origin = "toy-plugin",
      selection_kind = "static-variant"
    }

    tcrv_toy.lowering_boundary {
      source_kernel = "toy_missing_compute_role",
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
      message = "Toy metadata artifact candidate missing compute role op",
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
      runtime_abi_kind = "toy-template-metadata",
      runtime_abi_name = "toy-metadata-boundary.v1",
      runtime_glue_role = "metadata-only-toy-template-boundary",
      required_capabilities = [@toy_template],
      artifact_kind = "metadata-diagnostic",
      selected_plan_metadata = [
        {name = "toy_template_capability_id", value = "toy.template", role = "capability-requirement", note = "records Toy capability id"},
        {name = "toy_template_abi", value = "toy-metadata-boundary.v1", role = "template-abi", note = "mirrors Toy template ABI"},
        {name = "toy_template_scope", value = "metadata-only", role = "evidence-scope", note = "non-executable Toy route"},
        {name = "toy_construction_protocol", value = "extension-family-construction-protocol.v1", role = "construction-protocol", note = "records Toy protocol"},
        {name = "toy_extension_archetype", value = "custom-riscv-extension-minimal", role = "extension-archetype", note = "records Toy archetype"},
        {name = "toy_semantic_role_graph", value = "configure->load->compute->store", role = "semantic-role-graph", note = "records Toy role graph"},
        {name = "toy_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface", role = "common-interface-realization", note = "records Toy common interfaces"},
        {name = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:__tcrv_toy_config;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:__tcrv_toy_load;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:__tcrv_toy_compute;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:__tcrv_toy_store", role = "typed-role-interface-realization", note = "records Toy typed roles"},
        {name = "toy_emitc_route_mapping", value = "none-executable-toy-template-metadata", role = "emitc-route-mapping", note = "records Toy route"},
        {name = "toy_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output", role = "evidence-profile", note = "records Toy evidence"}
      ]
    }
  }
}

// CHECK: selected Toy candidate requires one tcrv_toy.compute_skeleton
