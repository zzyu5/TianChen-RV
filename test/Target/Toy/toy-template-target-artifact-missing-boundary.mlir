// RUN: not tcrv-translate --tcrv-toy-template-artifact %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @toy_missing_materialized_emitc_boundary {
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
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
    }
    tcrv.exec.diagnostic {
      artifact_kind = "metadata-diagnostic",
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "tcrv_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "missing Toy compute_skeleton boundary",
      origin = "toy-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@toy_template],
      role = "direct variant",
      runtime_abi = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "toy-template-compute-runtime-c-abi.v1",
      runtime_glue_role = "emitc-cpp-toy-template-runtime-glue",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice
    }
    tcrv.exec.diagnostic {
      message = "selected Toy route without boundary",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
  }
}

// CHECK: selected Toy EmitC route requires one materialized tcrv_toy.compute_skeleton boundary
