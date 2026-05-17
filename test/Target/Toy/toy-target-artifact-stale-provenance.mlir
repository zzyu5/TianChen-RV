// RUN: not tcrv-translate --tcrv-export-target-header-artifact %s 2>&1 | FileCheck %s --check-prefix=STALE

module {
  tcrv.exec.kernel @toy_stale_header_export {
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
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_stale_header_export", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
    tcrv.exec.diagnostic {
      message = "selected Toy route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "runtime-callable-c-header",
      artifact_metadata = [
        {key = "toy_emitc_lowerable_route", value = "stale-toy-route"},
        {key = "toy_source_op", value = "tcrv_toy.compute_skeleton"},
        {key = "toy_source_role", value = "compute"},
        {key = "toy_source_op_interface", value = "TCRVEmitCLowerableOpInterface"},
        {key = "toy_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "toy_semantic_role_graph", value = "configure->load->compute->store"},
        {key = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"}
      ],
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "tcrv_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "Toy selected compute_skeleton exports a stale header artifact",
      origin = "toy-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@toy_template],
      role = "direct variant",
      runtime_abi = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "toy_value_count", c_type = "size_t", role = "runtime-element-count", ownership = "target-export-abi-owned"}
      ],
      runtime_glue_role = "emitc-cpp-toy-template-runtime-glue",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice
    }
  }
}

// STALE: toy_emitc_lowerable_route
// STALE-SAME: toy-template-compute-emitc-route
