// RUN: not tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --implicit-check-not="emitc.func"

module {
  tcrv.exec.kernel @template_missing_compute_boundary {
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
      tcrv_template.integration_contract = "template-zero-core-handoff.v1",
      tcrv_template.handoff_kind = "template-extension-lowering-boundary",
      tcrv_template.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_template.archetype = "custom-riscv-extension-minimal",
      tcrv_template.semantic_role_graph = "configure->load->compute->store",
      tcrv_template.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_template.typed_role_realization = "configure:template.role.configure.config_skeleton:tcrv_template.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:template.role.load.load_skeleton:tcrv_template.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:tcrv_template.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:template.role.store.store_skeleton:tcrv_template.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_template.emitc_route_mapping = "template-extension-compute-skeleton-emitc-route",
      tcrv_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected variant @template_zero_core_first_slice body to EmitC
