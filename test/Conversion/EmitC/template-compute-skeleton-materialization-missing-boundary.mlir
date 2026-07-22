// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

// The input deliberately omits the typed compute body.  Family construction
// must rebuild it before the construction-blind EmitC backend consumes it.

module {
  weft.exec.kernel @template_missing_compute_boundary {
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
      weft_template.construction_protocol = "extension-family-construction-protocol.v1",
      weft_template.archetype = "custom-riscv-extension-minimal",
      weft_template.semantic_role_graph = "configure->load->compute->store",
      weft_template.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
      weft_template.typed_role_realization = "configure:template.role.configure.config_skeleton:weft_template.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:template.role.load.load_skeleton:weft_template.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:weft_template.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:template.role.store.store_skeleton:weft_template.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface",
      weft_template.emitc_route_mapping = "template-extension-compute-skeleton-emitc-route",
      weft_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
    }
  }
}

// CHECK-LABEL: emitc.func @weft_emitc_template_missing_compute_boundary_template_zero_core_first_slice
// CHECK: call_opaque "weft_template_compute_skeleton"
