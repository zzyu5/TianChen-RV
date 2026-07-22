// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

// The input deliberately omits the typed compute body.  It is reconstructed by
// the Toy family, not synthesized by the artifact backend.

module {
  weft.exec.kernel @toy_missing_compute_boundary {
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
      weft_toy.construction_protocol = "extension-family-construction-protocol.v1",
      weft_toy.archetype = "custom-riscv-extension-minimal",
      weft_toy.semantic_role_graph = "configure->load->compute->store",
      weft_toy.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
      weft_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:weft_toy.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:toy.role.load.load_skeleton:weft_toy.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:weft_toy.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:toy.role.store.store_skeleton:weft_toy.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface",
      weft_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      weft_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
    }
  }
}

// CHECK-LABEL: emitc.func @weft_emitc_toy_missing_compute_boundary_toy_template_first_slice
// CHECK: call_opaque "weft_toy_template_compute"
