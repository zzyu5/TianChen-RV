// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

// A standalone materialization has one family owner.  The selected Toy route
// below is deliberately contaminated with a Scalar body.  The backend registry
// must reject the mixed module before Toy cleanup can erase the Scalar op and
// incorrectly report a complete conversion.
module {
  weft.exec.kernel @mixed_backend_kernel {
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
    weft_scalar.compute_skeleton {
      source_kernel = "mixed_backend_kernel",
      selected_variant = @toy_template_first_slice,
      scalar_immediate = 7 : i64
    }
    weft.exec.diagnostic {
      message = "selected Toy template route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected variant @toy_template_first_slice body to EmitC
