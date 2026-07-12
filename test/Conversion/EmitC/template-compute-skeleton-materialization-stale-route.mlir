// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  weft.exec.kernel @template_stale_route_mapping {
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
      weft_template.emitc_route_mapping = "template-extension-no-active-emitc-route",
      weft_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping"
    } {
    }
    weft.exec.diagnostic {
      message = "selected stale Template route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @template_zero_core_first_slice
    }
  }
}

// CHECK: must carry EmitC route mapping metadata 'weft_template.emitc_route_mapping'
