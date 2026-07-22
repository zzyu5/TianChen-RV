// RUN: not weft-opt %s --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --implicit-check-not="emitc.func"

module {
  weft.exec.kernel @tensorext_lite_missing_body {
    weft.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
    weft.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma],
      weft_tensorext_lite.fragment_abi = "tensorext-lite-fragment-boundary.v1",
      weft_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template",
      weft_tensorext_lite.construction_protocol = "extension-family-construction-protocol.v1",
      weft_tensorext_lite.archetype = "fragment-mma-like",
      weft_tensorext_lite.semantic_role_graph = "configure->load_frag->tile_mma->store_frag",
      weft_tensorext_lite.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load_frag=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;tile_mma=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store_frag=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface",
      weft_tensorext_lite.typed_role_realization = "configure:tel.role.config:weft_tensorext_lite.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load_frag:tel.role.load_frag:weft_tensorext_lite.load_frag_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;tile_mma:tel.role.tile_mma:weft_tensorext_lite.tile_mma_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store_frag:tel.role.store_frag:weft_tensorext_lite.store_frag_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface",
      weft_tensorext_lite.emitc_route_mapping = "tensorext-lite-fragment-mma-emitc-route",
      weft_tensorext_lite.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
    }
  }
}

// CHECK: bound family construction for origin 'tensorext-lite-plugin' produced no family-typed final body in variant @tensorext_lite_tile_mma_first_slice
