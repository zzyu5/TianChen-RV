// RUN: not tcrv-translate --tcrv-export-target-artifact %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @tensorext_lite_target_export_blocked {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma],
      tcrv_tensorext_lite.fragment_abi = "tensorext-lite-fragment-boundary.v1",
      tcrv_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template",
      tcrv_tensorext_lite.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_tensorext_lite.archetype = "fragment-mma-like",
      tcrv_tensorext_lite.semantic_role_graph = "configure->load_frag->tile_mma->store_frag",
      tcrv_tensorext_lite.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.typed_role_realization = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.emitc_route_mapping = "tensorext-lite-fragment-mma-emitc-route",
      tcrv_tensorext_lite.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected TensorExtLite route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @tensorext_lite_tile_mma_first_slice
    }
    tcrv.exec.diagnostic {
      message = "TensorExtLite selected explicit role sequence materializes an EmitC module through the common TCRVEmitCLowerableRoute materializer; target artifact export remains unsupported for this family slice",
      origin = "tensorext-lite-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      role = "direct variant",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "info",
      status = "unsupported",
      target = @tensorext_lite_tile_mma_first_slice
    }
  }
}

// CHECK: requires exactly one supported target artifact emission-plan route; found none
