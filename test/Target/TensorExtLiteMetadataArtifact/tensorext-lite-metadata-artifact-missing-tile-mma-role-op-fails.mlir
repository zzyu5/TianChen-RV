// RUN: not tcrv-translate %s --tcrv-export-target-artifact 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @tensorext_lite_missing_compute_role {
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
      tcrv_tensorext_lite.typed_role_realization = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:__tcrv_tel_config;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:__tcrv_tel_load_frag;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:__tcrv_tel_tile_mma;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:__tcrv_tel_store_frag",
      tcrv_tensorext_lite.emitc_route_mapping = "none-executable-tensorext-lite-fragment-mma-metadata",
      tcrv_tensorext_lite.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
    } {
    }

    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static TensorExtLite generated route",
      severity = "note",
      status = "selected",
      target = @tensorext_lite_tile_mma_first_slice,
      origin = "tensorext-lite-plugin",
      selection_kind = "static-variant"
    }

    tcrv_tensorext_lite.lowering_boundary {
      source_kernel = "tensorext_lite_missing_compute_role",
      selected_variant = @tensorext_lite_tile_mma_first_slice,
      origin = "tensorext-lite-plugin",
      role = "direct variant",
      status = "metadata-only",
      required_capabilities = [@tensorext_lite_tile_mma],
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template",
      fragment_reason = "TensorExtLite metadata boundary only"
    }

    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "TensorExtLite metadata artifact candidate missing tile_mma role op",
      severity = "info",
      status = "supported",
      target = @tensorext_lite_tile_mma_first_slice,
      origin = "tensorext-lite-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "tensorext-lite-fragment-mma-generated-route",
      lowering_pipeline = "none-executable-tensorext-lite-fragment-mma-metadata",
      lowering_boundary = "tcrv_tensorext_lite.lowering_boundary",
      runtime_abi = "tensorext-lite-fragment-boundary.v1",
      runtime_abi_kind = "tensorext-lite-fragment-metadata",
      runtime_abi_name = "tensorext-lite-fragment-boundary.v1",
      runtime_glue_role = "metadata-only-tensorext-lite-fragment-mma-boundary",
      required_capabilities = [@tensorext_lite_tile_mma],
      artifact_kind = "metadata-diagnostic",
      selected_plan_metadata = [
        {name = "tensorext_lite_tile_mma_capability_id", value = "tensorext_lite.tile_mma", role = "capability-requirement", note = "records TensorExtLite capability id"},
        {name = "tensorext_lite_tile_mma_abi", value = "tensorext-lite-fragment-boundary.v1", role = "fragment-abi", note = "mirrors TensorExtLite fragment ABI"},
        {name = "tensorext_lite_tile_mma_scope", value = "metadata-only", role = "evidence-scope", note = "non-executable TensorExtLite route"},
        {name = "tensorext_lite_construction_protocol", value = "extension-family-construction-protocol.v1", role = "construction-protocol", note = "records TensorExtLite protocol"},
        {name = "tensorext_lite_extension_archetype", value = "fragment-mma-like", role = "extension-archetype", note = "records TensorExtLite archetype"},
        {name = "tensorext_lite_semantic_role_graph", value = "configure->load_frag->tile_mma->store_frag", role = "semantic-role-graph", note = "records TensorExtLite role graph"},
        {name = "tensorext_lite_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface", role = "common-interface-realization", note = "records TensorExtLite common interfaces"},
        {name = "tensorext_lite_typed_role_realization", value = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:__tcrv_tel_config;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:__tcrv_tel_load_frag;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:__tcrv_tel_tile_mma;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:__tcrv_tel_store_frag", role = "typed-role-interface-realization", note = "records TensorExtLite typed roles"},
        {name = "tensorext_lite_emitc_route_mapping", value = "none-executable-tensorext-lite-fragment-mma-metadata", role = "emitc-route-mapping", note = "records TensorExtLite route"},
        {name = "tensorext_lite_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output", role = "evidence-profile", note = "records TensorExtLite evidence"}
      ]
    }
  }
}

// CHECK: selected TensorExtLite candidate requires one tcrv_tensorext_lite.tile_mma_skeleton
