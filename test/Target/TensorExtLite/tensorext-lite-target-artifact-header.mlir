// RUN: tcrv-translate --tcrv-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="int main"
// RUN: sed '/^    tcrv_tensorext_lite.lowering_boundary/d' %s | not tcrv-translate --tcrv-export-target-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-BOUNDARY --implicit-check-not="tcrv_tensorext_lite_config" --implicit-check-not="tcrv_tensorext_lite_tile_mma"
// RUN: sed '/^    tcrv_tensorext_lite.lowering_boundary/d' %s | not tcrv-translate --tcrv-tensorext-lite-emitc-to-cpp 2>&1 | FileCheck %s --check-prefix=MISSING-BOUNDARY --implicit-check-not="tcrv_tensorext_lite_config" --implicit-check-not="tcrv_tensorext_lite_tile_mma"
// RUN: rm -rf %t.missing-boundary.bundle && mkdir %t.missing-boundary.bundle
// RUN: sed '/^    tcrv_tensorext_lite.lowering_boundary/d' %s | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.missing-boundary.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-BOUNDARY --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete" --implicit-check-not="tcrv_tensorext_lite_config" --implicit-check-not="tcrv_tensorext_lite_tile_mma"
// RUN: not test -e %t.missing-boundary.bundle/tianchenrv-target-artifact-bundle.index

module {
  tcrv.exec.kernel @tensorext_lite_header_export {
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
      tcrv_tensorext_lite.config_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 0 : i64, role_specific_interface = "TCRVConfigOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "configure", status = "role-op-boundary", typed_role = "tel.role.config"}
      tcrv_tensorext_lite.load_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 1 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.load_frag"}
      tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
      tcrv_tensorext_lite.store_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 3 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "store_frag", status = "role-op-boundary", typed_role = "tel.role.store_frag"}
    }
    tcrv_tensorext_lite.lowering_boundary {fragment_abi = "tensorext-lite-fragment-boundary.v1", handoff_kind = "tensorext-lite-fragment-mma-template", origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", status = "no-active-route"}
    tcrv.exec.diagnostic {
      message = "selected TensorExtLite route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @tensorext_lite_tile_mma_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "tensorext_lite_emitc_lowerable_route", value = "tensorext-lite-fragment-mma-emitc-route"},
        {key = "tensorext_lite_role_sequence", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_source_ops", value = "tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton"},
        {key = "tensorext_lite_source_roles", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_source_op_interface", value = "TCRVEmitCLowerableOpInterface"},
        {key = "tensorext_lite_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "tensorext_lite_extension_archetype", value = "fragment-mma-like"},
        {key = "tensorext_lite_semantic_role_graph", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"},
        {key = "tensorext_lite_typed_role_realization", value = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"},
        {key = "tensorext_lite_emitc_route_mapping", value = "tensorext-lite-fragment-mma-emitc-route"},
        {key = "tensorext_lite_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"}
      ],
      emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module",
      lowering_boundary = "tcrv_tensorext_lite.lowering_boundary",
      lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route",
      message = "TensorExtLite selected explicit role sequence materializes an EmitC module through the common TCRVEmitCLowerableRoute materializer and packages the MLIR EmitC C/C++ emitter output as a relocatable object artifact for the first slice",
      origin = "tensorext-lite-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@tensorext_lite_tile_mma],
      role = "direct variant",
      runtime_abi = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_glue_role = "emitc-cpp-tensorext-lite-fragment-runtime-glue",
      severity = "info",
      status = "supported",
      target = @tensorext_lite_tile_mma_first_slice
    }
  }
}

// HEADER: #ifndef TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: tianchenrv.tensorext_lite.origin_plugin: tensorext-lite-plugin
// HEADER: tianchenrv.tensorext_lite.selected_variant: @tensorext_lite_tile_mma_first_slice
// HEADER: tianchenrv.tensorext_lite.selected_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.tensorext_lite.runtime_abi_name: tensorext-lite-fragment-mma-runtime-c-abi.v1
// HEADER: tianchenrv.tensorext_lite.emitc_lowerable_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.role_sequence: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_ops: tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton
// HEADER: tianchenrv.tensorext_lite.source_roles: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.tensorext_lite.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.tensorext_lite.extension_archetype: fragment-mma-like
// HEADER: tianchenrv.tensorext_lite.semantic_role_graph: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.common_interface_realization: configure=TCRVExtensionOpInterface
// HEADER: tianchenrv.tensorext_lite.typed_role_realization: configure:tel.role.config
// HEADER: tianchenrv.tensorext_lite.emitc_route_mapping: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.evidence_profile: parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module
// HEADER: void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);
// HEADER: #endif

// MISSING-BOUNDARY: construction-template artifact adapter failed
// MISSING-BOUNDARY-SAME: requires one selected materialized tcrv_tensorext_lite.lowering_boundary before artifact export
