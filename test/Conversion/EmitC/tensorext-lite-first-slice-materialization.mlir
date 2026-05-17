// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=COMBINED

module {
  tcrv.exec.kernel @tensorext_lite_emitc_kernel {
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
      tcrv_tensorext_lite.config_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 0 : i64, role_specific_interface = "TCRVConfigOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel", source_role = "configure", status = "role-op-boundary", typed_role = "tel.role.config"}
      tcrv_tensorext_lite.load_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 1 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.load_frag"}
      tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
      tcrv_tensorext_lite.store_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 3 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_emitc_kernel", source_role = "store_frag", status = "role-op-boundary", typed_role = "tel.role.store_frag"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "runtime-callable-c-header"
// PLAN-SAME: artifact_metadata = [{key = "tensorext_lite_emitc_lowerable_route", value = "tensorext-lite-fragment-mma-emitc-route"}
// PLAN-SAME: {key = "tensorext_lite_role_sequence", value = "configure->load_frag->tile_mma->store_frag"}
// PLAN-SAME: {key = "tensorext_lite_source_ops", value = "tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton"}
// PLAN-SAME: {key = "tensorext_lite_source_roles", value = "configure->load_frag->tile_mma->store_frag"}
// PLAN-SAME: {key = "tensorext_lite_source_op_interface", value = "TCRVEmitCLowerableOpInterface"}
// PLAN-SAME: {key = "tensorext_lite_construction_protocol", value = "extension-family-construction-protocol.v1"}
// PLAN-SAME: {key = "tensorext_lite_semantic_role_graph", value = "configure->load_frag->tile_mma->store_frag"}
// PLAN-SAME: {key = "tensorext_lite_typed_role_realization", value = "configure:tel.role.config
// PLAN-SAME: }]
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module"
// PLAN-SAME: lowering_boundary = "tcrv_tensorext_lite.lowering_boundary"
// PLAN-SAME: lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route"
// PLAN-SAME: message = "TensorExtLite selected explicit role sequence materializes an EmitC module through the common TCRVEmitCLowerableRoute materializer and exports a declaration-only header artifact for the first slice"
// PLAN-SAME: origin = "tensorext-lite-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: runtime_abi = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-tensorext-lite-fragment-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @tensorext_lite_tile_mma_first_slice

// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.func @tcrv_emitc_tensorext_lite_emitc_kernel_tensorext_lite_tile_mma_first_slice
// EMITC-NOT: riscv_vector.h
// EMITC: tcrv_emitc.route_source_op=tcrv_tensorext_lite.config_skeleton role=configure op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.route_source_op=tcrv_tensorext_lite.load_frag_skeleton role=load_frag op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.route_source_op=tcrv_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.route_source_op=tcrv_tensorext_lite.store_frag_skeleton role=store_frag op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_tensorext_lite.config_skeleton role=configure op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_config
// EMITC: call_opaque "tcrv_tensorext_lite_config"
// EMITC: tcrv_emitc.source_op=tcrv_tensorext_lite.load_frag_skeleton role=load_frag op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_load_frag
// EMITC: call_opaque "tcrv_tensorext_lite_load_frag"
// EMITC: tcrv_emitc.source_op=tcrv_tensorext_lite.tile_mma_skeleton role=tile_mma op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_tile_mma
// EMITC: call_opaque "tcrv_tensorext_lite_tile_mma"
// EMITC: tcrv_emitc.source_op=tcrv_tensorext_lite.store_frag_skeleton role=store_frag op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_tensorext_lite_store_frag
// EMITC: call_opaque "tcrv_tensorext_lite_store_frag"
// EMITC-NOT: __riscv_

// COMBINED: emitc.include <"stdint.h">
// COMBINED: emitc.func @tcrv_emitc_tensorext_lite_emitc_kernel_tensorext_lite_tile_mma_first_slice
// COMBINED: call_opaque "tcrv_tensorext_lite_tile_mma"
