// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=ARTIFACT

module {
  // IR-LABEL: tcrv.exec.kernel @tensorext_lite_artifact_route
  // ARTIFACT: tianchenrv.tensorext_lite_metadata_artifact.version: 1
  // ARTIFACT: artifact_status: "non-executable-metadata-evidence"
  // ARTIFACT: runtime_execution_claim: "none"
  // ARTIFACT: hardware_execution_claim: "none"
  // ARTIFACT: correctness_claim: "none"
  // ARTIFACT: performance_claim: "none"
  // ARTIFACT: kernel: @tensorext_lite_artifact_route
  tcrv.exec.kernel @tensorext_lite_artifact_route {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }

    // IR: tcrv.exec.variant @tensorext_lite_tile_mma_first_slice
    // IR-SAME: origin = "tensorext-lite-plugin"
    // IR-SAME: requires = [@tensorext_lite_tile_mma]
    // IR: tcrv_tensorext_lite.lowering_boundary
    // IR-SAME: handoff_kind = "tensorext-lite-fragment-mma-template"
    // IR-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
    // IR-SAME: status = "metadata-only"
    // IR: tcrv_tensorext_lite.tile_mma_skeleton {emitc_call = "__tcrv_tel_tile_mma"
    // IR-SAME: selected_variant = @tensorext_lite_tile_mma_first_slice
    // IR-SAME: source_role = "tile_mma"
    // IR-SAME: typed_role = "tel.role.tile_mma"
    // IR: tcrv.exec.diagnostic
    // IR-SAME: artifact_kind = "metadata-diagnostic"
    // IR-SAME: emission_kind = "tensorext-lite-fragment-mma-generated-route"
    // IR-SAME: lowering_boundary = "tcrv_tensorext_lite.lowering_boundary"
    // IR-SAME: lowering_pipeline = "none-executable-tensorext-lite-fragment-mma-metadata"
    // IR-SAME: origin = "tensorext-lite-plugin"
    // IR-SAME: runtime_abi_kind = "tensorext-lite-fragment-metadata"
    // IR-SAME: status = "supported"
    // IR-SAME: target = @tensorext_lite_tile_mma_first_slice
    // ARTIFACT: selected_variant: @tensorext_lite_tile_mma_first_slice
    // ARTIFACT: role: "direct variant"
    // ARTIFACT: origin_plugin: "tensorext-lite-plugin"
    // ARTIFACT: route: "none-executable-tensorext-lite-fragment-mma-metadata"
    // ARTIFACT: emission_kind: "tensorext-lite-fragment-mma-generated-route"
    // ARTIFACT: artifact_kind: "metadata-diagnostic"
    // ARTIFACT: runtime_abi: "tensorext-lite-fragment-boundary.v1"
    // ARTIFACT: runtime_abi_kind: "tensorext-lite-fragment-metadata"
    // ARTIFACT: runtime_abi_name: "tensorext-lite-fragment-boundary.v1"
    // ARTIFACT: runtime_glue_role: "metadata-only-tensorext-lite-fragment-mma-boundary"
    // ARTIFACT: lowering_boundary: "tcrv_tensorext_lite.lowering_boundary"
    // ARTIFACT: fragment_abi: "tensorext-lite-fragment-boundary.v1"
    // ARTIFACT: handoff_kind: "tensorext-lite-fragment-mma-template"
    // ARTIFACT: required_capability: @tensorext_lite_tile_mma
    // ARTIFACT: required_capability_id: "tensorext_lite.tile_mma"
    // ARTIFACT: required_capability_kind: "fragment-mma-like"
    // ARTIFACT: construction_protocol: "extension-family-construction-protocol.v1"
    // ARTIFACT: extension_archetype: "fragment-mma-like"
    // ARTIFACT: semantic_role_graph: "configure->load_frag->tile_mma->store_frag"
    // ARTIFACT: family_name: "tensorext_lite"
    // ARTIFACT: family_architectural_namespace: "tcrv.tensorext_lite"
    // ARTIFACT: family_concrete_namespace: "tcrv_tensorext_lite"
    // ARTIFACT: semantic_role[2]:
    // ARTIFACT:   role: "tile_mma"
    // ARTIFACT:   operation: "tcrv_tensorext_lite.tile_mma_skeleton"
    // ARTIFACT: validated_role_op: "tcrv_tensorext_lite.tile_mma_skeleton"
    // ARTIFACT: validated_role_op_interface: "TCRVEmitCLowerableOpInterface"
    // ARTIFACT: validated_role_op_source: "tcrv_tensorext_lite.tile_mma_skeleton"
    // ARTIFACT: validated_role_op_source_role: "tile_mma"
    // ARTIFACT: typed_role[2]:
    // ARTIFACT:   typed_role: "tel.role.tile_mma"
    // ARTIFACT:   emitc_call: "__tcrv_tel_tile_mma"
    // ARTIFACT: emitc_route_id: "none-executable-tensorext-lite-fragment-mma-metadata"
    // ARTIFACT: evidence_profile: "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
    // ARTIFACT: generated_output_kind: "role-graph-emitc-source-skeleton"
    // ARTIFACT: generated_function: "tcrv_tensorext_lite_generated_tensorext_lite_tile_mma_first_slice"
    // ARTIFACT: generated_emitc_step[2]:
    // ARTIFACT:   role: "tile_mma"
    // ARTIFACT:   operation: "tcrv_tensorext_lite.tile_mma_skeleton"
    // ARTIFACT:   emitc_call: "__tcrv_tel_tile_mma"
    // ARTIFACT: generated_source:
    // ARTIFACT:   void tcrv_tensorext_lite_generated_tensorext_lite_tile_mma_first_slice(void) {
    // ARTIFACT:     __tcrv_tel_tile_mma();
    // ARTIFACT: selected_plan_metadata[0]:
    // ARTIFACT:   name: "tensorext_lite_tile_mma_capability_id"
    // ARTIFACT:   value: "tensorext_lite.tile_mma"
    // ARTIFACT: selected_plan_metadata[1]:
    // ARTIFACT:   name: "tensorext_lite_tile_mma_abi"
    // ARTIFACT:   value: "tensorext-lite-fragment-boundary.v1"
    // ARTIFACT: selected_plan_metadata[2]:
    // ARTIFACT:   name: "tensorext_lite_tile_mma_scope"
    // ARTIFACT:   value: "metadata-only"
    // ARTIFACT: selected_plan_metadata[3]:
    // ARTIFACT:   name: "tensorext_lite_construction_protocol"
    // ARTIFACT:   value: "extension-family-construction-protocol.v1"
    // ARTIFACT: selected_plan_metadata[4]:
    // ARTIFACT:   name: "tensorext_lite_extension_archetype"
    // ARTIFACT: selected_plan_metadata[5]:
    // ARTIFACT:   name: "tensorext_lite_semantic_role_graph"
    // ARTIFACT: selected_plan_metadata[6]:
    // ARTIFACT:   name: "tensorext_lite_common_interface_realization"
    // ARTIFACT: selected_plan_metadata[7]:
    // ARTIFACT:   name: "tensorext_lite_typed_role_realization"
    // ARTIFACT: selected_plan_metadata[8]:
    // ARTIFACT:   name: "tensorext_lite_emitc_route_mapping"
    // ARTIFACT: selected_plan_metadata[9]:
    // ARTIFACT:   name: "tensorext_lite_evidence_profile"
  }
}
