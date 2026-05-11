// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=ARTIFACT

module {
  // IR-LABEL: tcrv.exec.kernel @template_artifact_route
  // ARTIFACT: tianchenrv.template_metadata_artifact.version: 1
  // ARTIFACT: artifact_status: "compiler-handoff-template-artifact"
  // ARTIFACT: artifact_description: "Template extension compiler handoff manifest only"
  // ARTIFACT: runtime_execution_claim: "none"
  // ARTIFACT: hardware_execution_claim: "none"
  // ARTIFACT: correctness_claim: "none"
  // ARTIFACT: performance_claim: "none"
  // ARTIFACT: kernel: @template_artifact_route
  tcrv.exec.kernel @template_artifact_route {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }

    // IR: tcrv.exec.variant @template_zero_core_first_slice
    // IR-SAME: origin = "template-plugin"
    // IR-SAME: requires = [@template_extension]
    // IR: tcrv_template.lowering_boundary
    // IR-SAME: handoff_kind = "template-extension-lowering-boundary"
    // IR-SAME: selected_variant = @template_zero_core_first_slice
    // IR-SAME: status = "metadata-only"
    // IR: tcrv.exec.diagnostic
    // IR-SAME: artifact_kind = "template-extension-handoff-manifest"
    // IR-SAME: emission_kind = "template-extension-manifest-route"
    // IR-SAME: lowering_boundary = "tcrv_template.lowering_boundary"
    // IR-SAME: lowering_pipeline = "template-extension-zero-core-manifest"
    // IR-SAME: origin = "template-plugin"
    // IR-SAME: runtime_abi_kind = "template-extension-handoff"
    // IR-SAME: status = "supported"
    // IR-SAME: target = @template_zero_core_first_slice
    // ARTIFACT: selected_variant: @template_zero_core_first_slice
    // ARTIFACT: role: "direct variant"
    // ARTIFACT: origin_plugin: "template-plugin"
    // ARTIFACT: route: "template-extension-zero-core-manifest"
    // ARTIFACT: emission_kind: "template-extension-manifest-route"
    // ARTIFACT: artifact_kind: "template-extension-handoff-manifest"
    // ARTIFACT: runtime_abi: "template-zero-core-handoff.v1"
    // ARTIFACT: runtime_abi_kind: "template-extension-handoff"
    // ARTIFACT: runtime_abi_name: "template-zero-core-handoff.v1"
    // ARTIFACT: runtime_glue_role: "metadata-only-template-extension-handoff"
    // ARTIFACT: lowering_boundary: "tcrv_template.lowering_boundary"
    // ARTIFACT: integration_contract: "template-zero-core-handoff.v1"
    // ARTIFACT: handoff_kind: "template-extension-lowering-boundary"
    // ARTIFACT: required_capability: @template_extension
    // ARTIFACT: required_capability_id: "template.extension"
    // ARTIFACT: required_capability_kind: "future-extension-template"
    // ARTIFACT: selected_plan_metadata[0]:
    // ARTIFACT:   name: "template_extension_capability_id"
    // ARTIFACT:   value: "template.extension"
    // ARTIFACT: selected_plan_metadata[1]:
    // ARTIFACT:   name: "template_extension_integration_contract"
    // ARTIFACT:   value: "template-zero-core-handoff.v1"
    // ARTIFACT: selected_plan_metadata[2]:
    // ARTIFACT:   name: "template_extension_scope"
    // ARTIFACT:   value: "zero-core-integration"
  }
}
