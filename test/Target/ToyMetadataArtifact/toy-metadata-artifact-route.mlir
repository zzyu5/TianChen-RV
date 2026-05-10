// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=ARTIFACT

module {
  // IR-LABEL: tcrv.exec.kernel @toy_artifact_route
  // ARTIFACT: tianchenrv.toy_metadata_artifact.version: 1
  // ARTIFACT: artifact_status: "non-executable-metadata-evidence"
  // ARTIFACT: runtime_execution_claim: "none"
  // ARTIFACT: correctness_claim: "none"
  // ARTIFACT: performance_claim: "none"
  // ARTIFACT: kernel: @toy_artifact_route
  tcrv.exec.kernel @toy_artifact_route {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }

    // IR: tcrv.exec.variant @toy_template_first_slice
    // IR-SAME: origin = "toy-plugin"
    // IR-SAME: requires = [@toy_template]
    // IR: tcrv_toy.lowering_boundary
    // IR-SAME: handoff_kind = "toy-lowering-template"
    // IR-SAME: selected_variant = @toy_template_first_slice
    // IR-SAME: status = "metadata-only"
    // IR: tcrv.exec.diagnostic
    // IR-SAME: artifact_kind = "metadata-diagnostic"
    // IR-SAME: emission_kind = "toy-template-metadata-route"
    // IR-SAME: lowering_boundary = "tcrv_toy.lowering_boundary"
    // IR-SAME: lowering_pipeline = "none-executable-toy-template-metadata"
    // IR-SAME: origin = "toy-plugin"
    // IR-SAME: runtime_abi_kind = "toy-template-metadata"
    // IR-SAME: status = "supported"
    // IR-SAME: target = @toy_template_first_slice
    // ARTIFACT: selected_variant: @toy_template_first_slice
    // ARTIFACT: role: "direct variant"
    // ARTIFACT: origin_plugin: "toy-plugin"
    // ARTIFACT: route: "none-executable-toy-template-metadata"
    // ARTIFACT: emission_kind: "toy-template-metadata-route"
    // ARTIFACT: artifact_kind: "metadata-diagnostic"
    // ARTIFACT: runtime_abi: "toy-metadata-boundary.v1"
    // ARTIFACT: runtime_abi_kind: "toy-template-metadata"
    // ARTIFACT: runtime_abi_name: "toy-metadata-boundary.v1"
    // ARTIFACT: runtime_glue_role: "metadata-only-toy-template-boundary"
    // ARTIFACT: lowering_boundary: "tcrv_toy.lowering_boundary"
    // ARTIFACT: template_abi: "toy-metadata-boundary.v1"
    // ARTIFACT: handoff_kind: "toy-lowering-template"
    // ARTIFACT: required_capability: @toy_template
    // ARTIFACT: required_capability_id: "toy.template"
    // ARTIFACT: required_capability_kind: "extension-template"
    // ARTIFACT: selected_plan_metadata[0]:
    // ARTIFACT:   name: "toy_template_capability_id"
    // ARTIFACT:   value: "toy.template"
    // ARTIFACT: selected_plan_metadata[1]:
    // ARTIFACT:   name: "toy_template_abi"
    // ARTIFACT:   value: "toy-metadata-boundary.v1"
    // ARTIFACT: selected_plan_metadata[2]:
    // ARTIFACT:   name: "toy_template_scope"
    // ARTIFACT:   value: "metadata-only"
  }
}
