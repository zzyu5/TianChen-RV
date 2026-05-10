// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | tcrv-opt --split-input-file | FileCheck %s --check-prefix=ROUNDTRIP

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_toy_template
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_toy_template
  tcrv.exec.kernel @pipeline_toy_template {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }

    // PIPE: tcrv.exec.variant @toy_template_first_slice
    // PIPE-SAME: condition = "toy_template_capability_available"
    // PIPE-SAME: guard = "plugin_local_toy_template_metadata"
    // PIPE-SAME: origin = "toy-plugin"
    // PIPE-SAME: policy = "metadata_only_toy_template_first_slice"
    // PIPE-SAME: requires = [@toy_template]
    // PIPE-SAME: tcrv_toy.handoff_kind = "toy-lowering-template"
    // PIPE-SAME: tcrv_toy.template_abi = "toy-metadata-boundary.v1"
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: preference_available = true
    // PIPE-SAME: preference_rank = 0
    // PIPE-SAME: preference_score
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @toy_template_first_slice
    // PIPE: tcrv_toy.lowering_boundary
    // PIPE-SAME: handoff_kind = "toy-lowering-template"
    // PIPE-SAME: origin = "toy-plugin"
    // PIPE-SAME: required_capabilities = [@toy_template]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: selected_variant = @toy_template_first_slice
    // PIPE-SAME: source_kernel = "pipeline_toy_template"
    // PIPE-SAME: status = "metadata-only"
    // PIPE-SAME: template_abi = "toy-metadata-boundary.v1"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: artifact_kind = "metadata-diagnostic"
    // PIPE-SAME: emission_kind = "toy-template-metadata-route"
    // PIPE-SAME: lowering_boundary = "tcrv_toy.lowering_boundary"
    // PIPE-SAME: lowering_pipeline = "none-executable-toy-template-metadata"
    // PIPE-SAME: origin = "toy-plugin"
    // PIPE-SAME: plan_kind = "plugin-emission-plan"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@toy_template]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: runtime_abi = "toy-metadata-boundary.v1"
    // PIPE-SAME: runtime_abi_kind = "toy-template-metadata"
    // PIPE-SAME: runtime_abi_name = "toy-metadata-boundary.v1"
    // PIPE-SAME: runtime_glue_role = "metadata-only-toy-template-boundary"
    // PIPE-SAME: selected_plan_metadata =
    // PIPE-SAME: name = "toy_template_capability_id"
    // PIPE-SAME: value = "toy.template"
    // PIPE-SAME: name = "toy_template_abi"
    // PIPE-SAME: value = "toy-metadata-boundary.v1"
    // PIPE-SAME: name = "toy_template_scope"
    // PIPE-SAME: value = "metadata-only"
    // PIPE-SAME: status = "supported"
    // PIPE-SAME: target = @toy_template_first_slice
    // ROUNDTRIP: tcrv_toy.lowering_boundary
    // ROUNDTRIP-SAME: selected_variant = @toy_template_first_slice
  }
}
