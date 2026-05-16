// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: tcrv.exec.kernel @pipeline_toy_template
  // PIPE: tcrv_toy.compute_skeleton {
  // PIPE-SAME: selected_variant = @toy_template_first_slice
  // PIPE: tcrv.exec.diagnostic
  // PIPE-SAME: message = "Toy template target artifact export route is deleted
  // PIPE-SAME: reason = "emission_plan"
  // PIPE-SAME: status = "unsupported"
  // PIPE-NOT: artifact_kind = "metadata-diagnostic"
  // PIPE-NOT: lowering_pipeline = "toy-template-compute-emitc-route"
  tcrv.exec.kernel @pipeline_toy_template {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }

  }
}
