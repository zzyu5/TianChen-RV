// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: tcrv.exec.kernel @pipeline_toy_template
  // PIPE: tcrv_toy.compute_skeleton {
  // PIPE-SAME: selected_variant = @toy_template_first_slice
  // PIPE: artifact_kind = "metadata-diagnostic"
  // PIPE-SAME: lowering_pipeline = "toy-template-compute-emitc-route"
  // PIPE-SAME: status = "supported"
  // PIPE-NOT: names unknown target artifact export route id 'toy-template-compute-emitc-route'
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
