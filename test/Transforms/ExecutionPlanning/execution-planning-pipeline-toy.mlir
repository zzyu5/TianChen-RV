// RUN: not tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: TianChen-RV execution plan coherence check failed for kernel @pipeline_toy_template
  // PIPE-SAME: selected target artifact front door @toy_template_first_slice as direct variant route 'toy-template-compute-emitc-route'
  // PIPE-SAME: names unknown target artifact export route id 'toy-template-compute-emitc-route'
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
