// RUN: not tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: TianChen-RV emission path check failed for kernel @pipeline_toy_template
  // PIPE-SAME: selected lowering-boundary validation failed before plugin emission routing
  // PIPE-SAME: selected path @toy_template_first_slice as direct variant requires one materialized plugin lowering boundary before emission planning
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
