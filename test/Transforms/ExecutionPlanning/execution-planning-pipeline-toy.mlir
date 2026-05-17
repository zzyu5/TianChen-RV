// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE

module {
  // PIPE: tcrv.exec.kernel @pipeline_toy_template
  // PIPE: tcrv_toy.compute_skeleton {
  // PIPE-SAME: selected_variant = @toy_template_first_slice
  // PIPE: tcrv.exec.diagnostic
  // PIPE-SAME: artifact_kind = "riscv-elf-relocatable-object"
  // PIPE-SAME: artifact_metadata = [{key = "toy_emitc_lowerable_route", value = "toy-template-compute-emitc-route"}
  // PIPE-SAME: {key = "toy_source_op", value = "tcrv_toy.compute_skeleton"}
  // PIPE-SAME: emission_kind = "materialized-emitc-cpp-toy-template-module"
  // PIPE-SAME: lowering_boundary = "tcrv_toy.compute_skeleton"
  // PIPE-SAME: lowering_pipeline = "toy-template-compute-emitc-route"
  // PIPE-SAME: origin = "toy-plugin"
  // PIPE-SAME: reason = "emission_plan"
  // PIPE-SAME: runtime_abi_name = "toy-template-compute-runtime-c-abi.v1"
  // PIPE-SAME: status = "supported"
  // PIPE-NOT: artifact_kind = "metadata-diagnostic"
  // PIPE-NOT: source-export
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
