// RUN: weft-opt %s --split-input-file --weft-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE

module {
  weft.exec.capability @toy_template {
    id = "toy.template",
    kind = "extension-template",
    status = "available",
    template_abi = "toy-metadata-boundary.v1",
    handoff_kind = "toy-lowering-template"
  }
  weft.exec.target @toy_profile {id = "toy.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@toy_template]}
  // PIPE: weft.exec.kernel @pipeline_toy_template
  // PIPE: weft_toy.compute_skeleton {
  // PIPE-SAME: selected_variant = @toy_template_first_slice
  // PIPE: weft.exec.diagnostic
  // PIPE-SAME: artifact_kind = "riscv-elf-relocatable-object"
  // PIPE-SAME: emission_kind = "materialized-emitc-cpp-toy-template-module"
  // PIPE-SAME: lowering_boundary = "weft_toy.compute_skeleton"
  // PIPE-SAME: lowering_pipeline = "toy-template-compute-emitc-route"
  // PIPE-SAME: origin = "toy-plugin"
  // PIPE-SAME: reason = "emission_plan"
  // PIPE-SAME: runtime_abi_name = "toy-template-compute-runtime-c-abi.v1"
  // PIPE-SAME: status = "supported"
  // PIPE-NOT: artifact_kind = "metadata-diagnostic"
  // PIPE-NOT: source-export
  weft.exec.kernel @pipeline_toy_template attributes {target = @toy_profile, problem = @canonical_problem} {
    weft.exec.template_compute_problem @canonical_problem {template_kind = "compute-skeleton"}
  }
}
