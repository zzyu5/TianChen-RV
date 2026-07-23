// RUN: weft-opt %s --weft-toy-materialize-template-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="weft_toy.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv"
// RUN: weft-opt %s --weft-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="weft_toy.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv"
// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-toy-materialize-template-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN
// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-NO-BUILTIN

module attributes {
  weft_toy.source_front_door = "template_compute",
  weft_toy.source_kernel = "toy_header_export"
} {
}

// BOUNDARY: weft.exec.target @toy_header_export_target_profile
// BOUNDARY: weft.exec.capability @toy_template
// BOUNDARY-SAME: handoff_kind = "toy-lowering-template"
// BOUNDARY-SAME: id = "toy.template"
// BOUNDARY-SAME: template_abi = "toy-metadata-boundary.v1"
// BOUNDARY: weft.exec.kernel @toy_header_export
// BOUNDARY: weft.exec.variant @toy_template_first_slice
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: requires = [@toy_template]
// BOUNDARY: weft_toy.compute_skeleton {
// BOUNDARY-SAME: selected_variant = @toy_template_first_slice
// BOUNDARY-SAME: source_kernel = "toy_header_export"
// BOUNDARY-SAME: template_reason = "toy-source-front-door-template-compute"
// BOUNDARY: weft.exec.diagnostic
// BOUNDARY-SAME: message = "selected Toy source front-door route"
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: status = "selected"
// BOUNDARY-SAME: target = @toy_template_first_slice

// PLAN: weft.exec.kernel @toy_header_export
// PLAN: weft_toy.compute_skeleton {
// PLAN-SAME: selected_variant = @toy_template_first_slice
// PLAN: weft.exec.diagnostic
// PLAN-SAME: message = "selected Toy source front-door route"
// PLAN-SAME: reason = "variant-selected"
// PLAN: weft.exec.diagnostic {{.*}}artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-toy-template-module"
// PLAN-SAME: lowering_boundary = "weft_toy.compute_skeleton"
// PLAN-SAME: lowering_pipeline = "toy-template-compute-emitc-route"
// PLAN-SAME: origin = "toy-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "direct variant"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "toy-template-compute-runtime-c-abi.v1"
// PLAN-SAME: target = @toy_template_first_slice

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: weft-toy-materialize-template-source-front-door

// PIPE-NO-BUILTIN: Weft-RV execution plan coherence check failed for kernel <missing>: requires at least one weft.exec.kernel
