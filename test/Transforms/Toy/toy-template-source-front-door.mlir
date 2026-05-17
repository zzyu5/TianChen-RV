// RUN: tcrv-opt %s --tcrv-toy-materialize-template-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="tcrv_toy.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv"
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="tcrv_toy.source_front_door" --implicit-check-not="source-seed" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv"
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-toy-materialize-template-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-NO-BUILTIN

module attributes {
  tcrv_toy.source_front_door = "template_compute",
  tcrv_toy.source_kernel = "toy_header_export"
} {
}

// BOUNDARY: tcrv.exec.kernel @toy_header_export
// BOUNDARY: tcrv.exec.capability @toy_template
// BOUNDARY-SAME: handoff_kind = "toy-lowering-template"
// BOUNDARY-SAME: id = "toy.template"
// BOUNDARY-SAME: template_abi = "toy-metadata-boundary.v1"
// BOUNDARY: tcrv.exec.variant @toy_template_first_slice
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: requires = [@toy_template]
// BOUNDARY-SAME: tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route"
// BOUNDARY: tcrv_toy.compute_skeleton {
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: role = "direct variant"
// BOUNDARY-SAME: role_order = 2 : i64
// BOUNDARY-SAME: selected_variant = @toy_template_first_slice
// BOUNDARY-SAME: source_kernel = "toy_header_export"
// BOUNDARY-SAME: source_role = "compute"
// BOUNDARY-SAME: status = "role-op-boundary"
// BOUNDARY-SAME: template_reason = "toy-source-front-door-template-compute"
// BOUNDARY: tcrv.exec.diagnostic
// BOUNDARY-SAME: message = "selected Toy source front-door route"
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: status = "selected"
// BOUNDARY-SAME: target = @toy_template_first_slice

// PLAN: tcrv.exec.kernel @toy_header_export
// PLAN: tcrv_toy.compute_skeleton {
// PLAN-SAME: selected_variant = @toy_template_first_slice
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: message = "selected Toy source front-door route"
// PLAN-SAME: reason = "variant-selected"
// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "runtime-callable-c-header"
// PLAN-SAME: {key = "toy_emitc_lowerable_route", value = "toy-template-compute-emitc-route"}
// PLAN-SAME: {key = "toy_source_op", value = "tcrv_toy.compute_skeleton"}
// PLAN-SAME: {key = "toy_source_role", value = "compute"}
// PLAN-SAME: {key = "toy_source_op_interface", value = "TCRVEmitCLowerableOpInterface"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-toy-template-module"
// PLAN-SAME: lowering_boundary = "tcrv_toy.compute_skeleton"
// PLAN-SAME: lowering_pipeline = "toy-template-compute-emitc-route"
// PLAN-SAME: origin = "toy-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "direct variant"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "toy-template-compute-runtime-c-abi.v1"
// PLAN-SAME: target = @toy_template_first_slice

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-toy-materialize-template-source-front-door

// PIPE-NO-BUILTIN: TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel
