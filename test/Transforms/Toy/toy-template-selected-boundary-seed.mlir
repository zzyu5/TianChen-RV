// RUN: tcrv-opt %s --tcrv-toy-materialize-template-selected-boundary-seed | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="func.func"
// RUN: tcrv-opt %s --tcrv-toy-materialize-template-selected-boundary-seed --tcrv-materialize-emission-plans --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC
// RUN: tcrv-opt %s --tcrv-toy-materialize-template-selected-boundary-seed --tcrv-materialize-emission-plans | tcrv-translate --tcrv-toy-template-artifact | FileCheck %s --check-prefix=ARTIFACT
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-toy-materialize-template-selected-boundary-seed 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN

module {
  func.func @toy_seed() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// BOUNDARY-LABEL: tcrv.exec.kernel @toy_seed_kernel
// BOUNDARY: tcrv.exec.capability @toy_template
// BOUNDARY-SAME: handoff_kind = "toy-lowering-template"
// BOUNDARY-SAME: id = "toy.template"
// BOUNDARY-SAME: kind = "extension-template"
// BOUNDARY-SAME: status = "available"
// BOUNDARY-SAME: template_abi = "toy-metadata-boundary.v1"
// BOUNDARY-LABEL: tcrv.exec.variant @toy_seed_toy_template
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: requires = [@toy_template]
// BOUNDARY-SAME: tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route"
// BOUNDARY-SAME: tcrv_toy.template_abi = "toy-metadata-boundary.v1"
// BOUNDARY: tcrv_toy.compute_skeleton {
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: required_capabilities = [@toy_template]
// BOUNDARY-SAME: role = "direct variant"
// BOUNDARY-SAME: role_order = 2 : i64
// BOUNDARY-SAME: role_specific_interface = "TCRVComputeOpInterface"
// BOUNDARY-SAME: selected_variant = @toy_seed_toy_template
// BOUNDARY-SAME: source_kernel = "toy_seed_kernel"
// BOUNDARY-SAME: source_role = "compute"
// BOUNDARY-SAME: status = "role-op-boundary"
// BOUNDARY-SAME: typed_role = "toy.role.compute.compute_skeleton"
// BOUNDARY: tcrv.exec.diagnostic
// BOUNDARY-SAME: origin = "toy-plugin"
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: selection_kind = "static-variant"
// BOUNDARY-SAME: status = "selected"
// BOUNDARY-SAME: target = @toy_seed_toy_template

// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.func @tcrv_emitc_toy_seed_kernel_toy_seed_toy_template
// EMITC-NOT: riscv_vector.h
// EMITC: tcrv_emitc.route_source_op=tcrv_toy.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_toy.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_toy_template_compute
// EMITC: call_opaque "tcrv_toy_template_compute"
// EMITC-NOT: __riscv_

// ARTIFACT: tianchenrv.toy_template_target_artifact: materialized
// ARTIFACT: route: "toy-template-compute-emitc-route"
// ARTIFACT: artifact_kind: "metadata-diagnostic"
// ARTIFACT: origin: "toy-plugin"
// ARTIFACT: selected_variant: @toy_seed_toy_template
// ARTIFACT: lowering_boundary: "tcrv_toy.compute_skeleton"
// ARTIFACT: function: "tcrv_emitc_toy_seed_kernel_toy_seed_toy_template"
// ARTIFACT: --- materialized_emitc_cpp_source ---
// ARTIFACT: #include <stdint.h>
// ARTIFACT-NOT: riscv_vector.h
// ARTIFACT: tcrv_emitc.route_source_op=tcrv_toy.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// ARTIFACT: tcrv_toy_template_compute
// ARTIFACT-NOT: __riscv_
// ARTIFACT: --- end_materialized_emitc_cpp_source ---

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-toy-materialize-template-selected-boundary-seed
