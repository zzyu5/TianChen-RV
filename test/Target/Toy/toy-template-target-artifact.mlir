// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-toy-template-artifact | FileCheck %s --check-prefix=ARTIFACT
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=GENERIC
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s --check-prefix=MANIFEST

module {
  tcrv.exec.kernel @toy_template_target_artifact_kernel {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      tcrv_toy.template_abi = "toy-metadata-boundary.v1",
      tcrv_toy.handoff_kind = "toy-lowering-template",
      tcrv_toy.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_toy.archetype = "custom-riscv-extension-minimal",
      tcrv_toy.semantic_role_graph = "configure->load->compute->store",
      tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected Toy template route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
  }
}

// HELP: --tcrv-toy-template-artifact
// HELP-SAME: selected Toy template path

// ARTIFACT: tianchenrv.toy_template_target_artifact: materialized
// ARTIFACT: route: "toy-template-compute-emitc-route"
// ARTIFACT: artifact_kind: "metadata-diagnostic"
// ARTIFACT: origin: "toy-plugin"
// ARTIFACT: emission_kind: "materialized-emitc-cpp-toy-template-module"
// ARTIFACT: selected_variant: @toy_template_first_slice
// ARTIFACT: lowering_boundary: "tcrv_toy.compute_skeleton"
// ARTIFACT: handoff_kind: "materialized-emitc-cpp-toy-template-source-artifact"
// ARTIFACT: function: "tcrv_emitc_toy_template_target_artifact_kernel_toy_template_first_slice"
// ARTIFACT: --- materialized_emitc_cpp_source ---
// ARTIFACT: #include <stdint.h>
// ARTIFACT-NOT: riscv_vector.h
// ARTIFACT: tcrv_emitc.route_source_op=tcrv_toy.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// ARTIFACT: tcrv_emitc.source_op=tcrv_toy.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_toy_template_compute
// ARTIFACT: tcrv_toy_template_compute
// ARTIFACT-NOT: __riscv_
// ARTIFACT: --- end_materialized_emitc_cpp_source ---

// GENERIC: tianchenrv.toy_template_target_artifact: materialized
// GENERIC: route: "toy-template-compute-emitc-route"
// GENERIC: selected_variant: @toy_template_first_slice
// GENERIC: tcrv_toy_template_compute

// MANIFEST: target_artifacts:
// MANIFEST: route: "toy-template-compute-emitc-route"
// MANIFEST: owner: "toy-plugin"
// MANIFEST: generic_front_door_selectable: true
// MANIFEST: selectable_via: "tcrv-export-target-artifact"
// MANIFEST: evidence_role: "compiler-artifact"
