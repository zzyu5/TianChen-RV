// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=ARTIFACT

module {
  // IR-LABEL: tcrv.exec.kernel @template_artifact_route
  // ARTIFACT: tianchenrv.template_metadata_artifact.version: 1
  // ARTIFACT: artifact_status: "compiler-construction-template-artifact"
  // ARTIFACT: artifact_description: "Template extension construction manifest artifact"
  // ARTIFACT: runtime_execution_claim: "none"
  // ARTIFACT: hardware_execution_claim: "none"
  // ARTIFACT: correctness_claim: "none"
  // ARTIFACT: performance_claim: "none"
  // ARTIFACT: kernel: @template_artifact_route
  tcrv.exec.kernel @template_artifact_route {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }

    // IR: tcrv.exec.variant @template_zero_core_first_slice
    // IR-SAME: origin = "template-plugin"
    // IR-SAME: requires = [@template_extension]
    // IR-SAME: tcrv_template.archetype = "custom-riscv-extension-minimal"
    // IR-SAME: tcrv_template.construction_protocol = "extension-family-construction-protocol.v1"
    // IR-SAME: tcrv_template.emitc_route_mapping = "template-extension-zero-core-manifest"
    // IR-SAME: tcrv_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
    // IR-SAME: tcrv_template.semantic_role_graph = "configure->load->compute->store"
    // IR-SAME: tcrv_template.typed_role_realization = "{{.*}}compute:template.role.compute.compute_skeleton{{.*}}"
    // IR: tcrv_template.lowering_boundary
    // IR-SAME: handoff_kind = "template-extension-lowering-boundary"
    // IR-SAME: selected_variant = @template_zero_core_first_slice
    // IR-SAME: status = "metadata-only"
    // IR: tcrv_template.compute_skeleton
    // IR-SAME: emitc_call = "__tcrv_template_compute"
    // IR-SAME: role_order = 2 : i64
    // IR-SAME: role_specific_interface = "TCRVComputeOpInterface"
    // IR-SAME: selected_variant = @template_zero_core_first_slice
    // IR-SAME: source_role = "compute"
    // IR-SAME: status = "role-op-boundary"
    // IR-SAME: typed_role = "template.role.compute.compute_skeleton"
    // IR: tcrv.exec.diagnostic
    // IR-SAME: artifact_kind = "template-extension-handoff-manifest"
    // IR-SAME: emission_kind = "template-extension-manifest-route"
    // IR-SAME: lowering_boundary = "tcrv_template.lowering_boundary"
    // IR-SAME: lowering_pipeline = "template-extension-zero-core-manifest"
    // IR-SAME: origin = "template-plugin"
    // IR-SAME: runtime_abi_kind = "template-extension-handoff"
    // IR-SAME: status = "supported"
    // IR-SAME: target = @template_zero_core_first_slice
    // ARTIFACT: selected_variant: @template_zero_core_first_slice
    // ARTIFACT: role: "direct variant"
    // ARTIFACT: origin_plugin: "template-plugin"
    // ARTIFACT: route: "template-extension-zero-core-manifest"
    // ARTIFACT: emission_kind: "template-extension-manifest-route"
    // ARTIFACT: artifact_kind: "template-extension-handoff-manifest"
    // ARTIFACT: runtime_abi: "template-zero-core-handoff.v1"
    // ARTIFACT: runtime_abi_kind: "template-extension-handoff"
    // ARTIFACT: runtime_abi_name: "template-zero-core-handoff.v1"
    // ARTIFACT: runtime_glue_role: "metadata-only-template-extension-handoff"
    // ARTIFACT: lowering_boundary: "tcrv_template.lowering_boundary"
    // ARTIFACT: integration_contract: "template-zero-core-handoff.v1"
    // ARTIFACT: handoff_kind: "template-extension-lowering-boundary"
    // ARTIFACT: required_capability: @template_extension
    // ARTIFACT: required_capability_id: "template.extension"
    // ARTIFACT: required_capability_kind: "future-extension-template"
    // ARTIFACT: construction_protocol: "extension-family-construction-protocol.v1"
    // ARTIFACT: extension_archetype: "custom-riscv-extension-minimal"
    // ARTIFACT: semantic_role_graph: "configure->load->compute->store"
    // ARTIFACT: family_name: "template"
    // ARTIFACT: family_architectural_namespace: "tcrv.template"
    // ARTIFACT: family_concrete_namespace: "tcrv_template"
    // ARTIFACT: family_plugin: "template-plugin"
    // ARTIFACT: semantic_role[0]:
    // ARTIFACT:   role: "configure"
    // ARTIFACT:   order: 0
    // ARTIFACT:   operation: "tcrv_template.config_skeleton"
    // ARTIFACT:   common_interfaces: "TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface"
    // ARTIFACT: semantic_role[2]:
    // ARTIFACT:   role: "compute"
    // ARTIFACT:   order: 2
    // ARTIFACT:   operation: "tcrv_template.compute_skeleton"
    // ARTIFACT:   common_interfaces: "TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"
    // ARTIFACT: common_interface_realization: "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"
    // ARTIFACT: validated_role_op: "tcrv_template.compute_skeleton"
    // ARTIFACT: validated_role_op_interface: "TCRVEmitCLowerableOpInterface"
    // ARTIFACT: validated_role_op_source: "tcrv_template.compute_skeleton"
    // ARTIFACT: validated_role_op_source_role: "compute"
    // ARTIFACT: typed_role_realization: "{{.*}}compute:template.role.compute.compute_skeleton:tcrv_template.compute_skeleton:TCRVComputeOpInterface:__tcrv_template_compute{{.*}}"
    // ARTIFACT: typed_role[2]:
    // ARTIFACT:   typed_role: "template.role.compute.compute_skeleton"
    // ARTIFACT:   role: "compute"
    // ARTIFACT:   order: 2
    // ARTIFACT:   operation: "tcrv_template.compute_skeleton"
    // ARTIFACT:   common_interfaces: "TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"
    // ARTIFACT:   role_specific_interface: "TCRVComputeOpInterface"
    // ARTIFACT:   emitc_lowerable_interface: "TCRVEmitCLowerableInterface"
    // ARTIFACT:   emitc_call: "__tcrv_template_compute"
    // ARTIFACT: emitc_route_id: "template-extension-zero-core-manifest"
    // ARTIFACT: emitc_emission_kind: "template-extension-manifest-route"
    // ARTIFACT: emitc_artifact_kind: "template-extension-handoff-manifest"
    // ARTIFACT: emitc_required_header: "template_extension_intrinsics.h"
    // ARTIFACT: emitc_role_to_call_map: "configure=__tcrv_template_config;load=__tcrv_template_load;compute=__tcrv_template_compute;store=__tcrv_template_store"
    // ARTIFACT: evidence_profile: "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
    // ARTIFACT: generated_output_kind: "role-graph-emitc-source-skeleton"
    // ARTIFACT: generated_function: "tcrv_template_generated_template_zero_core_first_slice"
    // ARTIFACT: generated_required_header: "template_extension_intrinsics.h"
    // ARTIFACT: generated_emitc_step[0]:
    // ARTIFACT:   role: "configure"
    // ARTIFACT:   typed_role: "template.role.configure.config_skeleton"
    // ARTIFACT:   operation: "tcrv_template.config_skeleton"
    // ARTIFACT:   role_specific_interface: "TCRVConfigOpInterface"
    // ARTIFACT:   emitc_lowerable_interface: "TCRVEmitCLowerableInterface"
    // ARTIFACT:   emitc_call: "__tcrv_template_config"
    // ARTIFACT:   source_line: "__tcrv_template_config();"
    // ARTIFACT: generated_emitc_step[2]:
    // ARTIFACT:   role: "compute"
    // ARTIFACT:   typed_role: "template.role.compute.compute_skeleton"
    // ARTIFACT:   operation: "tcrv_template.compute_skeleton"
    // ARTIFACT:   common_interfaces: "TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"
    // ARTIFACT:   role_specific_interface: "TCRVComputeOpInterface"
    // ARTIFACT:   emitc_lowerable_interface: "TCRVEmitCLowerableInterface"
    // ARTIFACT:   emitc_call: "__tcrv_template_compute"
    // ARTIFACT:   source_line: "__tcrv_template_compute();"
    // ARTIFACT: generated_source:
    // ARTIFACT:   #include "template_extension_intrinsics.h"
    // ARTIFACT:   void tcrv_template_generated_template_zero_core_first_slice(void) {
    // ARTIFACT:     /* role[0] configure via tcrv_template.config_skeleton */
    // ARTIFACT:     __tcrv_template_config();
    // ARTIFACT:     /* role[2] compute via tcrv_template.compute_skeleton */
    // ARTIFACT:     __tcrv_template_compute();
    // ARTIFACT:   }
    // ARTIFACT: selected_plan_metadata[0]:
    // ARTIFACT:   name: "template_extension_capability_id"
    // ARTIFACT:   value: "template.extension"
    // ARTIFACT: selected_plan_metadata[1]:
    // ARTIFACT:   name: "template_extension_integration_contract"
    // ARTIFACT:   value: "template-zero-core-handoff.v1"
    // ARTIFACT: selected_plan_metadata[2]:
    // ARTIFACT:   name: "template_extension_scope"
    // ARTIFACT:   value: "zero-core-integration"
    // ARTIFACT: selected_plan_metadata[3]:
    // ARTIFACT:   name: "template_construction_protocol"
    // ARTIFACT:   value: "extension-family-construction-protocol.v1"
    // ARTIFACT:   role: "construction-protocol"
    // ARTIFACT: selected_plan_metadata[4]:
    // ARTIFACT:   name: "template_extension_archetype"
    // ARTIFACT:   value: "custom-riscv-extension-minimal"
    // ARTIFACT: selected_plan_metadata[5]:
    // ARTIFACT:   name: "template_semantic_role_graph"
    // ARTIFACT:   value: "configure->load->compute->store"
    // ARTIFACT: selected_plan_metadata[6]:
    // ARTIFACT:   name: "template_common_interface_realization"
    // ARTIFACT:   role: "common-interface-realization"
    // ARTIFACT: selected_plan_metadata[7]:
    // ARTIFACT:   name: "template_typed_role_realization"
    // ARTIFACT:   role: "typed-role-interface-realization"
    // ARTIFACT: selected_plan_metadata[8]:
    // ARTIFACT:   name: "template_emitc_route_mapping"
    // ARTIFACT:   value: "template-extension-zero-core-manifest"
    // ARTIFACT: selected_plan_metadata[9]:
    // ARTIFACT:   name: "template_evidence_profile"
    // ARTIFACT:   value: "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|generated_output"
  }
}
