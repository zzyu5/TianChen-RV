// REQUIRES: tianchenrv-local-native-clangxx
// RUN: rm -f %t.o
// RUN: tcrv-translate --tcrv-export-target-artifact %s > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z67tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice"
// RUN: tcrv-translate --tcrv-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="int main"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-template-extension-compute-skeleton-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// This is lower-level materialized/planned exporter coverage. It intentionally
// starts from selected Template IR and does not exercise a source-level manual
// execution-planning-pipeline pipe into target artifact translation.

module {
  tcrv.exec.kernel @template_emitc_kernel {
    tcrv.exec.capability @template_extension {handoff_kind = "template-extension-lowering-boundary", id = "template.extension", integration_contract = "template-zero-core-handoff.v1", kind = "future-extension-template", status = "available"}
    tcrv.exec.variant @template_zero_core_first_slice attributes {condition = "template_extension_capability_available", guard = "plugin_local_template_extension_handoff_metadata", origin = "template-plugin", policy = "zero_core_template_extension_manifest_first_slice", requires = [@template_extension], tcrv_template.archetype = "custom-riscv-extension-minimal", tcrv_template.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface", tcrv_template.construction_protocol = "extension-family-construction-protocol.v1", tcrv_template.emitc_route_mapping = "template-extension-compute-skeleton-emitc-route", tcrv_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile", tcrv_template.handoff_kind = "template-extension-lowering-boundary", tcrv_template.integration_contract = "template-zero-core-handoff.v1", tcrv_template.semantic_role_graph = "configure->load->compute->store", tcrv_template.typed_role_realization = "configure:template.role.configure.config_skeleton:tcrv_template.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:template.role.load.load_skeleton:tcrv_template.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:tcrv_template.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:template.role.store.store_skeleton:tcrv_template.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"} {
    }
    tcrv.exec.diagnostic {message = "static variant selected by generic cost and capability planning", origin = "template-plugin", preference_available = true, preference_explanation = "Template extension construction-template first slice; route materializes an EmitC module from a selected compute_skeleton role boundary without runtime execution, correctness, or performance claim", preference_policy = "prefer Template only when explicit template.extension capability metadata is available", preference_rank = 0 : i64, preference_score = 5.000000e+01 : f64, preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @template_zero_core_first_slice}
    tcrv.exec.diagnostic {message = "no plugin-provided conservative fallback candidate is available; tcrv.exec.dispatch fallback is not invented", origin = "template-plugin", preference_available = true, preference_explanation = "Template extension construction-template first slice; route materializes an EmitC module from a selected compute_skeleton role boundary without runtime execution, correctness, or performance claim", preference_policy = "prefer Template only when explicit template.extension capability metadata is available", preference_rank = 0 : i64, preference_score = 5.000000e+01 : f64, preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name", reason = "fallback-coverage-missing", selection_kind = "missing-conservative-fallback", severity = "warning", status = "missing", target = @template_zero_core_first_slice}
    tcrv_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @template_zero_core_first_slice, source_kernel = "template_emitc_kernel", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
    tcrv.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object", artifact_metadata = [{key = "template_emitc_route_mapping", value = "template-extension-compute-skeleton-emitc-route"}, {key = "template_source_op", value = "tcrv_template.compute_skeleton"}, {key = "template_source_role", value = "compute"}, {key = "template_source_op_interface", value = "TCRVEmitCLowerableOpInterface"}, {key = "template_construction_protocol", value = "extension-family-construction-protocol.v1"}, {key = "template_semantic_role_graph", value = "configure->load->compute->store"}, {key = "template_typed_role_realization", value = "configure:template.role.configure.config_skeleton:tcrv_template.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:template.role.load.load_skeleton:tcrv_template.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:tcrv_template.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:template.role.store.store_skeleton:tcrv_template.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"}], emission_kind = "materialized-emitc-cpp-template-compute-skeleton-module", lowering_boundary = "tcrv_template.compute_skeleton", lowering_pipeline = "template-extension-compute-skeleton-emitc-route", message = "Template selected compute_skeleton route materializes a verified EmitC module through the common TCRVEmitCLowerableRoute materializer and exports generated C++ through the MLIR EmitC C/C++ emitter", origin = "template-plugin", plan_kind = "plugin-emission-plan", reason = "emission_plan", required_capabilities = [@template_extension], role = "direct variant", runtime_abi = "template-extension-compute-skeleton-runtime-c-abi.v1", runtime_abi_kind = "plugin-owned-runtime-abi", runtime_abi_name = "template-extension-compute-skeleton-runtime-c-abi.v1", runtime_glue_role = "emitc-cpp-template-compute-skeleton-runtime-glue", severity = "info", status = "supported", target = @template_zero_core_first_slice}
  }
}

// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// HEADER: #ifndef TIANCHENRV_TEMPLATE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.template.origin_plugin: template-plugin
// HEADER: tianchenrv.template.selected_variant: @template_zero_core_first_slice
// HEADER: tianchenrv.template.selected_route: template-extension-compute-skeleton-emitc-route
// HEADER: tianchenrv.template.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.template.runtime_abi_name: template-extension-compute-skeleton-runtime-c-abi.v1
// HEADER: tianchenrv.template.emitc_lowerable_route: template-extension-compute-skeleton-emitc-route
// HEADER: tianchenrv.template.source_op: tcrv_template.compute_skeleton
// HEADER: tianchenrv.template.source_role: compute
// HEADER: tianchenrv.template.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.template.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.template.semantic_role_graph: configure->load->compute->store
// HEADER: tianchenrv.template.typed_role_realization: configure:template.role.configure.config_skeleton
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice(void);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o"
// BUNDLE-INDEX: component_group: "template-compute-skeleton-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @template_zero_core_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: component[0]:
// BUNDLE-INDEX: selected_variant: @template_zero_core_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: route: "template-extension-compute-skeleton-emitc-route"
// BUNDLE-INDEX: owner: "template-plugin"
// BUNDLE-INDEX: runtime_abi: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 0
// BUNDLE-INDEX: artifact_metadata[0]:
// BUNDLE-INDEX: key: "template_emitc_route_mapping"
// BUNDLE-INDEX: value: "template-extension-compute-skeleton-emitc-route"
// BUNDLE-INDEX: key: "template_source_op"
// BUNDLE-INDEX: value: "tcrv_template.compute_skeleton"
// BUNDLE-INDEX: key: "template_source_role"
// BUNDLE-INDEX: value: "compute"
// BUNDLE-INDEX: key: "template_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: key: "template_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "template_semantic_role_graph"
// BUNDLE-INDEX: value: "configure->load->compute->store"
// BUNDLE-INDEX: key: "template_typed_role_realization"
// BUNDLE-INDEX: value: "configure:template.role.configure.config_skeleton
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-template-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: file_name: "artifact-1-runtime-callable-c-header-template-extension-compute-skeleton-emitc-route.header.h"
// BUNDLE-INDEX: component_group: "template-compute-skeleton-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "header"
// BUNDLE-INDEX: external_abi_name: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @template_zero_core_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "runtime-callable-c-header"
// BUNDLE-INDEX: route: "template-extension-compute-skeleton-emitc-route.header"
// BUNDLE-INDEX: owner: "template-plugin"
// BUNDLE-INDEX: runtime_abi: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "template-extension-compute-skeleton-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 0
// BUNDLE-INDEX: key: "template_emitc_route_mapping"
// BUNDLE-INDEX: value: "template-extension-compute-skeleton-emitc-route"
// BUNDLE-INDEX: key: "template_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-template-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
