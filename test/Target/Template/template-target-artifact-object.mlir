// REQUIRES: weft-local-native-clangxx
// RUN: rm -f %t.o
// RUN: weft-translate --weft-export-target-artifact %s > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z67weft_emitc_template_emitc_kernel_template_zero_core_first_slice"
// RUN: weft-translate --weft-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="int main"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-translate --weft-export-target-artifact-bundle --weft-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-template-extension-compute-skeleton-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/weft-target-artifact-bundle.index

// This is lower-level materialized/planned exporter coverage. It intentionally
// starts from selected Template IR and does not exercise a source-level manual
// execution-planning-pipeline pipe into target artifact translation.

module {
  weft.exec.kernel @template_emitc_kernel {
    weft.exec.capability @template_extension {handoff_kind = "template-extension-lowering-boundary", id = "template.extension", integration_contract = "template-zero-core-handoff.v1", kind = "future-extension-template", status = "available"}
    weft.exec.variant @template_zero_core_first_slice attributes {condition = "template_extension_capability_available", guard = "plugin_local_template_extension_handoff_metadata", origin = "template-plugin", policy = "zero_core_template_extension_manifest_first_slice", requires = [@template_extension], weft_template.archetype = "custom-riscv-extension-minimal", weft_template.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface", weft_template.construction_protocol = "extension-family-construction-protocol.v1", weft_template.emitc_route_mapping = "template-extension-compute-skeleton-emitc-route", weft_template.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile", weft_template.handoff_kind = "template-extension-lowering-boundary", weft_template.integration_contract = "template-zero-core-handoff.v1", weft_template.semantic_role_graph = "configure->load->compute->store", weft_template.typed_role_realization = "configure:template.role.configure.config_skeleton:weft_template.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:template.role.load.load_skeleton:weft_template.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:weft_template.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:template.role.store.store_skeleton:weft_template.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface"} {
    }
    weft.exec.diagnostic {message = "static variant selected by generic cost and capability planning", origin = "template-plugin", preference_available = true, preference_explanation = "Template extension construction-template first slice; route materializes an EmitC module from a selected compute_skeleton role boundary without runtime execution, correctness, or performance claim", preference_policy = "prefer Template only when explicit template.extension capability metadata is available", preference_rank = 0 : i64, preference_score = 5.000000e+01 : f64, preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @template_zero_core_first_slice}
    weft.exec.diagnostic {message = "no plugin-provided conservative fallback candidate is available; weft.exec.dispatch fallback is not invented", origin = "template-plugin", preference_available = true, preference_explanation = "Template extension construction-template first slice; route materializes an EmitC module from a selected compute_skeleton role boundary without runtime execution, correctness, or performance claim", preference_policy = "prefer Template only when explicit template.extension capability metadata is available", preference_rank = 0 : i64, preference_score = 5.000000e+01 : f64, preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name", reason = "fallback-coverage-missing", selection_kind = "missing-conservative-fallback", severity = "warning", status = "missing", target = @template_zero_core_first_slice}
    weft_template.compute_skeleton {origin = "template-plugin", required_capabilities = [@template_extension], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @template_zero_core_first_slice, source_kernel = "template_emitc_kernel", source_role = "compute", status = "role-op-boundary", typed_role = "template.role.compute.compute_skeleton"}
    weft.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object", artifact_metadata = [{key = "template_emitc_route_mapping", value = "template-extension-compute-skeleton-emitc-route"}, {key = "template_source_op", value = "weft_template.compute_skeleton"}, {key = "template_source_role", value = "compute"}, {key = "template_source_op_interface", value = "WEFTEmitCLowerableOpInterface"}, {key = "template_construction_protocol", value = "extension-family-construction-protocol.v1"}, {key = "template_semantic_role_graph", value = "configure->load->compute->store"}, {key = "template_typed_role_realization", value = "configure:template.role.configure.config_skeleton:weft_template.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:template.role.load.load_skeleton:weft_template.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:template.role.compute.compute_skeleton:weft_template.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:template.role.store.store_skeleton:weft_template.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface"}], emission_kind = "materialized-emitc-cpp-template-compute-skeleton-module", lowering_boundary = "weft_template.compute_skeleton", lowering_pipeline = "template-extension-compute-skeleton-emitc-route", message = "Template selected compute_skeleton route materializes a verified EmitC module through the common WEFTEmitCLowerableRoute materializer and exports generated C++ through the MLIR EmitC C/C++ emitter", origin = "template-plugin", plan_kind = "plugin-emission-plan", reason = "emission_plan", required_capabilities = [@template_extension], role = "direct variant", runtime_abi = "template-extension-compute-skeleton-runtime-c-abi.v1", runtime_abi_kind = "plugin-owned-runtime-abi", runtime_abi_name = "template-extension-compute-skeleton-runtime-c-abi.v1", runtime_glue_role = "emitc-cpp-template-compute-skeleton-runtime-glue", severity = "info", status = "supported", target = @template_zero_core_first_slice}
  }
}

// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_template_emitc_kernel_template_zero_core_first_slice

// BUNDLE-STDOUT: weft.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "weft-target-artifact-bundle.index"

// HEADER: #ifndef WEFT_TEMPLATE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: weft.template.origin_plugin: template-plugin
// HEADER: weft.template.selected_variant: @template_zero_core_first_slice
// HEADER: weft.template.selected_route: template-extension-compute-skeleton-emitc-route
// HEADER: weft.template.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.template.runtime_abi_name: template-extension-compute-skeleton-runtime-c-abi.v1
// HEADER: weft.template.emitc_lowerable_route: template-extension-compute-skeleton-emitc-route
// HEADER: weft.template.source_op: weft_template.compute_skeleton
// HEADER: weft.template.source_role: compute
// HEADER: weft.template.source_op_interface: WEFTEmitCLowerableOpInterface
// HEADER: weft.template.construction_protocol: extension-family-construction-protocol.v1
// HEADER: weft.template.semantic_role_graph: configure->load->compute->store
// HEADER: weft.template.typed_role_realization: configure:template.role.configure.config_skeleton
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void weft_emitc_template_emitc_kernel_template_zero_core_first_slice(void);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: weft.target_artifact_bundle.version: 1
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
// BUNDLE-INDEX: value: "weft_template.compute_skeleton"
// BUNDLE-INDEX: key: "template_source_role"
// BUNDLE-INDEX: value: "compute"
// BUNDLE-INDEX: key: "template_source_op_interface"
// BUNDLE-INDEX: value: "WEFTEmitCLowerableOpInterface"
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
// BUNDLE-INDEX: value: "WEFTEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-template-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
