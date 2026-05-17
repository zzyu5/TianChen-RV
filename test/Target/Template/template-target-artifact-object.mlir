// REQUIRES: tianchenrv-local-native-clangxx
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z67tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-template-extension-compute-skeleton-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-template-extension-compute-skeleton-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

module {
  tcrv.exec.kernel @template_emitc_kernel {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
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
