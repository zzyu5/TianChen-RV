// REQUIRES: tianchenrv-local-native-clangxx
// RUN: rm -f %t.o
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z54tcrv_emitc_toy_object_export_toy_template_first_slice"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

module attributes {
  tcrv_toy.source_front_door = "template_compute",
  tcrv_toy.source_kernel = "toy_object_export"
} {
}

// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_toy_object_export_toy_template_first_slice

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// HEADER: #ifndef TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.toy.origin_plugin: toy-plugin
// HEADER: tianchenrv.toy.selected_variant: @toy_template_first_slice
// HEADER: tianchenrv.toy.selected_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.toy.runtime_abi_name: toy-template-compute-runtime-c-abi.v1
// HEADER: tianchenrv.toy.runtime_abi_parameter[0]: size_t toy_value_count role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.toy.emitc_lowerable_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.source_op: tcrv_toy.compute_skeleton
// HEADER: tianchenrv.toy.source_role: compute
// HEADER: tianchenrv.toy.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.toy.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.toy.semantic_role_graph: configure->load->compute->store
// HEADER: tianchenrv.toy.typed_role_realization: configure:toy.role.configure.config_skeleton
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void tcrv_emitc_toy_object_export_toy_template_first_slice(size_t toy_value_count);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o"
// BUNDLE-INDEX: component_group: "toy-template-compute-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @toy_template_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: component[0]:
// BUNDLE-INDEX: selected_variant: @toy_template_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: route: "toy-template-compute-emitc-route"
// BUNDLE-INDEX: owner: "toy-plugin"
// BUNDLE-INDEX: runtime_abi: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 1
// BUNDLE-INDEX: runtime_abi_parameter[0]:
// BUNDLE-INDEX: c_name: "toy_value_count"
// BUNDLE-INDEX: c_type: "size_t"
// BUNDLE-INDEX: role: "runtime-element-count"
// BUNDLE-INDEX: ownership: "target-export-abi-owned"
// BUNDLE-INDEX: artifact_metadata[0]:
// BUNDLE-INDEX: key: "toy_emitc_lowerable_route"
// BUNDLE-INDEX: value: "toy-template-compute-emitc-route"
// BUNDLE-INDEX: key: "toy_source_op"
// BUNDLE-INDEX: value: "tcrv_toy.compute_skeleton"
// BUNDLE-INDEX: key: "toy_source_role"
// BUNDLE-INDEX: value: "compute"
// BUNDLE-INDEX: key: "toy_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: key: "toy_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "toy_semantic_role_graph"
// BUNDLE-INDEX: value: "configure->load->compute->store"
// BUNDLE-INDEX: key: "toy_typed_role_realization"
// BUNDLE-INDEX: value: "configure:toy.role.configure.config_skeleton
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: file_name: "artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h"
// BUNDLE-INDEX: component_group: "toy-template-compute-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "header"
// BUNDLE-INDEX: external_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @toy_template_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "runtime-callable-c-header"
// BUNDLE-INDEX: route: "toy-template-compute-emitc-route.header"
// BUNDLE-INDEX: owner: "toy-plugin"
// BUNDLE-INDEX: runtime_abi: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 1
// BUNDLE-INDEX: c_name: "toy_value_count"
// BUNDLE-INDEX: c_type: "size_t"
// BUNDLE-INDEX: key: "toy_emitc_lowerable_route"
// BUNDLE-INDEX: value: "toy-template-compute-emitc-route"
// BUNDLE-INDEX: key: "toy_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
