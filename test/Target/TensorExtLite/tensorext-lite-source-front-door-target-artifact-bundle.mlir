// REQUIRES: tianchenrv-local-riscv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z82tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-tensorext-lite-fragment-mma-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// This file carries no standalone input. The positive route intentionally
// starts from the TensorExtLite source-front-door fixture through the
// one-command source artifact bundle front door, materializes the plugin-owned
// EmitC module, packages the emitted C++ as a relocatable object, and writes a
// coherent object+declaration header bundle without claiming TensorExtLite
// runtime correctness or performance.

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice

// HEADER: #ifndef TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.tensorext_lite.origin_plugin: tensorext-lite-plugin
// HEADER: tianchenrv.tensorext_lite.selected_variant: @tensorext_lite_tile_mma_first_slice
// HEADER: tianchenrv.tensorext_lite.selected_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.tensorext_lite.runtime_abi_name: tensorext-lite-fragment-mma-runtime-c-abi.v1
// HEADER: tianchenrv.tensorext_lite.emitc_lowerable_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.role_sequence: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_ops: tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton
// HEADER: tianchenrv.tensorext_lite.source_roles: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.tensorext_lite.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.tensorext_lite.extension_archetype: fragment-mma-like
// HEADER: tianchenrv.tensorext_lite.semantic_role_graph: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.common_interface_realization: configure=TCRVExtensionOpInterface
// HEADER: tianchenrv.tensorext_lite.typed_role_realization: configure:tel.role.config
// HEADER: tianchenrv.tensorext_lite.emitc_route_mapping: tensorext-lite-fragment-mma-emitc-route
// HEADER: tianchenrv.tensorext_lite.evidence_profile: parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o"
// BUNDLE-INDEX: component_group: "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @tensorext_lite_tile_mma_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: component[0]:
// BUNDLE-INDEX: selected_variant: @tensorext_lite_tile_mma_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: route: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: owner: "tensorext-lite-plugin"
// BUNDLE-INDEX: runtime_abi: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 0
// BUNDLE-INDEX: artifact_metadata[0]:
// BUNDLE-INDEX: key: "tensorext_lite_emitc_lowerable_route"
// BUNDLE-INDEX: value: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: key: "tensorext_lite_role_sequence"
// BUNDLE-INDEX: value: "configure->load_frag->tile_mma->store_frag"
// BUNDLE-INDEX: key: "tensorext_lite_source_ops"
// BUNDLE-INDEX: value: "tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton"
// BUNDLE-INDEX: key: "tensorext_lite_source_roles"
// BUNDLE-INDEX: value: "configure->load_frag->tile_mma->store_frag"
// BUNDLE-INDEX: key: "tensorext_lite_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: key: "tensorext_lite_construction_protocol"
// BUNDLE-INDEX: value: "extension-family-construction-protocol.v1"
// BUNDLE-INDEX: key: "tensorext_lite_extension_archetype"
// BUNDLE-INDEX: value: "fragment-mma-like"
// BUNDLE-INDEX: key: "tensorext_lite_semantic_role_graph"
// BUNDLE-INDEX: value: "configure->load_frag->tile_mma->store_frag"
// BUNDLE-INDEX: key: "tensorext_lite_common_interface_realization"
// BUNDLE-INDEX: value: "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"
// BUNDLE-INDEX: key: "tensorext_lite_typed_role_realization"
// BUNDLE-INDEX: value: "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton
// BUNDLE-INDEX: key: "tensorext_lite_emitc_route_mapping"
// BUNDLE-INDEX: value: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: key: "tensorext_lite_evidence_profile"
// BUNDLE-INDEX: value: "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-tensorext-lite-fragment-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: file_name: "artifact-1-runtime-callable-c-header-tensorext-lite-fragment-mma-emitc-route.header.h"
// BUNDLE-INDEX: component_group: "tensorext-lite-fragment-mma-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "header"
// BUNDLE-INDEX: external_abi_name: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @tensorext_lite_tile_mma_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: artifact_kind: "runtime-callable-c-header"
// BUNDLE-INDEX: route: "tensorext-lite-fragment-mma-emitc-route.header"
// BUNDLE-INDEX: owner: "tensorext-lite-plugin"
// BUNDLE-INDEX: runtime_abi: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 0
// BUNDLE-INDEX: key: "tensorext_lite_emitc_lowerable_route"
// BUNDLE-INDEX: value: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: key: "tensorext_lite_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: key: "tensorext_lite_extension_archetype"
// BUNDLE-INDEX: value: "fragment-mma-like"
// BUNDLE-INDEX: key: "tensorext_lite_emitc_route_mapping"
// BUNDLE-INDEX: value: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: key: "tensorext_lite_evidence_profile"
// BUNDLE-INDEX: value: "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-tensorext-lite-fragment-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
