// REQUIRES: tianchenrv-local-riscv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle %S/tensorext-lite-target-artifact-header.mlir | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z82tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-tensorext-lite-fragment-mma-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// This lower-level exporter test intentionally starts from the materialized
// TensorExtLite target fixture in this directory. It does not exercise the
// source artifact front-door workflow.

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
// HEADER: tianchenrv.tensorext_lite.semantic_role_graph: configure->load_frag->tile_mma->store_frag
// HEADER: tianchenrv.tensorext_lite.typed_role_realization: configure:tel.role.config
// HEADER: void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);

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
// BUNDLE-INDEX: route: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: owner: "tensorext-lite-plugin"
// BUNDLE-INDEX: runtime_abi: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_name: "tensorext-lite-fragment-mma-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 0
// BUNDLE-INDEX: key: "tensorext_lite_emitc_lowerable_route"
// BUNDLE-INDEX: value: "tensorext-lite-fragment-mma-emitc-route"
// BUNDLE-INDEX: key: "tensorext_lite_source_op_interface"
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-tensorext-lite-fragment-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "tensorext-lite-fragment-mma-emitc-route.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
