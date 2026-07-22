// REQUIRES: weft-local-riscv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-tensorext-lite-fragment-mma-emitc-route.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z82weft_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-tensorext-lite-fragment-mma-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/weft-target-artifact-bundle.index

// This file carries no standalone input. The positive route intentionally
// starts from the TensorExtLite source-front-door fixture through the
// one-command source artifact bundle front door, materializes the plugin-owned
// EmitC module, packages the emitted C++ as a relocatable object, and writes a
// coherent object+declaration header bundle without claiming TensorExtLite
// runtime correctness or performance.

// BUNDLE-STDOUT: weft.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "weft-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice

// HEADER: #ifndef WEFT_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: weft.tensorext_lite.origin_plugin: tensorext-lite-plugin
// HEADER: weft.tensorext_lite.selected_variant: @tensorext_lite_tile_mma_first_slice
// HEADER: weft.tensorext_lite.selected_route: tensorext-lite-fragment-mma-emitc-route
// HEADER: weft.tensorext_lite.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.tensorext_lite.runtime_abi_name: tensorext-lite-fragment-mma-runtime-c-abi.v1
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void weft_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: weft.target_artifact_bundle.version: 1
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
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-tensorext-lite-fragment-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
