// REQUIRES: weft-local-native-clangxx
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-translate --weft-source-artifact-bundle-front-door --weft-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/Toy/toy-template-source-front-door.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z54weft_emitc_toy_header_export_toy_template_first_slice"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv" < %t.bundle/artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="weft_rvv" < %t.bundle/weft-target-artifact-bundle.index

// This non-RVV case proves the common translate front door consumes plugin
// source-front-door registrations instead of hard-coding RVV source semantics.

// STDOUT: weft.target_artifact_bundle_export: complete
// STDOUT: index_file: "weft-target-artifact-bundle.index"

// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_toy_header_export_toy_template_first_slice

// HEADER: #ifndef WEFT_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: weft.toy.origin_plugin: toy-plugin
// HEADER: weft.toy.selected_variant: @toy_template_first_slice
// HEADER: weft.toy.selected_route: toy-template-compute-emitc-route
// HEADER: weft.toy.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.toy.runtime_abi_name: toy-template-compute-runtime-c-abi.v1
// HEADER: weft.toy.runtime_abi_parameter[0]: size_t toy_value_count role=runtime-element-count ownership=target-export-abi-owned
// HEADER: weft.toy.emitc_lowerable_route: toy-template-compute-emitc-route
// HEADER: weft.toy.source_op_interface: WEFTEmitCLowerableOpInterface
// HEADER: extern "C" {
// HEADER: void weft_emitc_toy_header_export_toy_template_first_slice(size_t toy_value_count);

// INDEX: weft.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 2
// INDEX-LABEL: artifact[0]:
// INDEX: selected_variant: @toy_template_first_slice
// INDEX: route: "toy-template-compute-emitc-route"
// INDEX: owner: "toy-plugin"
// INDEX: runtime_abi_name: "toy-template-compute-runtime-c-abi.v1"
// INDEX: runtime_abi_parameter_count: 1
// INDEX: c_name: "toy_value_count"
// INDEX: key: "toy_emitc_lowerable_route"
// INDEX: value: "toy-template-compute-emitc-route"
// INDEX: key: "toy_source_op_interface"
// INDEX: value: "WEFTEmitCLowerableOpInterface"
// INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// INDEX: evidence_role: "relocatable-object"
// INDEX-LABEL: artifact[1]:
// INDEX: route: "toy-template-compute-emitc-route.header"
// INDEX: evidence_role: "header-declaration"
