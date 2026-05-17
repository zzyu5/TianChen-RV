// REQUIRES: tianchenrv-local-native-clangxx
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/Toy/toy-template-source-front-door.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z54tcrv_emitc_toy_header_export_toy_template_first_slice"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv" < %t.bundle/artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="tcrv_rvv" < %t.bundle/tianchenrv-target-artifact-bundle.index

// This non-RVV case proves the common translate front door consumes plugin
// source-front-door registrations instead of hard-coding RVV source semantics.

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_toy_header_export_toy_template_first_slice

// HEADER: #ifndef TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: tianchenrv.toy.origin_plugin: toy-plugin
// HEADER: tianchenrv.toy.selected_variant: @toy_template_first_slice
// HEADER: tianchenrv.toy.selected_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.toy.runtime_abi_name: toy-template-compute-runtime-c-abi.v1
// HEADER: tianchenrv.toy.runtime_abi_parameter[0]: size_t toy_value_count role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.toy.emitc_lowerable_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: extern "C" {
// HEADER: void tcrv_emitc_toy_header_export_toy_template_first_slice(size_t toy_value_count);

// INDEX: tianchenrv.target_artifact_bundle.version: 1
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
// INDEX: value: "TCRVEmitCLowerableOpInterface"
// INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// INDEX: evidence_role: "relocatable-object"
// INDEX-LABEL: artifact[1]:
// INDEX: route: "toy-template-compute-emitc-route.header"
// INDEX: evidence_role: "header-declaration"
