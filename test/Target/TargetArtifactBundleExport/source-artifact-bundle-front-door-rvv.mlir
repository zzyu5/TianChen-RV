// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" < %t.bundle/tianchenrv-target-artifact-bundle.index

// This file intentionally has no standalone input. The RUN lines prove that
// the source MLIR fixture reaches the target object/header bundle through one
// tcrv-translate production front door, without a manual tcrv-opt pipe.

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER: tianchenrv.rvv.selected_variant: @vector_source_rvv_i32_add
// HEADER: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.emitc_loop: emitc.for
// HEADER: tianchenrv.rvv.multi_vl: supported
// HEADER: extern "C" {
// HEADER: void tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 2
// INDEX-LABEL: artifact[0]:
// INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// INDEX: selected_variant: @vector_source_rvv_i32_add
// INDEX: role: "dispatch case"
// INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family"
// INDEX: owner: "rvv-plugin"
// INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: runtime_abi_parameter[1]:
// INDEX: c_name: "rhs"
// INDEX: runtime_abi_parameter[2]:
// INDEX: c_name: "out"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: key: "rvv_emitc_lowerable_route"
// INDEX: value: "rvv-i32m1-add-emitc-route"
// INDEX: key: "tcrv_rvv.emitc_loop"
// INDEX: value: "emitc.for"
// INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// INDEX: evidence_role: "relocatable-object"
// INDEX-LABEL: artifact[1]:
// INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family.header"
// INDEX: evidence_role: "header-declaration"
