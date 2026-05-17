// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" < %t.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.sub.bundle && mkdir %t.sub.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.sub.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj --symbols %t.sub.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL-SUB
// RUN: FileCheck %s --check-prefix=HEADER-SUB --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.sub.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX-SUB --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" < %t.sub.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.mul.bundle && mkdir %t.mul.bundle
// RUN: tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=%t.mul.bundle %S/../../Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir | FileCheck %s --check-prefix=STDOUT
// RUN: llvm-readobj --symbols %t.mul.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL-MUL
// RUN: FileCheck %s --check-prefix=HEADER-MUL --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.mul.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX-MUL --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" < %t.mul.bundle/tianchenrv-target-artifact-bundle.index

// This file intentionally has no standalone input. The RUN lines prove that
// RVV add/sub/mul source MLIR fixtures reach target object/header bundles
// through one tcrv-translate production front door, without a manual tcrv-opt
// pipe.

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add
// SYMBOL-SUB: Name: tcrv_emitc_vector_source_sub_kernel_vector_source_sub_rvv_i32_sub
// SYMBOL-MUL: Name: tcrv_emitc_vector_source_mul_kernel_vector_source_mul_rvv_i32_mul

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

// HEADER-SUB: tianchenrv.rvv.selected_variant: @vector_source_sub_rvv_i32_sub
// HEADER-SUB: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER-SUB: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-sub-callable-c-abi.v1
// HEADER-SUB: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER-SUB: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER-SUB: tianchenrv.rvv.multi_vl: supported
// HEADER-SUB: void tcrv_emitc_vector_source_sub_kernel_vector_source_sub_rvv_i32_sub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-MUL: tianchenrv.rvv.selected_variant: @vector_source_mul_rvv_i32_mul
// HEADER-MUL: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER-MUL: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-mul-callable-c-abi.v1
// HEADER-MUL: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER-MUL: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER-MUL: tianchenrv.rvv.multi_vl: supported
// HEADER-MUL: void tcrv_emitc_vector_source_mul_kernel_vector_source_mul_rvv_i32_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

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

// INDEX-SUB: bundle_status: "complete"
// INDEX-SUB: external_abi_name: "rvv-i32m1-sub-callable-c-abi.v1"
// INDEX-SUB: selected_variant: @vector_source_sub_rvv_i32_sub
// INDEX-SUB: route: "rvv-i32m1-arithmetic-emitc-route-family"
// INDEX-SUB: owner: "rvv-plugin"
// INDEX-SUB: runtime_abi_name: "rvv-i32m1-sub-callable-c-abi.v1"
// INDEX-SUB: key: "rvv_emitc_lowerable_route"
// INDEX-SUB: value: "rvv-i32m1-sub-emitc-route"
// INDEX-SUB: key: "rvv_arithmetic_op"
// INDEX-SUB: value: "sub"
// INDEX-SUB: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"

// INDEX-MUL: bundle_status: "complete"
// INDEX-MUL: external_abi_name: "rvv-i32m1-mul-callable-c-abi.v1"
// INDEX-MUL: selected_variant: @vector_source_mul_rvv_i32_mul
// INDEX-MUL: route: "rvv-i32m1-arithmetic-emitc-route-family"
// INDEX-MUL: owner: "rvv-plugin"
// INDEX-MUL: runtime_abi_name: "rvv-i32m1-mul-callable-c-abi.v1"
// INDEX-MUL: key: "rvv_emitc_lowerable_route"
// INDEX-MUL: value: "rvv-i32m1-mul-emitc-route"
// INDEX-MUL: key: "rvv_arithmetic_op"
// INDEX-MUL: value: "mul"
// INDEX-MUL: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
