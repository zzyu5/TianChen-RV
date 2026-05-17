// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z39tcrv_emitc_seed_kernel_seed_rvv_i32_add"
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="return;" --implicit-check-not="int main" --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="rvv-direct-microkernel" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_seed_kernel_seed_rvv_i32_add

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// HEADER: #ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.rvv.origin_plugin: rvv-plugin
// HEADER: tianchenrv.rvv.selected_variant: @seed_rvv_i32_add
// HEADER: tianchenrv.rvv.selected_route: rvv-i32m1-arithmetic-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-i32m1-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_parameter[0]: const int32_t *lhs role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[1]: const int32_t *rhs role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[2]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_abi_parameter[3]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.rvv.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.runtime_avl_abi_parameter: n
// HEADER: tianchenrv.rvv.vl_def: tcrv_rvv.setvl
// HEADER: tianchenrv.rvv.vl_scope: tcrv_rvv.with_vl
// HEADER: tianchenrv.rvv.emitc_loop: emitc.for
// HEADER: tianchenrv.rvv.loop_induction: offset
// HEADER: tianchenrv.rvv.loop_step: full_chunk_vl
// HEADER: tianchenrv.rvv.remaining_avl: n-offset
// HEADER: tianchenrv.rvv.pointer_advance: offset
// HEADER: tianchenrv.rvv.bounded_slice: multi-vl-i32m1-arithmetic
// HEADER: tianchenrv.rvv.multi_vl: supported
// HEADER: #ifdef __cplusplus
// HEADER: extern "C" {
// HEADER: #endif
// HEADER: void tcrv_emitc_seed_kernel_seed_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #ifdef __cplusplus
// HEADER: } /* extern "C" */
// HEADER: #endif

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o"
// BUNDLE-INDEX: component_group: "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @seed_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: component[0]:
// BUNDLE-INDEX: selected_variant: @seed_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter[0]:
// BUNDLE-INDEX: c_name: "lhs"
// BUNDLE-INDEX: role: "lhs-input-buffer"
// BUNDLE-INDEX: runtime_abi_parameter[1]:
// BUNDLE-INDEX: c_name: "rhs"
// BUNDLE-INDEX: role: "rhs-input-buffer"
// BUNDLE-INDEX: runtime_abi_parameter[2]:
// BUNDLE-INDEX: c_name: "out"
// BUNDLE-INDEX: role: "output-buffer"
// BUNDLE-INDEX: runtime_abi_parameter[3]:
// BUNDLE-INDEX: c_name: "n"
// BUNDLE-INDEX: role: "runtime-element-count"
// BUNDLE-INDEX: artifact_metadata[0]:
// BUNDLE-INDEX: key: "rvv_emitc_lowerable_route"
// BUNDLE-INDEX: value: "rvv-i32m1-add-emitc-route"
// BUNDLE-INDEX: key: "rvv_arithmetic_op"
// BUNDLE-INDEX: value: "add"
// BUNDLE-INDEX: key: "tcrv_rvv.runtime_avl_source"
// BUNDLE-INDEX: value: "runtime_abi:n"
// BUNDLE-INDEX: key: "tcrv_rvv.vl_def"
// BUNDLE-INDEX: value: "tcrv_rvv.setvl"
// BUNDLE-INDEX: key: "tcrv_rvv.vl_scope"
// BUNDLE-INDEX: value: "tcrv_rvv.with_vl"
// BUNDLE-INDEX: key: "tcrv_rvv.runtime_abi_order"
// BUNDLE-INDEX: value: "lhs,rhs,out,n"
// BUNDLE-INDEX: key: "tcrv_rvv.runtime_avl_abi_parameter"
// BUNDLE-INDEX: value: "n"
// BUNDLE-INDEX: key: "tcrv_rvv.emitc_loop"
// BUNDLE-INDEX: value: "emitc.for"
// BUNDLE-INDEX: key: "tcrv_rvv.loop_step"
// BUNDLE-INDEX: value: "full_chunk_vl"
// BUNDLE-INDEX: key: "tcrv_rvv.remaining_avl"
// BUNDLE-INDEX: value: "n-offset"
// BUNDLE-INDEX: key: "tcrv_rvv.pointer_advance"
// BUNDLE-INDEX: value: "offset"
// BUNDLE-INDEX: key: "tcrv_rvv.bounded_slice"
// BUNDLE-INDEX: value: "multi-vl-i32m1-arithmetic"
// BUNDLE-INDEX: key: "tcrv_rvv.multi_vl"
// BUNDLE-INDEX: value: "supported"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: file_name: "artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h"
// BUNDLE-INDEX: component_group: "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "header"
// BUNDLE-INDEX: external_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @seed_rvv_i32_add
// BUNDLE-INDEX: role: "dispatch case"
// BUNDLE-INDEX: artifact_kind: "runtime-callable-c-header"
// BUNDLE-INDEX: route: "rvv-i32m1-arithmetic-emitc-route-family.header"
// BUNDLE-INDEX: owner: "rvv-plugin"
// BUNDLE-INDEX: runtime_abi_kind: "plugin-owned-runtime-abi"
// BUNDLE-INDEX: runtime_abi_name: "rvv-i32m1-add-callable-c-abi.v1"
// BUNDLE-INDEX: key: "tcrv_rvv.runtime_avl_source"
// BUNDLE-INDEX: value: "runtime_abi:n"
// BUNDLE-INDEX: key: "tcrv_rvv.bounded_slice"
// BUNDLE-INDEX: value: "multi-vl-i32m1-arithmetic"
// BUNDLE-INDEX: key: "tcrv_rvv.multi_vl"
// BUNDLE-INDEX: value: "supported"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-rvv-intrinsic-object"
// BUNDLE-INDEX: evidence_role: "header-declaration"
