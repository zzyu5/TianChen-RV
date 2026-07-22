// REQUIRES: weft-local-rvv-object-clang
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-opt %S/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact-bundle --weft-target-artifact-bundle-output-dir=%t.bundle | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="source-front-door" --implicit-check-not="weft_rvv.i32_" < %t.bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not="descriptor" --implicit-check-not="direct-C" --implicit-check-not="source-export" --implicit-check-not="source-front-door" --implicit-check-not="weft_rvv.i32_" < %t.bundle/weft-target-artifact-bundle.index

// This target artifact bundle test intentionally starts from the existing
// pre-realized selected-body fixture. It proves the selected-boundary producer,
// provider-built route, and target artifact validator reach the generated
// object/header bundle before runtime evidence is collected by the script.

// BUNDLE-STDOUT: weft.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "weft-target-artifact-bundle.index"

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_pre_dual_cmp_mask_select_kernel_pre_rvv_dual_cmp_mask_select

// HEADER: #ifndef WEFT_RVV_EXACT_BODY_ARTIFACT_H
// HEADER-DAG: weft.rvv.selected_variant: @pre_rvv_dual_cmp_mask_select
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-DAG: weft.rvv.runtime_abi_parameter[0]: const int32_t *cmp_lhs_a role=lhs-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[1]: int32_t rhs_scalar_a role=rhs-scalar-value ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[2]: const int32_t *cmp_lhs_b role=rhs-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[3]: int32_t rhs_scalar_b role=rhs-secondary-scalar-value ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[4]: const int32_t *true_value role=true-value-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[5]: const int32_t *false_value role=false-value-input-buffer ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[6]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER-DAG: weft.rvv.runtime_abi_parameter[7]: size_t n role=runtime-element-count ownership=target-export-abi-owned
// HEADER: void weft_emitc_pre_dual_cmp_mask_select_kernel_pre_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);

// INDEX: weft.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 2
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-exact-typed-body-callable-c-abi.v2"
// INDEX: selected_variant: @pre_rvv_dual_cmp_mask_select
// INDEX: route: "rvv-generic-typed-body-emitc-route-family"
// INDEX: owner: "rvv-plugin"
// INDEX: runtime_abi_parameter_count: 8
// INDEX: c_name: "cmp_lhs_a"
// INDEX: role: "lhs-input-buffer"
// INDEX: c_name: "rhs_scalar_a"
// INDEX: role: "rhs-scalar-value"
// INDEX: c_name: "cmp_lhs_b"
// INDEX: role: "rhs-input-buffer"
// INDEX: c_name: "rhs_scalar_b"
// INDEX: role: "rhs-secondary-scalar-value"
// INDEX: c_name: "true_value"
// INDEX: role: "true-value-input-buffer"
// INDEX: c_name: "false_value"
// INDEX: role: "false-value-input-buffer"
// INDEX: c_name: "out"
// INDEX: role: "output-buffer"
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
