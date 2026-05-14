// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.vsub.bundle && mkdir %t.vsub.bundle
// RUN: tcrv-opt %S/../RVVMicrokernel/rvv-microkernel-family-sub.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vsub.bundle > %t.vsub.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.vsub.stdout
// RUN: test -s %t.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-i32-vsub-microkernel-c.c
// RUN: test -s %t.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-i32-vsub-microkernel-header.h
// RUN: test -s %t.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-i32-vsub-microkernel-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-i32-vadd-microkernel-external-abi.v1 --implicit-check-not=rvv-i32-vadd-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-microkernel-header --implicit-check-not=tcrv-export-rvv-microkernel-object < %t.vsub.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not="int main(void)" --implicit-check-not="_self_check" < %t.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-i32-vsub-microkernel-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="__riscv" --implicit-check-not="int main" --implicit-check-not="_self_check" < %t.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-i32-vsub-microkernel-header.h
// RUN: llvm-readobj --file-headers --symbols %t.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-i32-vsub-microkernel-object.o | FileCheck %s --check-prefixes=OBJ,VSUB-OBJ --implicit-check-not="Name: main"

module @target_artifact_bundle_rvv_vsub_test_anchor {
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-i32-vsub-microkernel-c.c"
// INDEX: component_group: "rvv-i32-vsub-microkernel-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: selected_variant: @rvv_sub_slice
// INDEX: role: "direct variant"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-i32-vsub-microkernel-c"
// INDEX: owner: "rvv-plugin"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: role: "runtime-element-count"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-i32-vsub-microkernel-header.h"
// INDEX: component_group: "rvv-i32-vsub-microkernel-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-i32-vsub-microkernel-header"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: evidence_role: "header-declaration"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-i32-vsub-microkernel-object.o"
// INDEX: component_group: "rvv-i32-vsub-microkernel-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-i32-vsub-microkernel-object"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vsub-runtime-callable-c-function.v1"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_sub -> tcrv_rvv.i32_store */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vsub_microkernel_export_i32_vsub_rvv_sub_slice__tcrv_emitc_body
// SOURCE: __riscv_vsub_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_export_i32_vsub_rvv_sub_slice

// HEADER: /* emitc_body_mapping_source: selected_plan_metadata */
// HEADER: /* emitc_body_mapping_status: selected RVV EmitC body mapping was validated before source/header/object artifact export; this header remains declaration-only and carries no intrinsic include. */
// HEADER: #ifndef TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_EXPORT_I32_VSUB_RVV_SUB_SLICE_H
// HEADER: void tcrv_rvv_i32_vsub_microkernel_export_i32_vsub_rvv_sub_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif /* TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_EXPORT_I32_VSUB_RVV_SUB_SLICE_H */

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable
// VSUB-OBJ: Name: tcrv_rvv_i32_vsub_microkernel_export_i32_vsub_rvv_sub_slice
