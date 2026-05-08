// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.direct.bundle && mkdir %t.direct.bundle
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.direct.bundle > %t.direct.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.direct.stdout
// RUN: test -s %t.direct.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c
// RUN: test -s %t.direct.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h
// RUN: test -s %t.direct.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o
// RUN: FileCheck %s --check-prefix=DIRECT-INDEX --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.direct.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=DIRECT-SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.direct.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c
// RUN: FileCheck %s --check-prefix=DIRECT-HEADER --implicit-check-not=") {" --implicit-check-not="__riscv" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.direct.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h
// RUN: llvm-readobj --file-headers --symbols %t.direct.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o | FileCheck %s --check-prefixes=OBJ,DIRECT-OBJ --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=DIRECT-SOURCE-FRONTDOOR --implicit-check-not="#ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=DIRECT-HEADER-FRONTDOOR --implicit-check-not="__riscv_vadd"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.direct.single.o
// RUN: llvm-readobj --file-headers --symbols %t.direct.single.o | FileCheck %s --check-prefixes=OBJ,DIRECT-OBJ --implicit-check-not="Name: main"

// RUN: rm -rf %t.dispatch.bundle && mkdir %t.dispatch.bundle
// RUN: tcrv-opt %S/../RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.dispatch.bundle > %t.dispatch.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.dispatch.stdout
// RUN: test -s %t.dispatch.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.dispatch.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.dispatch.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=DISPATCH-INDEX --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=DISPATCH-SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: FileCheck %s --check-prefix=DISPATCH-HEADER --implicit-check-not=") {" --implicit-check-not="__riscv" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.dispatch.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o | FileCheck %s --check-prefixes=OBJ,DISPATCH-OBJ --implicit-check-not="Name: main"
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @target_artifact_bundle_positive_input {
  tcrv.exec.kernel @bundle_i32_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
  }
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// DIRECT-INDEX: tianchenrv.target_artifact_bundle.version: 1
// DIRECT-INDEX: bundle_status: "complete"
// DIRECT-INDEX: artifact_count: 3
// DIRECT-INDEX: artifact[0]:
// DIRECT-INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c"
// DIRECT-INDEX: selected_variant: @rvv_first_slice
// DIRECT-INDEX: role: "direct variant"
// DIRECT-INDEX: component[0]:
// DIRECT-INDEX: selected_variant: @rvv_first_slice
// DIRECT-INDEX: artifact_kind: "runtime-callable-c-source"
// DIRECT-INDEX: route: "tcrv-export-rvv-microkernel-c"
// DIRECT-INDEX: owner: "rvv-plugin"
// DIRECT-INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// DIRECT-INDEX: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// DIRECT-INDEX: evidence_role: "compiler-artifact"
// DIRECT-INDEX: artifact[1]:
// DIRECT-INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h"
// DIRECT-INDEX: artifact_kind: "runtime-callable-c-header"
// DIRECT-INDEX: route: "tcrv-export-rvv-microkernel-header"
// DIRECT-INDEX: evidence_role: "header-declaration"
// DIRECT-INDEX: artifact[2]:
// DIRECT-INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o"
// DIRECT-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// DIRECT-INDEX: route: "tcrv-export-rvv-microkernel-object"
// DIRECT-INDEX: evidence_role: "relocatable-object"

// DIRECT-SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// DIRECT-SOURCE: /* selected_kernel: @bundle_i32_vadd */
// DIRECT-SOURCE: /* selected_role: direct variant */
// DIRECT-SOURCE: /* artifact_kind: runtime-callable-c-source */
// DIRECT-SOURCE: void tcrv_rvv_i32_vadd_microkernel_bundle_i32_vadd_rvv_first_slice
// DIRECT-SOURCE: __riscv_vadd_vv_i32m1

// DIRECT-HEADER: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_BUNDLE_I32_VADD_RVV_FIRST_SLICE_H
// DIRECT-HEADER: void tcrv_rvv_i32_vadd_microkernel_bundle_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// DIRECT-HEADER: #endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_BUNDLE_I32_VADD_RVV_FIRST_SLICE_H */

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable
// DIRECT-OBJ: Name: tcrv_rvv_i32_vadd_microkernel_bundle_i32_vadd_rvv_first_slice

// DIRECT-SOURCE-FRONTDOOR: /* artifact_kind: runtime-callable-c-source */
// DIRECT-SOURCE-FRONTDOOR: void tcrv_rvv_i32_vadd_microkernel_bundle_i32_vadd_rvv_first_slice
// DIRECT-SOURCE-FRONTDOOR: __riscv_vadd_vv_i32m1

// DIRECT-HEADER-FRONTDOOR: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_BUNDLE_I32_VADD_RVV_FIRST_SLICE_H
// DIRECT-HEADER-FRONTDOOR: void tcrv_rvv_i32_vadd_microkernel_bundle_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// DISPATCH-INDEX: tianchenrv.target_artifact_bundle.version: 1
// DISPATCH-INDEX: bundle_status: "complete"
// DISPATCH-INDEX: artifact_count: 3
// DISPATCH-INDEX: artifact[0]:
// DISPATCH-INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
// DISPATCH-INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// DISPATCH-INDEX: component_role: "source"
// DISPATCH-INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// DISPATCH-INDEX: selected_surface: "dispatch"
// DISPATCH-INDEX: component[0]:
// DISPATCH-INDEX: selected_variant: @rvv_first_slice
// DISPATCH-INDEX: role: "dispatch case"
// DISPATCH-INDEX: component[1]:
// DISPATCH-INDEX: selected_variant: @scalar_fallback_first_slice
// DISPATCH-INDEX: role: "dispatch fallback"
// DISPATCH-INDEX: artifact_kind: "runtime-callable-c-source"
// DISPATCH-INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// DISPATCH-INDEX: owner: "rvv-scalar-dispatch-target"
// DISPATCH-INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// DISPATCH-INDEX: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// DISPATCH-INDEX: runtime_abi_parameter[0]:
// DISPATCH-INDEX: c_name: "lhs"
// DISPATCH-INDEX: c_type: "const int32_t *"
// DISPATCH-INDEX: role: "lhs-input-buffer"
// DISPATCH-INDEX: ownership: "target-export-abi-owned"
// DISPATCH-INDEX: runtime_abi_parameter[1]:
// DISPATCH-INDEX: c_name: "rhs"
// DISPATCH-INDEX: role: "rhs-input-buffer"
// DISPATCH-INDEX: runtime_abi_parameter[2]:
// DISPATCH-INDEX: c_name: "out"
// DISPATCH-INDEX: c_type: "int32_t *"
// DISPATCH-INDEX: role: "output-buffer"
// DISPATCH-INDEX: runtime_abi_parameter[3]:
// DISPATCH-INDEX: c_name: "n"
// DISPATCH-INDEX: c_type: "size_t"
// DISPATCH-INDEX: role: "runtime-element-count"
// DISPATCH-INDEX: runtime_abi_parameter[4]:
// DISPATCH-INDEX: c_name: "rvv_available"
// DISPATCH-INDEX: c_type: "int"
// DISPATCH-INDEX: role: "dispatch-availability-guard"
// DISPATCH-INDEX: ownership: "target-export-abi-owned"
// DISPATCH-INDEX: evidence_role: "compiler-artifact"
// DISPATCH-INDEX: artifact[1]:
// DISPATCH-INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// DISPATCH-INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// DISPATCH-INDEX: component_role: "header"
// DISPATCH-INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// DISPATCH-INDEX: artifact_kind: "runtime-callable-c-header"
// DISPATCH-INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
// DISPATCH-INDEX: runtime_abi_parameter[4]:
// DISPATCH-INDEX: role: "dispatch-availability-guard"
// DISPATCH-INDEX: evidence_role: "header-declaration"
// DISPATCH-INDEX: artifact[2]:
// DISPATCH-INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
// DISPATCH-INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// DISPATCH-INDEX: component_role: "object"
// DISPATCH-INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// DISPATCH-INDEX: artifact_kind: "riscv-elf-relocatable-object"
// DISPATCH-INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"
// DISPATCH-INDEX: runtime_abi_parameter[0]:
// DISPATCH-INDEX: role: "lhs-input-buffer"
// DISPATCH-INDEX: runtime_abi_parameter[4]:
// DISPATCH-INDEX: role: "dispatch-availability-guard"
// DISPATCH-INDEX: evidence_role: "relocatable-object"

// DISPATCH-SOURCE: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// DISPATCH-SOURCE: /* selected_kernel: @dispatch_vadd */
// DISPATCH-SOURCE: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// DISPATCH-SOURCE: /* dispatch_fallback_link: target=@scalar_fallback_first_slice, selected_scalar_callable=@scalar_fallback_first_slice */
// DISPATCH-SOURCE: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// DISPATCH-SOURCE: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// DISPATCH-SOURCE-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd
// DISPATCH-SOURCE: if (rvv_available)
// DISPATCH-SOURCE: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice(lhs, rhs, out, n);
// DISPATCH-SOURCE: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice(lhs, rhs, out, n);

// DISPATCH-HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// DISPATCH-HEADER: void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// DISPATCH-HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H */

// DISPATCH-OBJ: Name: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// DISPATCH-OBJ: Name: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// DISPATCH-OBJ: Name: tcrv_dispatch_i32_vadd_dispatch_vadd

// HELP: --tcrv-export-target-artifact-bundle
// HELP: --tcrv-target-artifact-bundle-output-dir
