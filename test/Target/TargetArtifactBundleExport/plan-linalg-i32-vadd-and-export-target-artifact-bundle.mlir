// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.frontend.bundle && mkdir %t.frontend.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.bundle %s > %t.frontend.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.stdout
// RUN: test -s %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c
// RUN: test -s %t.frontend.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h
// RUN: test -s %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c
// RUN: llvm-readobj --file-headers --symbols %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32_vadd_bundle_input {
  tcrv.exec.target @frontend_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.bundle",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_frontend_bundle_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i32_vadd",
        tcrv_frontend_target = @frontend_bundle_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
    }
    return
  }
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c"
// INDEX: component_group: "rvv-i32-vadd-microkernel-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "direct variant"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-microkernel-c"
// INDEX: owner: "rvv-plugin"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.vlenb_bytes"
// INDEX: value: "16"
// INDEX: selected_plan_metadata[1]:
// INDEX: name: "tcrv_rvv.i32_m1_lanes"
// INDEX: value: "4"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h"
// INDEX: component_group: "rvv-i32-vadd-microkernel-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-microkernel-header"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.vlenb_bytes"
// INDEX: evidence_role: "header-declaration"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o"
// INDEX: component_group: "rvv-i32-vadd-microkernel-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-microkernel-object"
// INDEX: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.vlenb_bytes"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* selected_kernel: @frontend_bundle_i32_vadd */
// SOURCE: /* selected_role: direct variant */
// SOURCE: /* selected_march: rv64gcv */
// SOURCE: /* selected_mabi: lp64d */
// SOURCE: /* artifact_kind: runtime-callable-c-source */
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32_vadd_rvv_first_slice
// SOURCE: __riscv_vadd_vv_i32m1

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable
// OBJ: Name: tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32_vadd_rvv_first_slice
