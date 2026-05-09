// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.frontend.vmul.bundle && mkdir %t.frontend.vmul.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.vmul.bundle %s > %t.frontend.vmul.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.vmul.stdout
// RUN: test -s %t.frontend.vmul.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c
// RUN: test -s %t.frontend.vmul.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h
// RUN: test -s %t.frontend.vmul.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vadd-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vsub-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1 --implicit-check-not=rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vadd-dispatch --implicit-check-not=tcrv-export-rvv-scalar-i32-vsub-dispatch < %t.frontend.vmul.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" < %t.frontend.vmul.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" < %t.frontend.vmul.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.vmul.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32_vmul_bundle_input {
  tcrv.exec.capability @no_rvv_policy {
    id = "generic.build.profile",
    kind = "build-policy",
    provides = ["build.policy.no_rvv"],
    status = "available"
  }

  tcrv.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }

  tcrv.exec.target @frontend_vmul_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vmul.bundle",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_frontend_bundle_vmul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i32_vmul",
        tcrv_frontend_target = @frontend_vmul_bundle_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vmul"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %product = arith.muli %a, %b : i32
      linalg.yield %product : i32
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
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vmul-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: component[1]:
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vmul-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: role: "runtime-element-count"
// INDEX: runtime_abi_parameter[4]:
// INDEX: role: "dispatch-availability-guard"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vmul-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vmul-dispatch-header"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vmul-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vmul-dispatch-object"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* Scope: one selected RVV i32-vmul dispatch case plus one scalar i32-vmul dispatch fallback. */
// SOURCE: /* rvv_artifact_route_id: tcrv-export-rvv-i32-vmul-microkernel-c */
// SOURCE: /* scalar_artifact_route_id: tcrv-export-scalar-i32-vmul-microkernel-c */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vmul_microkernel_frontend_bundle_i32_vmul_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vmul_microkernel_frontend_bundle_i32_vmul_scalar_fallback_first_slice */
// SOURCE: __riscv_vmul_vv_i32m1
// SOURCE: out[index] = lhs[index] * rhs[index];
// SOURCE: void tcrv_dispatch_i32_vmul_frontend_bundle_i32_vmul

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VMUL_DISPATCH_FRONTEND_BUNDLE_I32_VMUL_H
// HEADER: void tcrv_dispatch_i32_vmul_frontend_bundle_i32_vmul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VMUL_DISPATCH_FRONTEND_BUNDLE_I32_VMUL_H */

// OBJ: Format: elf64-littleriscv
// OBJ-DAG: Name: tcrv_dispatch_i32_vmul_frontend_bundle_i32_vmul
// OBJ-DAG: Name: tcrv_rvv_i32_vmul_microkernel_frontend_bundle_i32_vmul_rvv_first_slice
// OBJ-DAG: Name: tcrv_scalar_i32_vmul_microkernel_frontend_bundle_i32_vmul_scalar_fallback_first_slice
