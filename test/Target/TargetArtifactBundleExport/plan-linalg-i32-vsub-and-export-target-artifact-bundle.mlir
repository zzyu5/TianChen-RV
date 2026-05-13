// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.frontend.vsub.bundle && mkdir %t.frontend.vsub.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.vsub.bundle %s > %t.frontend.vsub.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.vsub.stdout
// RUN: test -s %t.frontend.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: test -s %t.frontend.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: test -s %t.frontend.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vadd-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vadd-dispatch < %t.frontend.vsub.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" < %t.frontend.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not=i32_vadd --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" < %t.frontend.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32_vsub_bundle_input {
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

  tcrv.exec.target @frontend_vsub_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vsub.bundle",
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

  func.func @source_frontend_bundle_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i32_vsub",
        tcrv_frontend_target = @frontend_vsub_bundle_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vsub"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %diff = arith.subi %a, %b : i32
      linalg.yield %diff : i32
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
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: component[1]:
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: c_type: "const int32_t *"
// INDEX: role: "lhs-input-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[1]:
// INDEX: c_name: "rhs"
// INDEX: c_type: "const int32_t *"
// INDEX: role: "rhs-input-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[2]:
// INDEX: c_name: "out"
// INDEX: c_type: "int32_t *"
// INDEX: role: "output-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: c_type: "size_t"
// INDEX: role: "runtime-element-count"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[4]:
// INDEX: c_name: "rvv_available"
// INDEX: c_type: "int"
// INDEX: role: "dispatch-availability-guard"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-header"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-object"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// SOURCE: /* selected_binary_config: dtype=i32, family=i32-vsub
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: selected_role=dispatch case */
// SOURCE: /* rvv_artifact_route_id: tcrv-export-rvv-i32-vsub-microkernel-c */
// SOURCE: /* rvv_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* rvv_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* rvv_selected_plan_metadata[14]: name=tcrv_rvv.runtime_avl_source, value=runtime-element-count-abi-parameter, role=rvv-runtime-vl-avl-boundary
// SOURCE: /* rvv_selected_plan_metadata[16]: name=tcrv_rvv.runtime_vl_source, value=tcrv_rvv.setvl, role=rvv-runtime-vl-avl-boundary
// SOURCE: /* scalar_artifact_route_id: tcrv-export-scalar-i32-vsub-microkernel-c */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_bundle_i32_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_bundle_i32_vsub_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* dispatch_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vsub_frontend_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: __riscv_vsub_vv_i32m1
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i32_vsub_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_sub
// SOURCE: tcrv_scalar_i32_sub
// SOURCE: void tcrv_dispatch_i32_vsub_frontend_bundle_i32_vsub

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_BUNDLE_I32_VSUB_H
// HEADER: void tcrv_dispatch_i32_vsub_frontend_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_BUNDLE_I32_VSUB_H */

// OBJ: Format: elf64-littleriscv
// OBJ-DAG: Name: tcrv_dispatch_i32_vsub_frontend_bundle_i32_vsub
// OBJ-DAG: Name: tcrv_rvv_i32_vsub_microkernel_frontend_bundle_i32_vsub_rvv_first_slice
// OBJ-DAG: Name: tcrv_scalar_i32_vsub_microkernel_frontend_bundle_i32_vsub_scalar_fallback_first_slice
