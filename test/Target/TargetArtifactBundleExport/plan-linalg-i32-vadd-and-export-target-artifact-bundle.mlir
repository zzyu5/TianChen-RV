// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: rm -rf %t.frontend.bundle && mkdir %t.frontend.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.bundle %s > %t.frontend.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.stdout
// RUN: test -s %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.frontend.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32_vadd_bundle_input {
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

  tcrv.exec.target @frontend_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.bundle",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
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

// IR: tcrv.exec.capability @no_rvv_policy
// IR-SAME: provides = ["build.policy.no_rvv"]
// IR: tcrv.exec.capability @scalar_fallback
// IR-SAME: id = "scalar.fallback"
// IR: tcrv.exec.target @frontend_bundle_profile
// IR-SAME: capability_providers = [@no_rvv_policy, @scalar_fallback]
// IR: tcrv.exec.kernel @frontend_bundle_i32_vadd
// IR-SAME: target = @frontend_bundle_profile
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: abi_role = "lhs-input-buffer"
// IR-SAME: access = "read"
// IR-SAME: c_type = "const int32_t *"
// IR: tcrv.exec.runtime_param @abi_runtime_element_count
// IR-SAME: abi_role = "runtime-element-count"
// IR-SAME: c_name = "n"
// IR-SAME: c_type = "size_t"
// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR-SAME: c_name = "rvv_available"
// IR-SAME: c_type = "int"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch case"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: status = "supported"
// IR-SAME: target = @scalar_fallback_first_slice

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: component[1]:
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
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
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.vlenb_bytes"
// INDEX: value: "16"
// INDEX: role: "selected-rvv-capacity-fact"
// INDEX: selected_plan_metadata[1]:
// INDEX: name: "tcrv_rvv.i32_m1_lanes"
// INDEX: value: "4"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
// INDEX: runtime_abi_parameter[4]:
// INDEX: role: "dispatch-availability-guard"
// INDEX: evidence_role: "header-declaration"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"
// INDEX: runtime_abi_parameter[0]:
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[4]:
// INDEX: role: "dispatch-availability-guard"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// SOURCE: /* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
// SOURCE: /* selected_kernel: @frontend_bundle_i32_vadd */
// SOURCE: /* rvv_selected_variant: @rvv_first_slice */
// SOURCE: /* rvv_selected_role: dispatch case */
// SOURCE: /* scalar_selected_variant: @scalar_fallback_first_slice */
// SOURCE: /* scalar_selected_role: dispatch fallback */
// SOURCE: /* dispatch_runtime_param[1]: symbol=@abi_dispatch_availability_guard, abi_role=dispatch-availability-guard, c_name=rvv_available, c_type=int, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// SOURCE: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32_vadd_rvv_first_slice
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: void tcrv_scalar_i32_vadd_microkernel_frontend_bundle_i32_vadd_scalar_fallback_first_slice
// SOURCE: out[index] = lhs[index] + rhs[index];
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i32_vadd_frontend_bundle_i32_vadd
// SOURCE: if (rvv_available)
// SOURCE: tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32_vadd_rvv_first_slice(lhs, rhs, out, n);
// SOURCE: return;
// SOURCE: tcrv_scalar_i32_vadd_microkernel_frontend_bundle_i32_vadd_scalar_fallback_first_slice(lhs, rhs, out, n);

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_FRONTEND_BUNDLE_I32_VADD_H
// HEADER: extern "C" {
// HEADER: void tcrv_dispatch_i32_vadd_frontend_bundle_i32_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_FRONTEND_BUNDLE_I32_VADD_H */

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable
// OBJ: Name: tcrv_dispatch_i32_vadd_frontend_bundle_i32_vadd
