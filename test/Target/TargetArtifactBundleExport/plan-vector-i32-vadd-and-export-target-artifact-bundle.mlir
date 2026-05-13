// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: rm -rf %t.vector.bundle && mkdir %t.vector.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.bundle %s > %t.vector.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.vector.stdout
// RUN: test -s %t.vector.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.vector.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.vector.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.vector.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.vector.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.vector.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.vector.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

module @plan_vector_i32_vadd_bundle_input {
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

  tcrv.exec.target @vector_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.vector.bundle",
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

  func.func @source_vector_bundle_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_bundle_i32_vadd",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_bundle_profile
      } {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    return
  }
}

// IR: tcrv.exec.capability @no_rvv_policy
// IR-SAME: provides = ["build.policy.no_rvv"]
// IR: tcrv.exec.capability @scalar_fallback
// IR-SAME: id = "scalar.fallback"
// IR: tcrv.exec.target @vector_bundle_profile
// IR-SAME: capability_providers = [@no_rvv_policy, @scalar_fallback]
// IR: tcrv.exec.kernel @frontend_vector_bundle_i32_vadd
// IR-SAME: target = @vector_bundle_profile
// IR-SAME: tcrv_frontend_lowering = "i32-vadd"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "must-equal-source-vector-extent"
// IR-SAME: tcrv_frontend_source_authority = "source-vector-transfer-read-write-fixed-extent"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-transfer-fixed-i32-vadd.v1"
// IR-SAME: tcrv_frontend_source_vector_extent = 16 : i64
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: abi_role = "lhs-input-buffer"
// IR-SAME: c_type = "const int32_t *"
// IR: tcrv.exec.runtime_param @abi_runtime_element_count
// IR-SAME: abi_role = "runtime-element-count"
// IR-SAME: c_name = "n"
// IR-SAME: c_type = "size_t"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "must-equal-source-vector-extent"
// IR-SAME: tcrv_frontend_source_authority = "source-vector-transfer-read-write-fixed-extent"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-transfer-fixed-i32-vadd.v1"
// IR-SAME: tcrv_frontend_source_vector_extent = 16 : i64
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
// IR: tcrv_rvv.i32_vadd_microkernel
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: selected_vector_shape = "i32m1"
// IR-SAME: source_kernel = "frontend_vector_bundle_i32_vadd"
// IR: %[[VL:.*]] = tcrv_rvv.setvl
// IR: %[[LHS_VEC:.*]] = tcrv_rvv.i32_load %[[VL]]
// IR-SAME: buffer_role = "lhs-input-buffer"
// IR: %[[RHS_VEC:.*]] = tcrv_rvv.i32_load %[[VL]]
// IR-SAME: buffer_role = "rhs-input-buffer"
// IR: %[[SUM_VEC:.*]] = tcrv_rvv.i32_add %[[LHS_VEC]], %[[RHS_VEC]], %[[VL]]
// IR: tcrv_rvv.i32_store %[[SUM_VEC]], %[[VL]]
// IR-SAME: buffer_role = "output-buffer"
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
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: c_type: "size_t"
// INDEX: role: "runtime-element-count"
// INDEX: runtime_abi_parameter[4]:
// INDEX: c_name: "rvv_available"
// INDEX: c_type: "int"
// INDEX: role: "dispatch-availability-guard"
// INDEX: name: "tcrv_rvv.selected_vector_shape"
// INDEX-NEXT: value: "i32m1"
// INDEX-NEXT: role: "selected-rvv-vector-shape-config"
// INDEX: name: "tcrv_frontend.source_kind"
// INDEX-NEXT: value: "mlir-vector-transfer-fixed-i32-vadd.v1"
// INDEX-NEXT: role: "source-frontdoor-extent-authority"
// INDEX: name: "tcrv_frontend.source_authority"
// INDEX-NEXT: value: "source-vector-transfer-read-write-fixed-extent"
// INDEX-NEXT: role: "source-frontdoor-extent-authority"
// INDEX: name: "tcrv_frontend.source_vector_extent"
// INDEX-NEXT: value: "16"
// INDEX-NEXT: role: "source-frontdoor-extent-authority"
// INDEX: name: "tcrv_frontend.runtime_element_count_constraint"
// INDEX-NEXT: value: "must-equal-source-vector-extent"
// INDEX-NEXT: role: "source-frontdoor-extent-authority"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_element_count_c_name"
// INDEX-NEXT: value: "n"
// INDEX-NEXT: role: "rvv-dispatch-selected-config-contract"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// INDEX: component_role: "header"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
// INDEX: component_role: "object"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"

// SOURCE: /* selected_kernel: @frontend_vector_bundle_i32_vadd */
// SOURCE: /* selected_binary_config: {{.*}}descriptor_element_count=16, fixed_source_vector_extent=16, runtime_element_count_constraint=must-equal-source-vector-extent
// SOURCE: /* source_frontend_extent_authority: source_kind=mlir-vector-transfer-fixed-i32-vadd.v1, source_authority=source-vector-transfer-read-write-fixed-extent, source_vector_extent=16, runtime_element_count_constraint=must-equal-source-vector-extent */
// SOURCE: /* dispatch_runtime_element_count_constraint: n must equal fixed source vector extent 16 before dispatching to RVV or scalar callable branches */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vadd_microkernel_frontend_vector_bundle_i32_vadd_rvv_first_slice */
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vadd_frontend_vector_bundle_i32_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: /* Embedded selected RVV runtime-callable source artifact. */
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// SOURCE: /* arithmetic_family: i32-vadd */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_vector_bundle_i32_vadd_rvv_first_slice
// SOURCE: void tcrv_dispatch_i32_vadd_frontend_vector_bundle_i32_vadd
// SOURCE: // tcrv_emitc.runtime_element_count_constraint=must-equal-fixed-source-vector-extent
// SOURCE: bool {{.*}} = {{.*}} != 16;
// SOURCE: __builtin_trap();

// HEADER: /* source_frontend_extent_authority: source_kind=mlir-vector-transfer-fixed-i32-vadd.v1, source_authority=source-vector-transfer-read-write-fixed-extent, source_vector_extent=16, runtime_element_count_constraint=must-equal-source-vector-extent */
// HEADER: /* dispatch_runtime_element_count_constraint: n must equal fixed source vector extent 16 before dispatching to RVV or scalar callable branches */
// HEADER: void tcrv_dispatch_i32_vadd_frontend_vector_bundle_i32_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);

// OBJ: Format: elf64-littleriscv
// OBJ: Name: tcrv_dispatch_i32_vadd_frontend_vector_bundle_i32_vadd
