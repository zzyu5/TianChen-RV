// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=i64-vadd --implicit-check-not=i64-vsub --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: rm -rf %t.frontend.bundle && mkdir %t.frontend.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.bundle %s > %t.frontend.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.stdout
// RUN: test -s %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i64-vmul-dispatch-c.c
// RUN: test -s %t.frontend.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i64-vmul-dispatch-header.h
// RUN: test -s %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i64-vmul-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=i64-vadd --implicit-check-not=i64-vsub --implicit-check-not=int32_t --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=i64-vadd --implicit-check-not=i64-vsub --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i64 --implicit-check-not=__riscv_vsub_vv_i64 --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i64-vmul-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=int32_t --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.frontend.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i64-vmul-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i64-vmul-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i64_vmul_bundle_input {
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

  tcrv.exec.target @frontend_bundle_i64_vmul_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.bundle.i64.vmul",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
    bytes = 16 : i64,
    lanes = 2 : i64,
    sew_bits = 64 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_frontend_bundle_i64_vmul(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i64_vmul",
        tcrv_frontend_target = @frontend_bundle_i64_vmul_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vmul"
      }
      ins(%lhs, %rhs : memref<?xi64>, memref<?xi64>)
      outs(%out : memref<?xi64>) {
    ^bb0(%a: i64, %b: i64, %old: i64):
      %prod = arith.muli %a, %b : i64
      linalg.yield %prod : i64
    }
    return
  }
}

// IR-LABEL: tcrv.exec.kernel @frontend_bundle_i64_vmul
// IR-SAME: target = @frontend_bundle_i64_vmul_profile
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: c_type = "const int64_t *"
// IR: tcrv.exec.mem_window @abi_output_buffer
// IR-SAME: c_type = "int64_t *"
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
// IR-SAME: selected_vector_shape = "i64m1"
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv_scalar.i64_vmul_microkernel
// IR-SAME: element_count = 16 : i64
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: lowering_pipeline = "tcrv-export-scalar-i64-vmul-microkernel-c"
// IR-SAME: runtime_abi_name = "scalar-i64-vmul-runtime-callable-c-function.v1"

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: component_group: "rvv-scalar-i64-vmul-dispatch-external-abi.v1"
// INDEX: external_abi_name: "rvv-scalar-i64-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: route: "tcrv-export-rvv-scalar-i64-vmul-dispatch-c"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i64-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: c_type: "const int64_t *"
// INDEX: c_type: "int64_t *"
// INDEX: c_name: "rvv_available"
// INDEX-NEXT: c_type: "int"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.selected_vector_shape"
// INDEX: value: "i64m1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i64-vmul-dispatch-header"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i64-vmul-dispatch-object"

// SOURCE: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// SOURCE: /* Scope: one selected RVV i64-vmul dispatch case plus one scalar i64-vmul dispatch fallback. */
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.selected_binary_family, value=i64-vmul, role=typed-rvv-binary-source
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_source_op, value=tcrv_rvv.i64_mul, role=typed-rvv-emitc-source-op
// SOURCE: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.selected_binary_family, value=i64-vmul, role=typed-scalar-binary-source
// SOURCE: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.emitc_source_op, value=tcrv_scalar.i64_vmul_microkernel, role=typed-scalar-emitc-source-op
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vmul_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1
// SOURCE: void tcrv_scalar_i64_vmul_microkernel_frontend_bundle_i64_vmul_scalar_fallback_first_slice
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i64_vmul_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i64_mul
// SOURCE: tcrv_scalar_i64_mul
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i64_vmul_frontend_bundle_i64_vmul
// SOURCE: if (rvv_available)
// SOURCE: tcrv_rvv_i64_vmul_microkernel_frontend_bundle_i64_vmul_rvv_first_slice(lhs, rhs, out, n);
// SOURCE: return;
// SOURCE: tcrv_scalar_i64_vmul_microkernel_frontend_bundle_i64_vmul_scalar_fallback_first_slice(lhs, rhs, out, n);

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I64_VMUL_DISPATCH_FRONTEND_BUNDLE_I64_VMUL_H
// HEADER: extern "C" {
// HEADER: void tcrv_dispatch_i64_vmul_frontend_bundle_i64_vmul(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I64_VMUL_DISPATCH_FRONTEND_BUNDLE_I64_VMUL_H */

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable
// OBJ: Name: tcrv_dispatch_i64_vmul_frontend_bundle_i64_vmul
