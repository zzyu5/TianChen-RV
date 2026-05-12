// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=i32-vsub-microkernel.v1 --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vsub_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-c | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not="out[index]" --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-header | FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not="out[index]" --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c | FileCheck %s --check-prefix=HARNESS --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=ROUTE-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import sys; text=sys.stdin.read(); text=text.replace("tcrv_rvv.i32_vmul_microkernel", "tcrv_rvv.i32_vadd_microkernel", 1).replace("tcrv_rvv.i32_mul", "tcrv_rvv.i32_add", 1); sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-c 2>&1 | FileCheck %s --check-prefix=STALE-RVV-BODY --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import sys; text=sys.stdin.read(); text=text.replace("tcrv_scalar.i32_vmul_microkernel", "tcrv_scalar.i32_vadd_microkernel", 1); sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-c 2>&1 | FileCheck %s --check-prefix=STALE-SCALAR-BODY --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."

#map = affine_map<(d0) -> (d0)>

module @rvv_scalar_i32_vmul_dispatch_generic_route {
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

  tcrv.exec.target @frontend_vmul_dispatch_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vmul.dispatch",
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

  func.func @source_frontend_dispatch_vmul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_dispatch_i32_vmul",
        tcrv_frontend_target = @frontend_vmul_dispatch_profile
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

// IR-LABEL: tcrv.exec.kernel @frontend_dispatch_i32_vmul
// IR-SAME: target = @frontend_vmul_dispatch_profile
// IR-SAME: tcrv_frontend_lowering = "i32-vmul"
// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: origin = "rvv-plugin"
// IR-NOT: tcrv_rvv.lowering_descriptor
// IR: tcrv.exec.variant @scalar_fallback_first_slice
// IR-SAME: origin = "scalar-plugin"
// IR-NOT: tcrv_scalar.lowering_descriptor
// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.i32_vmul_microkernel
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32_vmul"
// IR: tcrv_rvv.i32_mul
// IR: tcrv_scalar.i32_vmul_microkernel
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32_vmul"
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "rvv-explicit-i32-vmul-microkernel-c-source"
// IR-SAME: lowering_pipeline = "tcrv-export-rvv-i32-vmul-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch case"
// IR-SAME: runtime_abi = "rvv-i32-vmul-runtime-callable-c-abi.v1"
// IR-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// IR-SAME: runtime_abi_name = "rvv-i32-vmul-runtime-callable-c-function.v1"
// IR-SAME: runtime_glue_role = "runtime-callable-i32-vmul-function"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "scalar-explicit-i32-vmul-microkernel-c-source"
// IR-SAME: lowering_pipeline = "tcrv-export-scalar-i32-vmul-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: runtime_abi = "scalar-i32-vmul-runtime-callable-c-abi.v1"
// IR-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
// IR-SAME: runtime_abi_name = "scalar-i32-vmul-runtime-callable-c-function.v1"
// IR-SAME: runtime_glue_role = "runtime-callable-i32-vmul-fallback-function"
// IR-SAME: status = "supported"
// IR-SAME: target = @scalar_fallback_first_slice

// SOURCE: /* Scope: one selected RVV i32-vmul dispatch case plus one scalar i32-vmul dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_dispatch_i32_vmul */
// SOURCE: /* rvv_artifact_route_id: tcrv-export-rvv-i32-vmul-microkernel-c */
// SOURCE: /* rvv_runtime_abi: rvv-i32-vmul-runtime-callable-c-abi.v1 */
// SOURCE: /* rvv_runtime_abi_name: rvv-i32-vmul-runtime-callable-c-function.v1 */
// SOURCE: /* rvv_runtime_glue_role: runtime-callable-i32-vmul-function */
// SOURCE: /* scalar_artifact_route_id: tcrv-export-scalar-i32-vmul-microkernel-c */
// SOURCE: /* scalar_runtime_abi: scalar-i32-vmul-runtime-callable-c-abi.v1 */
// SOURCE: /* scalar_runtime_abi_name: scalar-i32-vmul-runtime-callable-c-function.v1 */
// SOURCE: /* scalar_runtime_glue_role: runtime-callable-i32-vmul-fallback-function */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vmul_microkernel_frontend_dispatch_i32_vmul_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vmul_microkernel_frontend_dispatch_i32_vmul_scalar_fallback_first_slice */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vmul_microkernel_frontend_dispatch_i32_vmul_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vmul_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vmul_microkernel_frontend_dispatch_i32_vmul_rvv_first_slice
// SOURCE: void tcrv_scalar_i32_vmul_microkernel_frontend_dispatch_i32_vmul_scalar_fallback_first_slice
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i32_vmul_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_mul
// SOURCE: tcrv_scalar_i32_mul
// SOURCE: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// SOURCE: // tcrv_emitc.dispatch_guard_value=rvv_available
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i32_vmul_frontend_dispatch_i32_vmul
// SOURCE: bool [[SOURCE_GUARD:v[0-9]+]] = {{v[0-9]+}} != 0;
// SOURCE: if ([[SOURCE_GUARD]])
// SOURCE: tcrv_rvv_i32_vmul_microkernel_frontend_dispatch_i32_vmul_rvv_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// SOURCE: return;
// SOURCE: tcrv_scalar_i32_vmul_microkernel_frontend_dispatch_i32_vmul_scalar_fallback_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VMUL_DISPATCH_FRONTEND_DISPATCH_I32_VMUL_H
// HEADER: #define TIANCHENRV_RVV_SCALAR_I32_VMUL_DISPATCH_FRONTEND_DISPATCH_I32_VMUL_H
// HEADER: extern "C" {
// HEADER: void tcrv_dispatch_i32_vmul_frontend_dispatch_i32_vmul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VMUL_DISPATCH_FRONTEND_DISPATCH_I32_VMUL_H */

// HARNESS: /* Scope: one selected RVV i32-vmul dispatch case plus one scalar i32-vmul dispatch fallback. */
// HARNESS: __riscv_vmul_vv_i32m1
// HARNESS: // tcrv_emitc.source_op=tcrv_scalar.i32_vmul_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_mul
// HARNESS: tcrv_scalar_i32_mul
// HARNESS: /* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
// HARNESS: if (out[index] != lhs[index] * rhs[index])
// HARNESS: tcrv_dispatch_i32_vmul_frontend_dispatch_i32_vmul_self_check_one(7, 0)
// HARNESS: tcrv_dispatch_i32_vmul_frontend_dispatch_i32_vmul_self_check_one(16, 1)
// HARNESS: tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv

// ROUTE-MISMATCH: direct source export route expected i32-vadd dispatch artifacts, got i32-vmul

// STALE-RVV-BODY: selected RVV dispatch case component body authority failed before RVV+scalar dispatch artifact emission
// STALE-RVV-BODY-SAME: supported emission-plan route 'tcrv-export-rvv-i32-vmul-microkernel-c'
// STALE-RVV-BODY-SAME: does not match RVV binary microkernel descriptor 'tcrv-export-rvv-microkernel-c'

// STALE-SCALAR-BODY: selected scalar dispatch fallback component body authority failed before RVV+scalar dispatch artifact emission
// STALE-SCALAR-BODY-SAME: supported emission-plan route 'tcrv-export-scalar-i32-vmul-microkernel-c'
// STALE-SCALAR-BODY-SAME: does not match scalar microkernel route 'tcrv-export-scalar-microkernel-c'
