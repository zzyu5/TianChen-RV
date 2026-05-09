// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not="out[index]" --implicit-check-not=i32_vadd --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | sed -e 's/tcrv-export-scalar-i32-vsub-microkernel-c/tcrv-export-scalar-microkernel-c/' -e 's/scalar-explicit-i32-vsub-microkernel-c-source/scalar-explicit-i32-vadd-microkernel-c-source/' -e 's/scalar-i32-vsub-runtime-callable-c-abi.v1/scalar-i32-vadd-runtime-callable-c-abi.v1/' -e 's/scalar-i32-vsub-runtime-callable-c-function.v1/scalar-i32-vadd-runtime-callable-c-function.v1/' -e 's/runtime-callable-i32-vsub-fallback-function/runtime-callable-i32-vadd-fallback-function/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export." --implicit-check-not="void tcrv_dispatch_i32_vsub"

#map = affine_map<(d0) -> (d0)>

module @rvv_scalar_i32_vsub_dispatch_generic_route {
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

  tcrv.exec.target @frontend_vsub_dispatch_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vsub.dispatch",
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

  func.func @source_frontend_dispatch_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_dispatch_i32_vsub",
        tcrv_frontend_target = @frontend_vsub_dispatch_profile
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

// IR-LABEL: tcrv.exec.kernel @frontend_dispatch_i32_vsub
// IR-SAME: target = @frontend_vsub_dispatch_profile
// IR-SAME: tcrv_frontend_lowering = "i32-vsub"
// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: origin = "rvv-plugin"
// IR-SAME: tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1"
// IR: tcrv.exec.variant @scalar_fallback_first_slice
// IR-SAME: origin = "scalar-plugin"
// IR-SAME: tcrv_scalar.lowering_descriptor = "i32-vsub-microkernel.v1"
// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.i32_vsub_microkernel
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32_vsub"
// IR: tcrv_rvv.i32_sub
// IR: tcrv_scalar.i32_vsub_microkernel
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32_vsub"
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "rvv-explicit-i32-vsub-microkernel-c-source"
// IR-SAME: lowering_pipeline = "tcrv-export-rvv-i32-vsub-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch case"
// IR-SAME: runtime_abi = "rvv-i32-vsub-runtime-callable-c-abi.v1"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "scalar-explicit-i32-vsub-microkernel-c-source"
// IR-SAME: lowering_pipeline = "tcrv-export-scalar-i32-vsub-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: runtime_abi = "scalar-i32-vsub-runtime-callable-c-abi.v1"
// IR-SAME: status = "supported"
// IR-SAME: target = @scalar_fallback_first_slice

// SOURCE: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_dispatch_i32_vsub */
// SOURCE: /* rvv_artifact_route_id: tcrv-export-rvv-i32-vsub-microkernel-c */
// SOURCE: /* rvv_runtime_abi: rvv-i32-vsub-runtime-callable-c-abi.v1 */
// SOURCE: /* rvv_runtime_abi_name: rvv-i32-vsub-runtime-callable-c-function.v1 */
// SOURCE: /* rvv_runtime_glue_role: runtime-callable-i32-vsub-function */
// SOURCE: /* scalar_artifact_route_id: tcrv-export-scalar-i32-vsub-microkernel-c */
// SOURCE: /* scalar_runtime_abi: scalar-i32-vsub-runtime-callable-c-abi.v1 */
// SOURCE: /* scalar_runtime_abi_name: scalar-i32-vsub-runtime-callable-c-function.v1 */
// SOURCE: /* scalar_runtime_glue_role: runtime-callable-i32-vsub-fallback-function */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice */
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice
// SOURCE: __riscv_vsub_vv_i32m1
// SOURCE: void tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice
// SOURCE: out[index] = lhs[index] - rhs[index];
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i32_vsub_frontend_dispatch_i32_vsub
// SOURCE: if (rvv_available)
// SOURCE: tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice(lhs, rhs, out, n);
// SOURCE: return;
// SOURCE: tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice(lhs, rhs, out, n);

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_DISPATCH_I32_VSUB_H
// HEADER: #define TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_DISPATCH_I32_VSUB_H
// HEADER: extern "C" {
// HEADER: void tcrv_dispatch_i32_vsub_frontend_dispatch_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_DISPATCH_I32_VSUB_H */

// MISMATCH: selected RVV dispatch case callable family 'i32-vsub' does not match selected scalar dispatch fallback callable family 'i32-vadd'
