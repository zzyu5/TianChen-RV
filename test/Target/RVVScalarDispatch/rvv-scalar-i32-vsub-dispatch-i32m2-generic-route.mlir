// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=linalg.generic --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-header | FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not="out[index]" --implicit-check-not=i32_vadd --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c | FileCheck %s --check-prefix=HARNESS --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.selected_vector_shape = "i32m2"/s//tcrv_rvv.selected_vector_shape = "i32m1"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=SHAPE-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text = sys.stdin.read(); pattern = r"\{name = \"tcrv_rvv\.selected_vector_suffix\"([^}]*)value = \"i32m2\"\}"; text, count = re.subn(pattern, lambda m: "{name = \"tcrv_rvv.selected_vector_suffix\"" + m.group(1) + "value = \"i32m1\"}", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=PLAN-METADATA-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text = sys.stdin.read(); pattern = r"\{name = \"tcrv_rvv\.selected_binary_family\"([^}]*)value = \"i32-vsub\"\}"; text, count = re.subn(pattern, lambda m: "{name = \"tcrv_rvv.selected_binary_family\"" + m.group(1) + "value = \"i64-vsub\"}", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=SELECTED-CONFIG-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text = sys.stdin.read(); pattern = r"\{name = \"tcrv_rvv\.runtime_element_count_c_name\"([^}]*)value = \"n\"\}"; text, count = re.subn(pattern, lambda m: "{name = \"tcrv_rvv.runtime_element_count_c_name\"" + m.group(1) + "value = \"16\"}", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=RUNTIME-CONTROL-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."

#map = affine_map<(d0) -> (d0)>

module @rvv_scalar_i32_vsub_i32m2_dispatch_generic_route {
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

  tcrv.exec.target @frontend_vsub_i32m2_dispatch_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vsub.i32m2.dispatch",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m2",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_frontend_dispatch_i32m2_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_dispatch_i32m2_vsub",
        tcrv_frontend_target = @frontend_vsub_i32m2_dispatch_profile
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

// IR-LABEL: tcrv.exec.kernel @frontend_dispatch_i32m2_vsub
// IR-SAME: target = @frontend_vsub_i32m2_dispatch_profile
// IR-SAME: tcrv_frontend_lowering = "i32-vsub"
// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: origin = "rvv-plugin"
// IR-NOT: tcrv_rvv.lowering_descriptor
// IR-SAME: tcrv_rvv.selected_vector_shape = "i32m2"
// IR-SAME: tcrv_rvv.selected_vector_type = "vint32m2_t"
// IR: tcrv.exec.variant @scalar_fallback_first_slice
// IR-SAME: origin = "scalar-plugin"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.i32_vsub_microkernel
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32m2_vsub"
// IR: tcrv_rvv.i32_sub
// IR: tcrv_scalar.i32_vsub_microkernel
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR-SAME: source_kernel = "frontend_dispatch_i32m2_vsub"

// SOURCE: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_dispatch_i32m2_vsub */
// SOURCE: /* selected_binary_config: dtype=i32, family=i32-vsub, operator=subtract, shape=i32m2, sew=32, lmul=m2
// SOURCE: /* rvv_selected_plan_metadata[0]: name=tcrv_rvv.selected_vector_shape, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[6]: name=tcrv_rvv.selected_vector_suffix, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[8]: name=tcrv_rvv.selected_vector_sew_capability, value=rvv.i32_m2.sew32, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[9]: name=tcrv_rvv.selected_vector_lmul_capability, value=rvv.i32_m2.lmul_m2, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[10]: name=tcrv_rvv.selected_tail_policy_capability, value=rvv.i32_m2.tail_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[11]: name=tcrv_rvv.selected_mask_policy_capability, value=rvv.i32_m2.mask_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[12]: name=tcrv_rvv.vlenb_bytes, value=16, role=rvv-base-capacity-fact
// SOURCE: /* rvv_selected_plan_metadata[13]: name=tcrv_rvv.base_i32_m1_lanes, value=4, role=rvv-base-capacity-fact
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32m2_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32m2_vsub_scalar_fallback_first_slice */
// SOURCE: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
// SOURCE: /* control_plane_config: sew=32, lmul=m2, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// SOURCE: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: __riscv_vsetvl_e32m2
// SOURCE: __riscv_vle32_v_i32m2
// SOURCE: __riscv_vsub_vv_i32m2
// SOURCE: __riscv_vse32_v_i32m2
// SOURCE: int32_t difference = tcrv_scalar_i32_sub(lhs[index], rhs[index]);
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i32_vsub_frontend_dispatch_i32m2_vsub
// SOURCE: if (rvv_available)
// SOURCE: tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32m2_vsub_rvv_first_slice(lhs, rhs, out, n);
// SOURCE: return;
// SOURCE: tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32m2_vsub_scalar_fallback_first_slice(lhs, rhs, out, n);

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_DISPATCH_I32M2_VSUB_H
// HEADER: void tcrv_dispatch_i32_vsub_frontend_dispatch_i32m2_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_DISPATCH_I32M2_VSUB_H */

// HARNESS: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// HARNESS: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2
// HARNESS: __riscv_vsub_vv_i32m2
// HARNESS: int32_t difference = tcrv_scalar_i32_sub(lhs[index], rhs[index]);
// HARNESS: /* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
// HARNESS: if (out[index] != lhs[index] - rhs[index])
// HARNESS: tcrv_dispatch_i32_vsub_frontend_dispatch_i32m2_vsub_self_check_one(7, 0)
// HARNESS: tcrv_dispatch_i32_vsub_frontend_dispatch_i32m2_vsub_self_check_one(16, 1)
// HARNESS: tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv

// SHAPE-MISMATCH: selected vector-shape shape must be 'i32m2'
// PLAN-METADATA-MISMATCH: selected_plan_metadata 'tcrv_rvv.selected_vector_suffix' vector suffix must be 'i32m2'
// SELECTED-CONFIG-MISMATCH: route id 'tcrv-export-rvv-i32-vsub-microkernel-c' selected_plan_metadata 'tcrv_rvv.selected_binary_family' must use value 'i32-vsub'
// RUNTIME-CONTROL-MISMATCH: selected_plan_metadata 'tcrv_rvv.runtime_element_count_c_name' runtime element-count C name must be 'n'
