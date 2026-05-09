// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=EXPORT --implicit-check-not="int main(void)" --implicit-check-not=_self_check --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

// EXPORT: /* TianChen-RV RVV runtime-callable microkernel C export. */
// EXPORT: #include <riscv_vector.h>
// EXPORT: /* selected_kernel: @module_profile_pipeline */
// EXPORT: /* selected_variant: @rvv_first_slice */
// EXPORT: /* selected_march: rv64gcv */
// EXPORT: /* required_capabilities: @module_rvv_profile */
// EXPORT: void tcrv_rvv_i32_vadd_microkernel_module_profile_pipeline_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)
// EXPORT: __riscv_vsetvl_e32m1
// EXPORT: __riscv_vadd_vv_i32m1

module {
  // CHECK-LABEL: tcrv.exec.target @module_rvv_profile
  // CHECK-SAME: id = "rvv.profile.module"
  // CHECK-SAME: kind = "profile"
  // CHECK-SAME: provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run"]
  tcrv.exec.target @module_rvv_profile {
    id = "rvv.profile.module",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run"],
    architecture = "riscv64",
    bytes = 32 : i64,
    lanes = 8 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    isa_vector_hints = "rv64gcv_zvl128b",
    count = 64 : i64,
    selected_march = "rv64gcv",
    status = "available"
  }

  // CHECK-LABEL: tcrv.exec.kernel @module_profile_pipeline
  // CHECK-SAME: target = @module_rvv_profile
  tcrv.exec.kernel @module_profile_pipeline attributes {target = @module_rvv_profile} {
    // CHECK: tcrv.exec.variant @rvv_first_slice
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: requires = [@module_rvv_profile]
    // CHECK-SAME: tcrv_rvv.base_i32_m1_lanes = 8 : i64
    // CHECK-SAME: tcrv_rvv.element_count = 32 : i64
    // CHECK-SAME: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
    // CHECK-SAME: tcrv_rvv.required_march = "rv64gcv"
    // CHECK-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
    // CHECK-SAME: tcrv_rvv.vlenb_bytes = 32 : i64

    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: preference_explanation = "RVV metadata-only first slice; capability-derived base_i32_m1_lanes=8 is a plugin-local selection heuristic input, not a runtime performance claim"
    // CHECK-SAME: preference_score = 1.250000e-01 : f64
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "fallback-coverage-missing"
    // CHECK-SAME: selection_kind = "missing-conservative-fallback"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: target = @rvv_first_slice

    // CHECK: tcrv.exec.mem_window @abi_lhs_input_buffer
    // CHECK-SAME: abi_role = "lhs-input-buffer"
    // CHECK: tcrv.exec.mem_window @abi_rhs_input_buffer
    // CHECK-SAME: abi_role = "rhs-input-buffer"
    // CHECK: tcrv.exec.mem_window @abi_output_buffer
    // CHECK-SAME: abi_role = "output-buffer"
    // CHECK: tcrv.exec.runtime_param @abi_runtime_element_count
    // CHECK-SAME: abi_role = "runtime-element-count"

    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: base_i32_m1_lanes = 8 : i64
    // CHECK-SAME: capability_summary = "rvv.profile.module"
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@module_rvv_profile]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: selected_vector_shape = "i32m1"
    // CHECK-SAME: source_kernel = "module_profile_pipeline"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: vlenb_bytes = 32 : i64
    // CHECK: tcrv_rvv.i32_vadd_microkernel
    // CHECK-SAME: element_count = 32 : i64
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@module_rvv_profile]
    // CHECK-SAME: required_march = "rv64gcv"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: selected_vector_shape = "i32m1"
    // CHECK-SAME: source_kernel = "module_profile_pipeline"
    // CHECK: tcrv_rvv.i32_load
    // CHECK: tcrv_rvv.i32_add
    // CHECK: tcrv_rvv.i32_store

    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "runtime-callable-c-source"
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@module_rvv_profile]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
    // CHECK-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"
    // CHECK-SAME: runtime_glue_role = "runtime-callable-i32-vadd-function"
    // CHECK-SAME: selected_plan_metadata =
    // CHECK-SAME: name = "tcrv_rvv.selected_vector_shape"
    // CHECK-SAME: role = "selected-rvv-vector-shape-config"
    // CHECK-SAME: value = "i32m1"
    // CHECK-SAME: name = "tcrv_rvv.selected_vector_sew"
    // CHECK-SAME: value = "32"
    // CHECK-SAME: name = "tcrv_rvv.selected_vector_lmul"
    // CHECK-SAME: value = "m1"
    // CHECK-SAME: name = "tcrv_rvv.vlenb_bytes"
    // CHECK-SAME: note = "base i32 M1 capacity fact from target/profile evidence; not selected vector shape, runtime input, VL/AVL, or performance evidence"
    // CHECK-SAME: role = "rvv-base-capacity-fact"
    // CHECK-SAME: value = "32"
    // CHECK-SAME: name = "tcrv_rvv.base_i32_m1_lanes"
    // CHECK-SAME: note = "base i32 M1 capacity fact from target/profile evidence; not selected vector shape, runtime input, VL/AVL, or performance evidence"
    // CHECK-SAME: role = "rvv-base-capacity-fact"
    // CHECK-SAME: value = "8"
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}
