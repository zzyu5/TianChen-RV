// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  // CHECK-LABEL: tcrv.exec.target @module_rvv_profile
  // CHECK-SAME: id = "rvv.profile.module"
  // CHECK-SAME: kind = "profile"
  // CHECK-SAME: provides = ["rvv", "rvv.hart_count", "rvv.probe.compile_run"]
  tcrv.exec.target @module_rvv_profile {
    id = "rvv.profile.module",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.probe.compile_run"],
    architecture = "riscv64",
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
    // CHECK-SAME: tcrv_rvv.element_count = 16 : i64
    // CHECK-SAME: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
    // CHECK-SAME: tcrv_rvv.required_march = "rv64gcv"

    // CHECK: tcrv.exec.diagnostic
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
    // CHECK-SAME: capability_summary = "rvv.profile.module"
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@module_rvv_profile]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "module_profile_pipeline"
    // CHECK-SAME: status = "unsupported"
    // CHECK: tcrv_rvv.i32_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@module_rvv_profile]
    // CHECK-SAME: required_march = "rv64gcv"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "module_profile_pipeline"
    // CHECK: tcrv_rvv.i32_vadd_dataflow

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
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}
