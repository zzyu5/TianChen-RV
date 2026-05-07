// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | tcrv-opt --split-input-file | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: tcrv-opt %s --split-input-file --tcrv-execution-planning-pipeline | not tcrv-opt --split-input-file --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=SECOND

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_rvv_plus_scalar
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_rvv_plus_scalar
  tcrv.exec.kernel @pipeline_rvv_plus_scalar {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE-SAME: condition = "rvv_capability_properties_available"
    // PIPE-SAME: guard = "plugin_local_rvv_property_evidence"
    // PIPE-SAME: origin = "rvv-plugin"
    // PIPE-SAME: policy = "metadata_only_first_slice"
    // PIPE-SAME: requires = [@rvv]
    // PIPE-SAME: tcrv_rvv.element_count = 16 : i64
    // PIPE-SAME: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
    // PIPE-SAME: tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: policy = "portable_scalar_fallback_first_slice"
    // PIPE-SAME: requires = [@scalar_fallback]
    // PIPE-SAME: tcrv_scalar.element_count = 16 : i64
    // PIPE-SAME: tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
    // PIPE: tcrv.exec.dispatch
    // PIPE: tcrv.exec.case @rvv_first_slice
    // PIPE-SAME: condition = "rvv_capability_properties_available"
    // PIPE-SAME: guard = "plugin_local_rvv_property_evidence"
    // PIPE-SAME: origin = "rvv-plugin"
    // PIPE-SAME: policy = "metadata_only_first_slice"
    // PIPE-SAME: preference_available = true
    // PIPE-SAME: preference_rank = 0
    // PIPE-SAME: preference_score
    // PIPE: tcrv.exec.fallback @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: preference_available = true
    // PIPE-SAME: preference_rank = 1
    // PIPE-SAME: preference_score
    // PIPE: tcrv_rvv.lowering_boundary
    // PIPE-SAME: capability_summary = "rvv"
    // PIPE-SAME: origin = "rvv-plugin"
    // PIPE-SAME: required_capabilities = [@rvv]
    // PIPE-SAME: role = "dispatch case"
    // PIPE-SAME: selected_variant = @rvv_first_slice
    // PIPE-SAME: source_kernel = "pipeline_rvv_plus_scalar"
    // PIPE-SAME: status = "unsupported"
    // PIPE: tcrv_rvv.i32_vadd_microkernel
    // PIPE-SAME: element_count = 16 : i64
    // PIPE-SAME: origin = "rvv-plugin"
    // PIPE-SAME: required_capabilities = [@rvv]
    // PIPE-SAME: required_march = "rv64gcv"
    // PIPE-SAME: role = "dispatch case"
    // PIPE-SAME: selected_variant = @rvv_first_slice
    // PIPE-SAME: source_kernel = "pipeline_rvv_plus_scalar"
    // PIPE: tcrv_rvv.i32_vadd_dataflow
    // PIPE-SAME: lhs_role = "lhs-input-buffer"
    // PIPE-SAME: out_role = "output-buffer"
    // PIPE-SAME: rhs_role = "rhs-input-buffer"
    // PIPE-SAME: runtime_n_role = "runtime-element-count"
    // PIPE: tcrv_scalar.lowering_boundary
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "dispatch fallback"
    // PIPE-SAME: selected_variant = @scalar_fallback_first_slice
    // PIPE-SAME: source_kernel = "pipeline_rvv_plus_scalar"
    // PIPE-SAME: status = "metadata-only"
    // PIPE: tcrv_scalar.i32_vadd_microkernel
    // PIPE-SAME: element_count = 16 : i64
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "dispatch fallback"
    // PIPE-SAME: selected_variant = @scalar_fallback_first_slice
    // PIPE-SAME: source_kernel = "pipeline_rvv_plus_scalar"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@rvv]
    // PIPE-SAME: role = "dispatch case"
    // PIPE-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
    // PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
    // PIPE-SAME: runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"
    // PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vadd-function"
    // PIPE-SAME: status = "supported"
    // PIPE-SAME: target = @rvv_first_slice
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: artifact_kind = "runtime-callable-c-source"
    // PIPE-SAME: emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source"
    // PIPE-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "dispatch fallback"
    // PIPE-SAME: runtime_abi = "scalar-i32-vadd-runtime-callable-c-abi.v1"
    // PIPE-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
    // PIPE-SAME: runtime_abi_name = "scalar-i32-vadd-runtime-callable-c-function.v1"
    // PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vadd-fallback-function"
    // PIPE-SAME: status = "supported"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // ROUNDTRIP: tcrv.exec.dispatch
    // ROUNDTRIP: tcrv_rvv.lowering_boundary
    // ROUNDTRIP-SAME: selected_variant = @rvv_first_slice
    // ROUNDTRIP: tcrv_rvv.i32_vadd_microkernel
    // ROUNDTRIP-SAME: selected_variant = @rvv_first_slice
    // ROUNDTRIP: tcrv_rvv.i32_vadd_dataflow
    // ROUNDTRIP: tcrv_scalar.lowering_boundary
    // ROUNDTRIP-SAME: selected_variant = @scalar_fallback_first_slice
    // ROUNDTRIP: tcrv_scalar.i32_vadd_microkernel
    // ROUNDTRIP-SAME: selected_variant = @scalar_fallback_first_slice
  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_conflicting_rvv_plus_scalar
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_conflicting_rvv_plus_scalar
  tcrv.exec.kernel @pipeline_conflicting_rvv_plus_scalar {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      conflicts = ["build.policy.no_rvv"],
      status = "available"
    }
    tcrv.exec.capability @no_rvv_policy {
      id = "generic.build.profile",
      kind = "build-policy",
      provides = ["build.policy.no_rvv"],
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE-SAME: requires = [@rvv]
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE: tcrv.exec.dispatch
    // PIPE: tcrv.exec.case @rvv_first_slice
    // PIPE-SAME: guard = "plugin_local_rvv_property_evidence"
    // PIPE-SAME: origin = "rvv-plugin"
    // PIPE-SAME: policy = "metadata_only_first_slice"
    // PIPE: tcrv.exec.fallback @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // ROUNDTRIP: tcrv.exec.dispatch
    // ROUNDTRIP: tcrv.exec.case @rvv_first_slice
    // ROUNDTRIP-SAME: guard = "plugin_local_rvv_property_evidence"
    // ROUNDTRIP: tcrv.exec.fallback @scalar_fallback_first_slice
  }
}

// -----

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_scalar_after_rvv_decline
  // ROUNDTRIP-LABEL: tcrv.exec.kernel @pipeline_scalar_after_rvv_decline
  tcrv.exec.kernel @pipeline_scalar_after_rvv_decline {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE-NOT: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: policy = "portable_scalar_fallback_first_slice"
    // PIPE-SAME: requires = [@scalar_fallback]
    // PIPE-SAME: tcrv_scalar.element_count = 16 : i64
    // PIPE-SAME: tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
    // PIPE-NOT: tcrv_rvv.lowering_boundary
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: fallback_role = "conservative"
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: preference_available = true
    // PIPE-SAME: preference_rank = 0
    // PIPE-SAME: preference_score
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice
    // PIPE: tcrv_scalar.lowering_boundary
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: selected_variant = @scalar_fallback_first_slice
    // PIPE-SAME: source_kernel = "pipeline_scalar_after_rvv_decline"
    // PIPE-SAME: status = "metadata-only"
    // PIPE: tcrv_scalar.i32_vadd_microkernel
    // PIPE-SAME: element_count = 16 : i64
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: selected_variant = @scalar_fallback_first_slice
    // PIPE-SAME: source_kernel = "pipeline_scalar_after_rvv_decline"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: artifact_kind = "runtime-callable-c-source"
    // PIPE-SAME: emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source"
    // PIPE-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: required_capabilities = [@scalar_fallback]
    // PIPE-SAME: role = "direct variant"
    // PIPE-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
    // PIPE-SAME: runtime_abi_name = "scalar-i32-vadd-runtime-callable-c-function.v1"
    // PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vadd-fallback-function"
    // PIPE-SAME: status = "supported"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // ROUNDTRIP-NOT: tcrv.exec.variant @rvv_first_slice
    // ROUNDTRIP: tcrv.exec.variant @scalar_fallback_first_slice
    // ROUNDTRIP: tcrv_scalar.lowering_boundary
    // ROUNDTRIP-SAME: selected_variant = @scalar_fallback_first_slice
    // ROUNDTRIP: tcrv_scalar.i32_vadd_microkernel
    // ROUNDTRIP-SAME: selected_variant = @scalar_fallback_first_slice
    // ROUNDTRIP: tcrv.exec.diagnostic
    // ROUNDTRIP-SAME: target = @scalar_fallback_first_slice
  }
}

// SECOND: TianChen-RV variant selection failed for kernel @pipeline_rvv_plus_scalar
// SECOND-SAME: kernel already contains a direct tcrv.exec.dispatch
