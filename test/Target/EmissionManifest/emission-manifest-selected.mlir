// RUN: tcrv-translate --tcrv-export-emission-manifest %s | FileCheck %s

module @manifest_inputs {
  tcrv.exec.kernel @z_scalar_direct {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.diagnostic {
      fallback_role = "conservative",
      message = "fallback-only variant selected by generic cost and capability planning",
      origin = "scalar-plugin",
      preference_available = true,
      preference_explanation = "portable scalar fallback first slice; coverage-oriented metadata route, not a performance claim",
      preference_policy = "prefer only as conservative fallback when better plugin-owned variants are unavailable or not selected",
      preference_rank = 0 : i64,
      preference_score = 1000.0 : f64,
      preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv_scalar.lowering_boundary {
      fallback_reason = "scalar fallback selected boundary is plugin-owned metadata only",
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "z_scalar_direct",
      status = "metadata-only"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "metadata-diagnostic",
      emission_kind = "portable-scalar-fallback-metadata-route",
      lowering_boundary = "tcrv_scalar.lowering_boundary",
      lowering_pipeline = "none-executable-metadata-only",
      message = "scalar fallback first slice records a portable fallback metadata route for compiler decisions only; it does not emit objects, link a runtime, run hardware, prove correctness, or measure performance",
      origin = "scalar-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      runtime_abi = "none-metadata-only",
      runtime_abi_kind = "host-scalar-fallback-metadata",
      runtime_abi_name = "portable-scalar-fallback-metadata-abi.v1",
      runtime_glue_role = "metadata-only-host-fallback-boundary",
      severity = "note",
      status = "metadata-only",
      target = @scalar_fallback_first_slice
    }
  }

  tcrv.exec.kernel @a_rvv_direct {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    tcrv.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      origin = "rvv-plugin",
      preference_available = true,
      preference_explanation = "RVV metadata-only first slice; no runtime performance claim",
      preference_policy = "plugin-local RVV capability participation",
      preference_rank = 0 : i64,
      preference_score = 1.0 : f64,
      preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      capability_summary = "rvv",
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "a_rvv_direct",
      status = "unsupported",
      unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "rvv-unsupported-metadata-boundary",
      lowering_boundary = "tcrv_rvv.lowering_boundary",
      lowering_pipeline = "rvv-none-executable-unsupported",
      message = "RVV metadata-only first slice has no RVV lowering pipeline, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-none-executable-unsupported",
      runtime_abi_kind = "rvv-plugin-deferred-runtime-abi",
      runtime_abi_name = "rvv-executable-runtime-abi-deferred",
      runtime_glue_role = "deferred-rvv-runtime-glue",
      severity = "error",
      status = "unsupported",
      target = @rvv_first_slice
    }
  }

  tcrv.exec.kernel @m_dispatch {
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
    tcrv.exec.variant @rvv_case attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_case {
        condition = "capability_available",
        guard = "plugin_local_rvv_first_slice",
        origin = "rvv-plugin",
        policy = "metadata_only_first_slice",
        preference_available = true,
        preference_explanation = "RVV metadata-only first slice; no runtime performance claim",
        preference_policy = "plugin-local RVV capability participation",
        preference_rank = 0 : i64,
        preference_score = 1.0 : f64,
        preference_tie_break = "best explicit plugin preference score; equal scores use fallback role, original IR order, then symbol name"
      }
      tcrv.exec.fallback @scalar_fallback_first_slice {
        fallback_role = "conservative",
        origin = "scalar-plugin",
        preference_available = true,
        preference_explanation = "portable scalar fallback first slice; coverage-oriented metadata route, not a performance claim",
        preference_policy = "prefer only as conservative fallback when better plugin-owned variants are unavailable or not selected",
        preference_rank = 1 : i64,
        preference_score = 1000.0 : f64,
        preference_tie_break = "ranked by explicit plugin preference score; fallback role, original IR order, then symbol name remain the stable tie-breaks"
      }
    }
    tcrv_rvv.lowering_boundary {
      capability_summary = "rvv",
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "dispatch case",
      selected_variant = @rvv_case,
      source_kernel = "m_dispatch",
      status = "unsupported",
      unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
    }
    tcrv_scalar.lowering_boundary {
      fallback_reason = "scalar fallback selected boundary is plugin-owned metadata only",
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "m_dispatch",
      status = "metadata-only"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "rvv-unsupported-metadata-boundary",
      lowering_boundary = "tcrv_rvv.lowering_boundary",
      lowering_pipeline = "rvv-none-executable-unsupported",
      message = "RVV metadata-only first slice has no RVV lowering pipeline, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "dispatch case",
      runtime_abi = "rvv-none-executable-unsupported",
      runtime_abi_kind = "rvv-plugin-deferred-runtime-abi",
      runtime_abi_name = "rvv-executable-runtime-abi-deferred",
      runtime_glue_role = "deferred-rvv-runtime-glue",
      severity = "error",
      status = "unsupported",
      target = @rvv_case
    }
    tcrv.exec.diagnostic {
      artifact_kind = "metadata-diagnostic",
      emission_kind = "portable-scalar-fallback-metadata-route",
      lowering_boundary = "tcrv_scalar.lowering_boundary",
      lowering_pipeline = "none-executable-metadata-only",
      message = "scalar fallback first slice records a portable fallback metadata route for compiler decisions only; it does not emit objects, link a runtime, run hardware, prove correctness, or measure performance",
      origin = "scalar-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      runtime_abi = "none-metadata-only",
      runtime_abi_kind = "host-scalar-fallback-metadata",
      runtime_abi_name = "portable-scalar-fallback-metadata-abi.v1",
      runtime_glue_role = "metadata-only-host-fallback-boundary",
      severity = "note",
      status = "metadata-only",
      target = @scalar_fallback_first_slice
    }
  }
}

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK: module: "manifest_inputs"
// CHECK: kernel_count: 3

// CHECK-LABEL: kernel @a_rvv_direct
// CHECK: selected_surface: selected-marker
// CHECK: selection_kind: "static-variant"
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "direct variant"
// CHECK: origin: "rvv-plugin"
// CHECK: emission_status: "unsupported"
// CHECK: emission_kind: "rvv-unsupported-metadata-boundary"
// CHECK: lowering_boundary: "tcrv_rvv.lowering_boundary"
// CHECK: runtime_abi_kind: "rvv-plugin-deferred-runtime-abi"
// CHECK: runtime_abi_name: "rvv-executable-runtime-abi-deferred"
// CHECK: runtime_glue_role: "deferred-rvv-runtime-glue"
// CHECK: required_capabilities: [@rvv]
// CHECK: explanation: "RVV metadata-only first slice has no RVV lowering pipeline
// CHECK: preference:
// CHECK: policy: "plugin-local RVV capability participation"

// CHECK-LABEL: kernel @m_dispatch
// CHECK: selected_surface: dispatch
// CHECK: dispatch_case[0]: @rvv_case
// CHECK: dispatch_fallback: @scalar_fallback_first_slice
// CHECK: path[0]:
// CHECK: selected_variant: @rvv_case
// CHECK: role: "dispatch case"
// CHECK: origin: "rvv-plugin"
// CHECK: emission_status: "unsupported"
// CHECK: required_capabilities: [@rvv]
// CHECK: path[1]:
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "dispatch fallback"
// CHECK: origin: "scalar-plugin"
// CHECK: emission_status: "metadata-only"
// CHECK: emission_kind: "portable-scalar-fallback-metadata-route"
// CHECK: lowering_boundary: "tcrv_scalar.lowering_boundary"
// CHECK: runtime_abi: "none-metadata-only"
// CHECK: runtime_abi_kind: "host-scalar-fallback-metadata"
// CHECK: runtime_abi_name: "portable-scalar-fallback-metadata-abi.v1"
// CHECK: runtime_glue_role: "metadata-only-host-fallback-boundary"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: fallback_role: "conservative"

// CHECK-LABEL: kernel @z_scalar_direct
// CHECK: selected_surface: selected-marker
// CHECK: selection_kind: "fallback-only"
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "direct variant"
// CHECK: origin: "scalar-plugin"
// CHECK: emission_status: "metadata-only"
// CHECK: emission_kind: "portable-scalar-fallback-metadata-route"
// CHECK: lowering_boundary: "tcrv_scalar.lowering_boundary"
// CHECK: runtime_abi_kind: "host-scalar-fallback-metadata"
// CHECK: runtime_abi_name: "portable-scalar-fallback-metadata-abi.v1"
// CHECK: runtime_glue_role: "metadata-only-host-fallback-boundary"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: preference:
// CHECK: available: true
// CHECK: fallback_role: "conservative"
