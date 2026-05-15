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
      preference_explanation = "portable scalar fallback first slice; conservative fallback envelope, not an executable route or performance claim",
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
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "scalar-fallback-unsupported-emission",
      lowering_pipeline = "scalar-fallback-no-materialized-emitc-route",
      message = "scalar fallback first slice has no materialized extension-family body, EmitC lowering, runtime ABI, target artifact route, or metadata-only emission route",
      origin = "scalar-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      runtime_abi = "scalar-fallback-no-runtime-abi",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "error",
      status = "unsupported",
      target = @scalar_fallback_first_slice
    }
  }

  tcrv.exec.kernel @a_rvv_direct {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
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
      message = "RVV first slice has no materialized EmitC lowering, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-none-executable-unsupported",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "error",
      status = "unsupported",
      target = @rvv_first_slice
    }
  }

  tcrv.exec.kernel @m_dispatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
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
        preference_explanation = "portable scalar fallback first slice; conservative fallback envelope, not an executable route or performance claim",
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
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "rvv-unsupported-metadata-boundary",
      lowering_boundary = "tcrv_rvv.lowering_boundary",
      lowering_pipeline = "rvv-none-executable-unsupported",
      message = "RVV first slice has no materialized EmitC lowering, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "dispatch case",
      runtime_abi = "rvv-none-executable-unsupported",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "error",
      status = "unsupported",
      target = @rvv_case
    }
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "scalar-fallback-unsupported-emission",
      lowering_pipeline = "scalar-fallback-no-materialized-emitc-route",
      message = "scalar fallback first slice has no materialized extension-family body, EmitC lowering, runtime ABI, target artifact route, or metadata-only emission route",
      origin = "scalar-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      runtime_abi = "scalar-fallback-no-runtime-abi",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "error",
      status = "unsupported",
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
// CHECK: runtime_abi_kind: "unsupported-plugin-runtime-abi"
// CHECK: runtime_abi_name: "unsupported-emission-runtime-abi"
// CHECK: runtime_glue_role: "no-runtime-glue-unsupported"
// CHECK: required_capabilities: [@rvv]
// CHECK: explanation: "RVV first slice has no materialized EmitC lowering
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
// CHECK: emission_status: "unsupported"
// CHECK: emission_kind: "scalar-fallback-unsupported-emission"
// CHECK: runtime_abi: "scalar-fallback-no-runtime-abi"
// CHECK: runtime_abi_kind: "unsupported-plugin-runtime-abi"
// CHECK: runtime_abi_name: "unsupported-emission-runtime-abi"
// CHECK: runtime_glue_role: "no-runtime-glue-unsupported"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: fallback_role: "conservative"

// CHECK-LABEL: kernel @z_scalar_direct
// CHECK: selected_surface: selected-marker
// CHECK: selection_kind: "fallback-only"
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "direct variant"
// CHECK: origin: "scalar-plugin"
// CHECK: emission_status: "unsupported"
// CHECK: emission_kind: "scalar-fallback-unsupported-emission"
// CHECK: runtime_abi_kind: "unsupported-plugin-runtime-abi"
// CHECK: runtime_abi_name: "unsupported-emission-runtime-abi"
// CHECK: runtime_glue_role: "no-runtime-glue-unsupported"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: preference:
// CHECK: available: true
// CHECK: fallback_role: "conservative"
