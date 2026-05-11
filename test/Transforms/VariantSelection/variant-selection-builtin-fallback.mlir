// RUN: tcrv-opt %s --split-input-file --tcrv-select-variants --tcrv-check-capability-requires --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_plus_scalar
  tcrv.exec.kernel @public_rvv_plus_scalar {
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
    // CHECK: tcrv.exec.variant @rvv_first_slice
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: fallback_role = "conservative"
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK-NOT: tcrv.exec.dispatch
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: preference_available = true
    // CHECK-SAME: preference_rank = 0
    // CHECK-SAME: preference_score
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK-NOT: runtime_guard
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-NOT: tcrv_scalar.lowering_boundary
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_conflicting_rvv_plus_scalar
  tcrv.exec.kernel @public_conflicting_rvv_plus_scalar {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      conflicts = ["build.policy.no_rvv"],
      status = "available"
    }
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
    // CHECK: tcrv.exec.variant @rvv_first_slice
    // CHECK-SAME: requires = [@rvv]
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: fallback_role = "conservative"
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK: tcrv.exec.dispatch
    // CHECK: tcrv.exec.case @rvv_first_slice
    // CHECK-SAME: condition = "capability_available"
    // CHECK-SAME: guard = "plugin_local_rvv_first_slice"
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: policy = "metadata_only_first_slice"
    // CHECK-SAME: runtime_guard_required = true
    // CHECK: tcrv.exec.fallback @scalar_fallback_first_slice
    // CHECK-SAME: fallback_role = "conservative"
    // CHECK-SAME: origin = "scalar-plugin"
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_scalar_only
  tcrv.exec.kernel @public_scalar_only {
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
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: fallback_role = "conservative"
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: preference_available = true
    // CHECK-SAME: preference_rank = 0
    // CHECK-SAME: preference_score
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "fallback-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
    // CHECK: tcrv_scalar.lowering_boundary
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
    // CHECK-SAME: runtime_abi_name = "scalar-i32-vadd-runtime-callable-c-function.v1"
    // CHECK-SAME: runtime_glue_role = "runtime-callable-i32-vadd-fallback-function"
    // CHECK-SAME: selected_plan_metadata = [{{.*}}name = "tcrv_scalar.emitc_source_op"{{.*}}value = "tcrv_scalar.i32_vadd_microkernel"{{.*}}name = "tcrv_scalar.emitc_lowerable_op_interface"{{.*}}value = "TCRVEmitCLowerableOpInterface"
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_without_scalar_fallback
  tcrv.exec.kernel @public_rvv_without_scalar_fallback {
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
    // CHECK-NOT: tcrv.exec.dispatch
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: preference_available = true
    // CHECK-SAME: preference_rank = 0
    // CHECK-SAME: preference_score
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: message = "no plugin-provided conservative fallback candidate is available; tcrv.exec.dispatch fallback is not invented"
    // CHECK-SAME: reason = "fallback-coverage-missing"
    // CHECK-SAME: selection_kind = "missing-conservative-fallback"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}
