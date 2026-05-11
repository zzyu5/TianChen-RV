// RUN: tcrv-opt %s --split-input-file --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_plus_scalar_boundary
  tcrv.exec.kernel @public_rvv_plus_scalar_boundary {
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
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: capability_summary = "rvv"
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "public_rvv_plus_scalar_boundary"
    // CHECK-SAME: status = "unsupported"
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
  // CHECK-LABEL: tcrv.exec.kernel @public_scalar_only_scalar_boundary
  tcrv.exec.kernel @public_scalar_only_scalar_boundary {
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
    // CHECK-SAME: selection_kind = "fallback-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
    // CHECK: tcrv_scalar.lowering_boundary
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "public_scalar_only_scalar_boundary"
    // CHECK-SAME: status = "metadata-only"
    // CHECK-NOT: tcrv_rvv.lowering_boundary
    // CHECK: tcrv_scalar.i32_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "runtime-callable-c-source"
    // CHECK-SAME: emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source"
    // CHECK-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
    // CHECK-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
    // CHECK-SAME: selected_plan_metadata = [{{.*}}name = "tcrv_scalar.emitc_source_op"{{.*}}value = "tcrv_scalar.i32_vadd_microkernel"{{.*}}name = "tcrv_scalar.emitc_lowerable_op_interface"{{.*}}value = "TCRVEmitCLowerableOpInterface"
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_without_scalar_boundary
  tcrv.exec.kernel @public_rvv_without_scalar_boundary {
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
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "fallback-coverage-missing"
    // CHECK-SAME: selection_kind = "missing-conservative-fallback"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "public_rvv_without_scalar_boundary"
    // CHECK-SAME: status = "unsupported"
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
