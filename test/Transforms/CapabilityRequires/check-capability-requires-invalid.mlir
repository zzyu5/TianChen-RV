// RUN: not weft-opt %s --weft-check-capability-requires 2>&1 | FileCheck %s

weft.exec.kernel @disabled_runtime attributes {} {
  weft.exec.capability @runtime_probe {
    id = "portable.runtime",
    kind = "runtime-offload",
    status = "disabled"
  }
  // CHECK: error: static variant @portable_path requires unavailable capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "disabled"
  // CHECK-SAME: kernel @disabled_runtime
  // CHECK-SAME: not protected by weft.exec.dispatch case
  weft.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@runtime_probe]
  } {
  }
}

weft.exec.kernel @unguarded_dispatch_case attributes {} {
  weft.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  weft.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  weft.exec.dispatch attributes {} {
    // CHECK: error: unguarded dispatch case in kernel @unguarded_dispatch_case targets variant @runtime_offload_path with unavailable required capability @runtime_probe
    // CHECK-SAME: id = "portable.runtime.probe"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: runtime_guard_required
    // CHECK-SAME: condition/guard/policy annotations alone are not semantic guard requirements
    weft.exec.case @runtime_offload_path {
      condition = "legacy_condition_annotation",
      guard = "legacy_guard_annotation",
      policy = "legacy_policy_annotation"
    }
    weft.exec.fallback @portable_fallback
  }
}

weft.exec.kernel @unavailable_fallback attributes {} {
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  weft.exec.capability @portable_runtime {
    id = "portable.runtime",
    kind = "runtime-offload",
    status = "unavailable"
  }
  weft.exec.variant @available_case_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "runtime-fallback-plugin",
    requires = [@portable_runtime]
  } {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @available_case_path {condition = "generic_toolchain_available"}
    // CHECK: error: dispatch fallback in kernel @unavailable_fallback targets variant @portable_fallback with unavailable required capability @portable_runtime
    // CHECK-SAME: id = "portable.runtime"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "unavailable"
    weft.exec.fallback @portable_fallback
  }
}

weft.exec.kernel @static_conflict attributes {} {
  weft.exec.capability @inline_asm {
    id = "vendor.inline_asm",
    kind = "toolchain",
    relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm"]>,
    status = "available"
  }
  weft.exec.capability @no_inline {
    id = "build.policy.no_inline_asm",
    kind = "build-policy",
    status = "available"
  }
  // CHECK: error: static variant @inline_asm_path requires conflicting capability @inline_asm
  // CHECK-SAME: id = "vendor.inline_asm"
  // CHECK-SAME: kind = "toolchain"
  // CHECK-SAME: conflicting with available capability @no_inline
  // CHECK-SAME: id = "build.policy.no_inline_asm"
  // CHECK-SAME: via conflict id "build.policy.no_inline_asm"
  // CHECK-SAME: kernel @static_conflict
  // CHECK-SAME: not protected by weft.exec.dispatch case
  weft.exec.variant @inline_asm_path attributes {
    origin = "inline-asm-plugin",
    requires = [@inline_asm]
  } {
  }
}

// The conflicting capability relation is expressed via the typed
// relations = #weft.capability_relations<conflicts = ["shape.dynamic"]> attr and
// the provider via relations = #weft.capability_relations<provides = ...>. The
// unguarded-dispatch conflict legality decision is driven by the typed
// attribute through the descriptor bridge.
weft.exec.kernel @unguarded_conflict_dispatch attributes {} {
  weft.exec.capability @fixed_shape_runtime {
    id = "runtime.fixed_shape",
    kind = "runtime-offload",
    relations = #weft.capability_relations<conflicts = ["shape.dynamic"]>,
    status = "available"
  }
  weft.exec.capability @shape_profile {
    id = "shape.profile",
    kind = "shape-policy",
    relations = #weft.capability_relations<provides = ["shape.dynamic"]>,
    status = "available"
  }
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  weft.exec.variant @fixed_shape_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@fixed_shape_runtime]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  weft.exec.dispatch attributes {} {
    // CHECK: error: unguarded dispatch case in kernel @unguarded_conflict_dispatch targets variant @fixed_shape_path with conflicting required capability @fixed_shape_runtime
    // CHECK-SAME: id = "runtime.fixed_shape"
    // CHECK-SAME: conflicting with available capability @shape_profile
    // CHECK-SAME: id = "shape.profile"
    // CHECK-SAME: via conflict id "shape.dynamic"
    // CHECK-SAME: runtime_guard_required
    weft.exec.case @fixed_shape_path
    weft.exec.fallback @portable_fallback
  }
}

weft.exec.kernel @conflicting_fallback attributes {} {
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  weft.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    relations = #weft.capability_relations<conflicts = ["build.policy.no_scalar_fallback"]>,
    status = "available"
  }
  weft.exec.capability @no_scalar_fallback {
    id = "build.policy.no_scalar_fallback",
    kind = "build-policy",
    status = "available"
  }
  weft.exec.variant @available_case_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "runtime-fallback-plugin",
    requires = [@scalar_fallback]
  } {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @available_case_path {condition = "generic_toolchain_available"}
    // CHECK: error: dispatch fallback in kernel @conflicting_fallback targets variant @portable_fallback with conflicting required capability @scalar_fallback
    // CHECK-SAME: id = "scalar.fallback"
    // CHECK-SAME: conflicting with available capability @no_scalar_fallback
    // CHECK-SAME: id = "build.policy.no_scalar_fallback"
    // CHECK-SAME: via conflict id "build.policy.no_scalar_fallback"
    weft.exec.fallback @portable_fallback
  }
}
