// RUN: not tcrv-opt %s --tcrv-check-capability-requires 2>&1 | FileCheck %s

tcrv.exec.kernel @disabled_runtime attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime",
    kind = "runtime-offload",
    status = "disabled"
  }
  // CHECK: error: static variant @portable_path requires unavailable capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "disabled"
  // CHECK-SAME: kernel @disabled_runtime
  // CHECK-SAME: not protected by tcrv.exec.dispatch case
  tcrv.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@runtime_probe]
  } {
  }
}

tcrv.exec.kernel @unguarded_dispatch_case attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  tcrv.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    // CHECK: error: unguarded dispatch case in kernel @unguarded_dispatch_case targets variant @runtime_offload_path with unavailable required capability @runtime_probe
    // CHECK-SAME: id = "portable.runtime.probe"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: runtime_guard_required
    // CHECK-SAME: condition/guard/policy annotations alone are not semantic guard requirements
    tcrv.exec.case @runtime_offload_path {
      condition = "legacy_condition_annotation",
      guard = "legacy_guard_annotation",
      policy = "legacy_policy_annotation"
    }
    tcrv.exec.fallback @portable_fallback
  }
}

tcrv.exec.kernel @unavailable_fallback attributes {} {
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  tcrv.exec.capability @portable_runtime {
    id = "portable.runtime",
    kind = "runtime-offload",
    availability = "unavailable"
  }
  tcrv.exec.variant @available_case_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "runtime-fallback-plugin",
    requires = [@portable_runtime]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @available_case_path {condition = "generic_toolchain_available"}
    // CHECK: error: dispatch fallback in kernel @unavailable_fallback targets variant @portable_fallback with unavailable required capability @portable_runtime
    // CHECK-SAME: id = "portable.runtime"
    // CHECK-SAME: kind = "runtime-offload"
    // CHECK-SAME: status = "unavailable"
    tcrv.exec.fallback @portable_fallback
  }
}

tcrv.exec.kernel @static_conflict attributes {} {
  tcrv.exec.capability @inline_asm {
    id = "vendor.inline_asm",
    kind = "toolchain",
    conflicts = ["build.policy.no_inline_asm"],
    status = "available"
  }
  tcrv.exec.capability @no_inline {
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
  // CHECK-SAME: not protected by tcrv.exec.dispatch case
  tcrv.exec.variant @inline_asm_path attributes {
    origin = "inline-asm-plugin",
    requires = [@inline_asm]
  } {
  }
}

tcrv.exec.kernel @unguarded_conflict_dispatch attributes {} {
  tcrv.exec.capability @fixed_shape_runtime {
    id = "runtime.fixed_shape",
    kind = "runtime-offload",
    conflicts = ["shape.dynamic"],
    status = "available"
  }
  tcrv.exec.capability @shape_profile {
    id = "shape.profile",
    kind = "shape-policy",
    provides = ["shape.dynamic"],
    status = "available"
  }
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  tcrv.exec.variant @fixed_shape_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@fixed_shape_runtime]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    // CHECK: error: unguarded dispatch case in kernel @unguarded_conflict_dispatch targets variant @fixed_shape_path with conflicting required capability @fixed_shape_runtime
    // CHECK-SAME: id = "runtime.fixed_shape"
    // CHECK-SAME: conflicting with available capability @shape_profile
    // CHECK-SAME: id = "shape.profile"
    // CHECK-SAME: via conflict id "shape.dynamic"
    // CHECK-SAME: runtime_guard_required
    tcrv.exec.case @fixed_shape_path
    tcrv.exec.fallback @portable_fallback
  }
}

tcrv.exec.kernel @conflicting_fallback attributes {} {
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  tcrv.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    conflicts = ["build.policy.no_scalar_fallback"],
    status = "available"
  }
  tcrv.exec.capability @no_scalar_fallback {
    id = "build.policy.no_scalar_fallback",
    kind = "build-policy",
    status = "available"
  }
  tcrv.exec.variant @available_case_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "runtime-fallback-plugin",
    requires = [@scalar_fallback]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    tcrv.exec.case @available_case_path {condition = "generic_toolchain_available"}
    // CHECK: error: dispatch fallback in kernel @conflicting_fallback targets variant @portable_fallback with conflicting required capability @scalar_fallback
    // CHECK-SAME: id = "scalar.fallback"
    // CHECK-SAME: conflicting with available capability @no_scalar_fallback
    // CHECK-SAME: id = "build.policy.no_scalar_fallback"
    // CHECK-SAME: via conflict id "build.policy.no_scalar_fallback"
    tcrv.exec.fallback @portable_fallback
  }
}
