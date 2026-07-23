// RUN: weft-opt %s --weft-check-capability-requires -split-input-file | FileCheck %s

// CHECK-LABEL: weft.exec.kernel @all_available
weft.exec.kernel @all_available attributes {} {
  // CHECK: weft.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  weft.exec.capability @generic_toolchain {id = "generic.toolchain", kind = "toolchain"}

  // CHECK: weft.exec.capability @portable_runtime
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "available"
  weft.exec.capability @portable_runtime {id = "portable.runtime", kind = "runtime-offload", status = "available"}

  // CHECK: weft.exec.variant @portable_path
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain, @portable_runtime]
  weft.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain, @portable_runtime]
  } {
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @guarded_runtime_dispatch
weft.exec.kernel @guarded_runtime_dispatch attributes {} {
  // CHECK: weft.exec.capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime.probe"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "unavailable"
  weft.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "unavailable"
  }

  // CHECK: weft.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }

  // CHECK: weft.exec.variant @runtime_offload_path
  // CHECK-SAME: origin = "runtime-offload-plugin"
  // CHECK-SAME: requires = [@runtime_probe]
  weft.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }

  // CHECK: weft.exec.variant @portable_fallback
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain]
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }

  // CHECK: weft.exec.dispatch
  weft.exec.dispatch attributes {} {
    // CHECK: weft.exec.case @runtime_offload_path
    // CHECK-SAME: policy = "runtime_probe_selects_path"
    // CHECK-SAME: runtime_guard_required = true
    weft.exec.case @runtime_offload_path {
      policy = "runtime_probe_selects_path",
      runtime_guard_required = true
    }
    // CHECK: weft.exec.fallback @portable_fallback
    weft.exec.fallback @portable_fallback
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @guarded_by_inherited_case_metadata
weft.exec.kernel @guarded_by_inherited_case_metadata attributes {} {
  weft.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  weft.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  // CHECK: weft.exec.variant @runtime_offload_path
  // CHECK-SAME: condition = "runtime_probe_available"
  // CHECK-SAME: guard = "runtime_guard_passed"
  // CHECK-SAME: policy = "prefer_runtime_when_guarded"
  weft.exec.variant @runtime_offload_path attributes {
    condition = "runtime_probe_available",
    guard = "runtime_guard_passed",
    origin = "runtime-offload-plugin",
    policy = "prefer_runtime_when_guarded",
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
    // CHECK: weft.exec.case @runtime_offload_path
    // CHECK-SAME: condition = "runtime_probe_available"
    // CHECK-SAME: guard = "runtime_guard_passed"
    // CHECK-SAME: policy = "prefer_runtime_when_guarded"
    // CHECK-SAME: runtime_guard_required = true
    weft.exec.case @runtime_offload_path {
      condition = "runtime_probe_available",
      guard = "runtime_guard_passed",
      policy = "prefer_runtime_when_guarded",
      runtime_guard_required = true
    }
    weft.exec.fallback @portable_fallback
  }
}

// -----

// The conflict is expressed via the typed
// relations = #weft.capability_relations<...> attribute. The guarded dispatch
// case is accepted (no error) because the conflict resolution -- driven by the
// typed attr through the descriptor bridge -- detects the conflict and the
// runtime_guard_required = true protects it. This proves CheckCapabilityRequires
// reads typed-sourced relations.
// CHECK-LABEL: weft.exec.kernel @guarded_conflict_dispatch
weft.exec.kernel @guarded_conflict_dispatch attributes {} {
  // CHECK: weft.exec.capability @inline_asm
  // CHECK-SAME: relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm"]>
  weft.exec.capability @inline_asm {
    id = "vendor.inline_asm",
    kind = "toolchain",
    relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm"]>,
    status = "available"
  }
  // CHECK: weft.exec.capability @no_inline_profile
  // CHECK-SAME: relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>
  weft.exec.capability @no_inline_profile {
    id = "build.policy.profile",
    kind = "build-policy",
    relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>,
    status = "available"
  }
  weft.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }
  // CHECK: weft.exec.variant @inline_asm_path
  // CHECK-SAME: requires = [@inline_asm]
  weft.exec.variant @inline_asm_path attributes {
    origin = "inline-asm-plugin",
    requires = [@inline_asm]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "portable-plugin",
    requires = [@scalar_fallback]
  } {
  }
  // CHECK: weft.exec.dispatch
  weft.exec.dispatch attributes {} {
    // CHECK: weft.exec.case @inline_asm_path
    // CHECK-SAME: guard = "runtime_policy_allows_inline_asm"
    // CHECK-SAME: runtime_guard_required = true
    weft.exec.case @inline_asm_path {
      guard = "runtime_policy_allows_inline_asm",
      runtime_guard_required = true
    }
    weft.exec.fallback @portable_fallback
  }
}
