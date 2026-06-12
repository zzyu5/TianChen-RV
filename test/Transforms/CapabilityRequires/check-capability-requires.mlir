// RUN: tcrv-opt %s --tcrv-check-capability-requires -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @all_available
tcrv.exec.kernel @all_available attributes {} {
  // CHECK: tcrv.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  tcrv.exec.capability @generic_toolchain {id = "generic.toolchain", kind = "toolchain"}

  // CHECK: tcrv.exec.capability @portable_runtime
  // CHECK-SAME: id = "portable.runtime"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "available"
  tcrv.exec.capability @portable_runtime {id = "portable.runtime", kind = "runtime-offload", status = "available"}

  // CHECK: tcrv.exec.variant @portable_path
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain, @portable_runtime]
  tcrv.exec.variant @portable_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain, @portable_runtime]
  } {
  }
}

// -----

// CHECK-LABEL: tcrv.exec.kernel @guarded_runtime_dispatch
tcrv.exec.kernel @guarded_runtime_dispatch attributes {} {
  // CHECK: tcrv.exec.capability @runtime_probe
  // CHECK-SAME: id = "portable.runtime.probe"
  // CHECK-SAME: kind = "runtime-offload"
  // CHECK-SAME: status = "unavailable"
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "unavailable"
  }

  // CHECK: tcrv.exec.capability @generic_toolchain
  // CHECK-SAME: id = "generic.toolchain"
  // CHECK-SAME: kind = "toolchain"
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }

  // CHECK: tcrv.exec.variant @runtime_offload_path
  // CHECK-SAME: origin = "runtime-offload-plugin"
  // CHECK-SAME: requires = [@runtime_probe]
  tcrv.exec.variant @runtime_offload_path attributes {
    origin = "runtime-offload-plugin",
    requires = [@runtime_probe]
  } {
  }

  // CHECK: tcrv.exec.variant @portable_fallback
  // CHECK-SAME: origin = "portable-plugin"
  // CHECK-SAME: requires = [@generic_toolchain]
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }

  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @runtime_offload_path
    // CHECK-SAME: policy = "runtime_probe_selects_path"
    // CHECK-SAME: runtime_guard_required = true
    tcrv.exec.case @runtime_offload_path {
      policy = "runtime_probe_selects_path",
      runtime_guard_required = true
    }
    // CHECK: tcrv.exec.fallback @portable_fallback
    tcrv.exec.fallback @portable_fallback
  }
}

// -----

// CHECK-LABEL: tcrv.exec.kernel @guarded_by_inherited_case_metadata
tcrv.exec.kernel @guarded_by_inherited_case_metadata attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "portable.runtime.probe",
    kind = "runtime-offload",
    status = "missing"
  }
  tcrv.exec.capability @generic_toolchain {
    id = "generic.toolchain",
    kind = "toolchain"
  }
  // CHECK: tcrv.exec.variant @runtime_offload_path
  // CHECK-SAME: condition = "runtime_probe_available"
  // CHECK-SAME: guard = "runtime_guard_passed"
  // CHECK-SAME: policy = "prefer_runtime_when_guarded"
  tcrv.exec.variant @runtime_offload_path attributes {
    condition = "runtime_probe_available",
    guard = "runtime_guard_passed",
    origin = "runtime-offload-plugin",
    policy = "prefer_runtime_when_guarded",
    requires = [@runtime_probe]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @runtime_offload_path
    // CHECK-SAME: condition = "runtime_probe_available"
    // CHECK-SAME: guard = "runtime_guard_passed"
    // CHECK-SAME: policy = "prefer_runtime_when_guarded"
    // CHECK-SAME: runtime_guard_required = true
    tcrv.exec.case @runtime_offload_path {
      condition = "runtime_probe_available",
      guard = "runtime_guard_passed",
      policy = "prefer_runtime_when_guarded",
      runtime_guard_required = true
    }
    tcrv.exec.fallback @portable_fallback
  }
}

// -----

// The conflict is expressed via the typed
// relations = #tcrv.capability_relations<...> attribute. The guarded dispatch
// case is accepted (no error) because the conflict resolution -- driven by the
// typed attr through the descriptor bridge -- detects the conflict and the
// runtime_guard_required = true protects it. This proves CheckCapabilityRequires
// reads typed-sourced relations.
// CHECK-LABEL: tcrv.exec.kernel @guarded_conflict_dispatch
tcrv.exec.kernel @guarded_conflict_dispatch attributes {} {
  // CHECK: tcrv.exec.capability @inline_asm
  // CHECK-SAME: relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]>
  tcrv.exec.capability @inline_asm {
    id = "vendor.inline_asm",
    kind = "toolchain",
    relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]>,
    status = "available"
  }
  // CHECK: tcrv.exec.capability @no_inline_profile
  // CHECK-SAME: relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>
  tcrv.exec.capability @no_inline_profile {
    id = "build.policy.profile",
    kind = "build-policy",
    relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>,
    status = "available"
  }
  tcrv.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }
  // CHECK: tcrv.exec.variant @inline_asm_path
  // CHECK-SAME: requires = [@inline_asm]
  tcrv.exec.variant @inline_asm_path attributes {
    origin = "inline-asm-plugin",
    requires = [@inline_asm]
  } {
  }
  tcrv.exec.variant @portable_fallback attributes {
    origin = "portable-plugin",
    requires = [@scalar_fallback]
  } {
  }
  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.case @inline_asm_path
    // CHECK-SAME: guard = "runtime_policy_allows_inline_asm"
    // CHECK-SAME: runtime_guard_required = true
    tcrv.exec.case @inline_asm_path {
      guard = "runtime_policy_allows_inline_asm",
      runtime_guard_required = true
    }
    tcrv.exec.fallback @portable_fallback
  }
}
