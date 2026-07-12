// RUN: weft-opt %s --weft-synthesize-variant-dispatch | FileCheck %s --check-prefix=SYNTH
// RUN: weft-opt %s --weft-synthesize-variant-dispatch --weft-check-capability-requires | FileCheck %s --check-prefix=PIPE

// SYNTH-LABEL: weft.exec.kernel @first_available_is_fallback
// PIPE-LABEL: weft.exec.kernel @first_available_is_fallback
weft.exec.kernel @first_available_is_fallback attributes {} {
  weft.exec.capability @runtime_probe {
    id = "generic.runtime.probe",
    kind = "runtime",
    status = "missing"
  }
  weft.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  weft.exec.capability @extra_capability {
    id = "generic.extra",
    kind = "toolchain",
    status = "available"
  }
  // SYNTH: weft.exec.variant @runtime_path
  // SYNTH-SAME: condition = "runtime_probe_available"
  // SYNTH-SAME: guard = "runtime_guard_passed"
  weft.exec.variant @runtime_path attributes {
    condition = "runtime_probe_available",
    guard = "runtime_guard_passed",
    origin = "runtime-plugin",
    policy = "prefer_runtime_when_guarded",
    requires = [@runtime_probe]
  } {
  }
  // SYNTH: weft.exec.variant @baseline_path
  weft.exec.variant @baseline_path attributes {
    fallback_role = "conservative",
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: weft.exec.variant @extra_path
  weft.exec.variant @extra_path attributes {
    origin = "extra-plugin",
    requires = [@extra_capability]
  } {
  }
  // SYNTH: weft.exec.dispatch
  // SYNTH: weft.exec.case @runtime_path
  // SYNTH-SAME: condition = "runtime_probe_available"
  // SYNTH-SAME: guard = "runtime_guard_passed"
  // SYNTH-SAME: policy = "prefer_runtime_when_guarded"
  // SYNTH-SAME: runtime_guard_required = true
  // SYNTH: weft.exec.case @extra_path
  // SYNTH: weft.exec.fallback @baseline_path
  // SYNTH-NOT: weft.exec.dispatch
  // PIPE: weft.exec.dispatch
  // PIPE: weft.exec.case @runtime_path
  // PIPE-SAME: condition = "runtime_probe_available"
  // PIPE-SAME: guard = "runtime_guard_passed"
  // PIPE-SAME: policy = "prefer_runtime_when_guarded"
  // PIPE-SAME: runtime_guard_required = true
  // PIPE: weft.exec.case @extra_path
  // PIPE: weft.exec.fallback @baseline_path
}

// SYNTH-LABEL: weft.exec.kernel @conflicting_available_case_with_clean_fallback
// PIPE-LABEL: weft.exec.kernel @conflicting_available_case_with_clean_fallback
weft.exec.kernel @conflicting_available_case_with_clean_fallback attributes {} {
  weft.exec.capability @fast_runtime {
    id = "generic.fast.runtime",
    kind = "runtime",
    relations = #weft.capability_relations<conflicts = ["build.policy.disable_fast_runtime"]>,
    status = "available"
  }
  weft.exec.capability @disable_fast_profile {
    id = "generic.build.profile",
    kind = "build-policy",
    relations = #weft.capability_relations<provides = ["build.policy.disable_fast_runtime"]>,
    status = "available"
  }
  weft.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain",
    status = "available"
  }
  weft.exec.variant @fast_runtime_path attributes {
    origin = "fast-runtime-plugin",
    requires = [@fast_runtime]
  } {
  }
  weft.exec.variant @baseline_path attributes {
    fallback_role = "conservative",
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: weft.exec.dispatch
  // SYNTH: weft.exec.case @fast_runtime_path
  // SYNTH-SAME: policy = "capability_dispatch_guard"
  // SYNTH-SAME: runtime_guard_required = true
  // SYNTH: weft.exec.fallback @baseline_path
  // PIPE: weft.exec.dispatch
  // PIPE: weft.exec.case @fast_runtime_path
  // PIPE-SAME: policy = "capability_dispatch_guard"
  // PIPE-SAME: runtime_guard_required = true
  // PIPE: weft.exec.fallback @baseline_path
}

// SYNTH-LABEL: weft.exec.kernel @already_dispatched
// PIPE-LABEL: weft.exec.kernel @already_dispatched
weft.exec.kernel @already_dispatched attributes {} {
  weft.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  weft.exec.variant @baseline_path attributes {
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  weft.exec.variant @alternate_path attributes {
    origin = "alternate-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: weft.exec.dispatch
  weft.exec.dispatch attributes {} {
    // SYNTH: weft.exec.case @alternate_path
    weft.exec.case @alternate_path {condition = "existing_generic_guard"}
    // SYNTH: weft.exec.fallback @baseline_path
    weft.exec.fallback @baseline_path
  }
  // SYNTH-NOT: weft.exec.dispatch
}

// SYNTH-LABEL: weft.exec.kernel @single_available_unchanged
weft.exec.kernel @single_available_unchanged attributes {} {
  weft.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  weft.exec.variant @baseline_path attributes {
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH-NOT: weft.exec.dispatch
}
