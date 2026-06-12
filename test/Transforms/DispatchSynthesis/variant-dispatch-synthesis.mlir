// RUN: tcrv-opt %s --tcrv-synthesize-variant-dispatch | FileCheck %s --check-prefix=SYNTH
// RUN: tcrv-opt %s --tcrv-synthesize-variant-dispatch --tcrv-check-capability-requires | FileCheck %s --check-prefix=PIPE

// SYNTH-LABEL: tcrv.exec.kernel @first_available_is_fallback
// PIPE-LABEL: tcrv.exec.kernel @first_available_is_fallback
tcrv.exec.kernel @first_available_is_fallback attributes {} {
  tcrv.exec.capability @runtime_probe {
    id = "generic.runtime.probe",
    kind = "runtime",
    status = "missing"
  }
  tcrv.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  tcrv.exec.capability @extra_capability {
    id = "generic.extra",
    kind = "toolchain",
    status = "available"
  }
  // SYNTH: tcrv.exec.variant @runtime_path
  // SYNTH-SAME: condition = "runtime_probe_available"
  // SYNTH-SAME: guard = "runtime_guard_passed"
  tcrv.exec.variant @runtime_path attributes {
    condition = "runtime_probe_available",
    guard = "runtime_guard_passed",
    origin = "runtime-plugin",
    policy = "prefer_runtime_when_guarded",
    requires = [@runtime_probe]
  } {
  }
  // SYNTH: tcrv.exec.variant @baseline_path
  tcrv.exec.variant @baseline_path attributes {
    fallback_role = "conservative",
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: tcrv.exec.variant @extra_path
  tcrv.exec.variant @extra_path attributes {
    origin = "extra-plugin",
    requires = [@extra_capability]
  } {
  }
  // SYNTH: tcrv.exec.dispatch
  // SYNTH: tcrv.exec.case @runtime_path
  // SYNTH-SAME: condition = "runtime_probe_available"
  // SYNTH-SAME: guard = "runtime_guard_passed"
  // SYNTH-SAME: policy = "prefer_runtime_when_guarded"
  // SYNTH-SAME: runtime_guard_required = true
  // SYNTH: tcrv.exec.case @extra_path
  // SYNTH: tcrv.exec.fallback @baseline_path
  // SYNTH-NOT: tcrv.exec.dispatch
  // PIPE: tcrv.exec.dispatch
  // PIPE: tcrv.exec.case @runtime_path
  // PIPE-SAME: condition = "runtime_probe_available"
  // PIPE-SAME: guard = "runtime_guard_passed"
  // PIPE-SAME: policy = "prefer_runtime_when_guarded"
  // PIPE-SAME: runtime_guard_required = true
  // PIPE: tcrv.exec.case @extra_path
  // PIPE: tcrv.exec.fallback @baseline_path
}

// SYNTH-LABEL: tcrv.exec.kernel @conflicting_available_case_with_clean_fallback
// PIPE-LABEL: tcrv.exec.kernel @conflicting_available_case_with_clean_fallback
tcrv.exec.kernel @conflicting_available_case_with_clean_fallback attributes {} {
  tcrv.exec.capability @fast_runtime {
    id = "generic.fast.runtime",
    kind = "runtime",
    relations = #tcrv.capability_relations<conflicts = ["build.policy.disable_fast_runtime"]>,
    status = "available"
  }
  tcrv.exec.capability @disable_fast_profile {
    id = "generic.build.profile",
    kind = "build-policy",
    relations = #tcrv.capability_relations<provides = ["build.policy.disable_fast_runtime"]>,
    status = "available"
  }
  tcrv.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain",
    status = "available"
  }
  tcrv.exec.variant @fast_runtime_path attributes {
    origin = "fast-runtime-plugin",
    requires = [@fast_runtime]
  } {
  }
  tcrv.exec.variant @baseline_path attributes {
    fallback_role = "conservative",
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: tcrv.exec.dispatch
  // SYNTH: tcrv.exec.case @fast_runtime_path
  // SYNTH-SAME: policy = "capability_dispatch_guard"
  // SYNTH-SAME: runtime_guard_required = true
  // SYNTH: tcrv.exec.fallback @baseline_path
  // PIPE: tcrv.exec.dispatch
  // PIPE: tcrv.exec.case @fast_runtime_path
  // PIPE-SAME: policy = "capability_dispatch_guard"
  // PIPE-SAME: runtime_guard_required = true
  // PIPE: tcrv.exec.fallback @baseline_path
}

// SYNTH-LABEL: tcrv.exec.kernel @already_dispatched
// PIPE-LABEL: tcrv.exec.kernel @already_dispatched
tcrv.exec.kernel @already_dispatched attributes {} {
  tcrv.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  tcrv.exec.variant @baseline_path attributes {
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  tcrv.exec.variant @alternate_path attributes {
    origin = "alternate-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // SYNTH: tcrv.exec.case @alternate_path
    tcrv.exec.case @alternate_path {condition = "existing_generic_guard"}
    // SYNTH: tcrv.exec.fallback @baseline_path
    tcrv.exec.fallback @baseline_path
  }
  // SYNTH-NOT: tcrv.exec.dispatch
}

// SYNTH-LABEL: tcrv.exec.kernel @single_available_unchanged
tcrv.exec.kernel @single_available_unchanged attributes {} {
  tcrv.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  tcrv.exec.variant @baseline_path attributes {
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
  // SYNTH-NOT: tcrv.exec.dispatch
}
