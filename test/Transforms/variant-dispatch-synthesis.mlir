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
  tcrv.exec.variant @runtime_path attributes {
    origin = "runtime-plugin",
    requires = [@runtime_probe]
  } {
  }
  // SYNTH: tcrv.exec.variant @baseline_path
  tcrv.exec.variant @baseline_path attributes {
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
  // SYNTH-SAME: policy = "capability_dispatch_guard"
  // SYNTH: tcrv.exec.case @extra_path
  // SYNTH: tcrv.exec.fallback @baseline_path
  // SYNTH-NOT: tcrv.exec.dispatch
  // PIPE: tcrv.exec.dispatch
  // PIPE: tcrv.exec.case @runtime_path
  // PIPE-SAME: policy = "capability_dispatch_guard"
  // PIPE: tcrv.exec.case @extra_path
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
