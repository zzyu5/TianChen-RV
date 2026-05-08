// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-dispatch-runtime-guards | FileCheck %s --check-prefix=GUARD
// RUN: tcrv-opt %s --split-input-file --tcrv-synthesize-variant-dispatch --tcrv-materialize-dispatch-runtime-guards --tcrv-check-capability-requires | FileCheck %s --check-prefix=SYNTH

module {
  // GUARD-LABEL: tcrv.exec.kernel @direct_guarded_dispatch
  // SYNTH-LABEL: tcrv.exec.kernel @direct_guarded_dispatch
  tcrv.exec.kernel @direct_guarded_dispatch {
    tcrv.exec.capability @runtime_probe {
      id = "generic.runtime.probe",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain",
      status = "available"
    }
    tcrv.exec.variant @runtime_path attributes {
      condition = "runtime_probe_available",
      guard = "runtime_guard_passed",
      origin = "runtime-plugin",
      policy = "prefer_runtime_when_guarded",
      requires = [@runtime_probe]
    } {
    }
    tcrv.exec.variant @baseline_path attributes {
      fallback_role = "conservative",
      origin = "baseline-plugin",
      requires = [@baseline_capability]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @runtime_path {
        condition = "runtime_probe_available",
        guard = "runtime_guard_passed",
        policy = "prefer_runtime_when_guarded"
      }
      tcrv.exec.fallback @baseline_path
    }

    // GUARD: tcrv.exec.runtime_param @abi_dispatch_availability_guard
    // GUARD-SAME: abi_role = "dispatch-availability-guard"
    // GUARD-SAME: c_name = "rvv_available"
    // GUARD-SAME: c_type = "int"
    // GUARD-SAME: ownership = "target-export-abi-owned"
    // GUARD-SAME: purpose = "runtime-abi-scalar"
    // GUARD: tcrv.exec.case @runtime_path
    // GUARD-SAME: condition = "runtime_probe_available"
    // GUARD-SAME: guard = "runtime_guard_passed"
    // GUARD-SAME: policy = "prefer_runtime_when_guarded"
    // GUARD-SAME: runtime_guard = @abi_dispatch_availability_guard
    // GUARD: tcrv.exec.fallback @baseline_path
    // GUARD-NOT: runtime_guard
    // GUARD: }
    // SYNTH: tcrv.exec.runtime_param @abi_dispatch_availability_guard
    // SYNTH: tcrv.exec.case @runtime_path
    // SYNTH-SAME: runtime_guard = @abi_dispatch_availability_guard
  }
}

// -----

module {
  // SYNTH-LABEL: tcrv.exec.kernel @synthesized_guarded_dispatch
  tcrv.exec.kernel @synthesized_guarded_dispatch {
    tcrv.exec.capability @fast_runtime {
      id = "generic.fast.runtime",
      kind = "runtime",
      conflicts = ["build.policy.disable_fast_runtime"],
      status = "available"
    }
    tcrv.exec.capability @disable_fast_profile {
      id = "generic.build.profile",
      kind = "build-policy",
      provides = ["build.policy.disable_fast_runtime"],
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

    // SYNTH: tcrv.exec.runtime_param @abi_dispatch_availability_guard
    // SYNTH-SAME: abi_role = "dispatch-availability-guard"
    // SYNTH: tcrv.exec.dispatch
    // SYNTH: tcrv.exec.case @fast_runtime_path
    // SYNTH-SAME: policy = "capability_dispatch_guard"
    // SYNTH-SAME: runtime_guard = @abi_dispatch_availability_guard
    // SYNTH: tcrv.exec.fallback @baseline_path
    // SYNTH-NOT: runtime_guard
    // SYNTH: }
  }
}
