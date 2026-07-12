// RUN: weft-opt %s --split-input-file --weft-materialize-dispatch-runtime-guards | FileCheck %s --check-prefix=GUARD
// RUN: weft-opt %s --split-input-file --weft-synthesize-variant-dispatch --weft-materialize-dispatch-runtime-guards --weft-check-capability-requires | FileCheck %s --check-prefix=SYNTH

module {
  // GUARD-LABEL: weft.exec.kernel @direct_guarded_dispatch
  // SYNTH-LABEL: weft.exec.kernel @direct_guarded_dispatch
  weft.exec.kernel @direct_guarded_dispatch {
    weft.exec.capability @runtime_probe {
      id = "generic.runtime.probe",
      kind = "runtime",
      status = "missing"
    }
    weft.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain",
      status = "available"
    }
    weft.exec.variant @runtime_path attributes {
      condition = "runtime_probe_available",
      guard = "runtime_guard_passed",
      origin = "runtime-plugin",
      policy = "prefer_runtime_when_guarded",
      requires = [@runtime_probe]
    } {
    }
    weft.exec.variant @baseline_path attributes {
      fallback_role = "conservative",
      origin = "baseline-plugin",
      requires = [@baseline_capability]
    } {
    }
    weft.exec.dispatch {
      weft.exec.case @runtime_path {
        condition = "runtime_probe_available",
        guard = "runtime_guard_passed",
        policy = "prefer_runtime_when_guarded",
        runtime_guard_required = true
      }
      weft.exec.fallback @baseline_path
    }

    // GUARD: weft.exec.runtime_param @abi_dispatch_availability_guard
    // GUARD-SAME: abi_role = "dispatch-availability-guard"
    // GUARD-SAME: c_name = "dispatch_available"
    // GUARD-SAME: c_type = "int"
    // GUARD-SAME: ownership = "target-export-abi-owned"
    // GUARD-SAME: purpose = "runtime-abi-scalar"
    // GUARD: weft.exec.case @runtime_path
    // GUARD-SAME: condition = "runtime_probe_available"
    // GUARD-SAME: guard = "runtime_guard_passed"
    // GUARD-SAME: policy = "prefer_runtime_when_guarded"
    // GUARD-SAME: runtime_guard = @abi_dispatch_availability_guard
    // GUARD-SAME: runtime_guard_required = true
    // GUARD: weft.exec.fallback @baseline_path
    // GUARD-NOT: runtime_guard
    // GUARD: }
    // SYNTH: weft.exec.runtime_param @abi_dispatch_availability_guard
    // SYNTH: weft.exec.case @runtime_path
    // SYNTH-SAME: runtime_guard = @abi_dispatch_availability_guard
    // SYNTH-SAME: runtime_guard_required = true
  }
}

// -----

module {
  // GUARD-LABEL: weft.exec.kernel @annotated_available_dispatch_no_typed_requirement
  weft.exec.kernel @annotated_available_dispatch_no_typed_requirement {
    weft.exec.capability @optional_runtime {
      id = "generic.optional.runtime",
      kind = "runtime",
      status = "available"
    }
    weft.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain",
      status = "available"
    }
    weft.exec.variant @annotated_path attributes {
      condition = "human_readable_condition_only",
      guard = "human_readable_guard_only",
      origin = "runtime-plugin",
      policy = "human_readable_policy_only",
      requires = [@optional_runtime]
    } {
    }
    weft.exec.variant @baseline_path attributes {
      fallback_role = "conservative",
      origin = "baseline-plugin",
      requires = [@baseline_capability]
    } {
    }
    weft.exec.dispatch {
      // GUARD-NOT: weft.exec.runtime_param @abi_dispatch_availability_guard
      // GUARD: weft.exec.case @annotated_path
      // GUARD-SAME: condition = "human_readable_condition_only"
      // GUARD-SAME: guard = "human_readable_guard_only"
      // GUARD-SAME: policy = "human_readable_policy_only"
      // GUARD-NOT: runtime_guard
      weft.exec.case @annotated_path {
        condition = "human_readable_condition_only",
        guard = "human_readable_guard_only",
        policy = "human_readable_policy_only"
      }
      // GUARD: weft.exec.fallback @baseline_path
      weft.exec.fallback @baseline_path
    }
  }
}

// -----

module {
  // SYNTH-LABEL: weft.exec.kernel @synthesized_guarded_dispatch
  weft.exec.kernel @synthesized_guarded_dispatch {
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

    // SYNTH: weft.exec.runtime_param @abi_dispatch_availability_guard
    // SYNTH-SAME: abi_role = "dispatch-availability-guard"
    // SYNTH: weft.exec.dispatch
    // SYNTH: weft.exec.case @fast_runtime_path
    // SYNTH-SAME: policy = "capability_dispatch_guard"
    // SYNTH-SAME: runtime_guard = @abi_dispatch_availability_guard
    // SYNTH-SAME: runtime_guard_required = true
    // SYNTH: weft.exec.fallback @baseline_path
    // SYNTH-NOT: runtime_guard
    // SYNTH: }
  }
}
