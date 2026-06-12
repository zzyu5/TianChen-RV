// RUN: not tcrv-opt %s --split-input-file --tcrv-synthesize-variant-dispatch 2>&1 | FileCheck %s

// CHECK: error: cannot synthesize tcrv.exec.dispatch for kernel @no_available_fallback: no direct variant is conflict-free and generically available as dispatch fallback under the kernel capability set
tcrv.exec.kernel @no_available_fallback attributes {} {
  tcrv.exec.capability @missing_runtime {
    id = "generic.missing.runtime",
    kind = "runtime",
    status = "missing"
  }
  tcrv.exec.capability @disabled_toolchain {
    id = "generic.disabled.toolchain",
    kind = "toolchain",
    status = "disabled"
  }
  tcrv.exec.variant @missing_path attributes {
    origin = "missing-plugin",
    requires = [@missing_runtime]
  } {
  }
  tcrv.exec.variant @disabled_path attributes {
    origin = "disabled-plugin",
    requires = [@disabled_toolchain]
  } {
  }
}

// -----

// CHECK: error: cannot synthesize tcrv.exec.dispatch for kernel @single_unavailable: no direct variant is conflict-free and generically available as dispatch fallback under the kernel capability set
tcrv.exec.kernel @single_unavailable attributes {} {
  tcrv.exec.capability @missing_runtime {
    id = "generic.missing.runtime",
    kind = "runtime",
    status = "missing"
  }
  tcrv.exec.variant @missing_path attributes {
    origin = "missing-plugin",
    requires = [@missing_runtime]
  } {
  }
}

// -----

// CHECK: error: cannot synthesize tcrv.exec.dispatch for kernel @no_conflict_free_fallback: no direct variant is conflict-free and generically available as dispatch fallback under the kernel capability set
tcrv.exec.kernel @no_conflict_free_fallback attributes {} {
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
  tcrv.exec.capability @baseline_conflict {
    id = "generic.baseline",
    kind = "toolchain",
    relations = #tcrv.capability_relations<conflicts = ["build.policy.disable_baseline"]>,
    status = "available"
  }
  tcrv.exec.capability @disable_baseline_profile {
    id = "generic.baseline.profile",
    kind = "build-policy",
    relations = #tcrv.capability_relations<provides = ["build.policy.disable_baseline"]>,
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
    requires = [@baseline_conflict]
  } {
  }
}

// -----

// CHECK: error: 'tcrv.exec.variant' op attribute 'requires' must contain only capability symbol references
tcrv.exec.kernel @malformed_requires attributes {} {
  tcrv.exec.capability @baseline_capability {
    id = "generic.baseline",
    kind = "toolchain"
  }
  tcrv.exec.capability @extra_capability {
    id = "generic.extra",
    kind = "toolchain"
  }
  tcrv.exec.variant @bad_path attributes {
    origin = "bad-plugin",
    requires = ["not_a_symbol"]
  } {
  }
  tcrv.exec.variant @baseline_path attributes {
    origin = "baseline-plugin",
    requires = [@baseline_capability]
  } {
  }
}
