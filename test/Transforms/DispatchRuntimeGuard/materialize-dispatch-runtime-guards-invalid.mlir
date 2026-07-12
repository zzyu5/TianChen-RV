// RUN: not weft-opt %s --weft-materialize-dispatch-runtime-guards 2>&1 | FileCheck %s --check-prefix=COLLISION

module {
  weft.exec.kernel @runtime_guard_symbol_collision {
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
    weft.exec.mem_window @abi_dispatch_availability_guard {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
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
  }
}

// COLLISION: direct symbol @abi_dispatch_availability_guard already exists and cannot be reused
// COLLISION-SAME: dispatch-availability-guard
