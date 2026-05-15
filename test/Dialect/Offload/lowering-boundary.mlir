// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @offload_boundary_valid
  tcrv.exec.kernel @offload_boundary_valid {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // CHECK: tcrv_offload.lowering_boundary
    // CHECK-SAME: handoff_kind = "runtime-offload"
    // CHECK-SAME: origin = "offload-plugin"
    // CHECK-SAME: required_capabilities = [@offload_runtime]
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: runtime_abi = "generic-runtime-offload-c-abi-handoff.v1"
    // CHECK-SAME: selected_variant = @offload_runtime_first_slice
    // CHECK-SAME: source_kernel = "offload_boundary_valid"
    // CHECK-SAME: status = "no-active-route"
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      handoff_reason = "runtime-offload boundary records no active route",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "dispatch case",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_valid",
      status = "no-active-route"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_wrong_status {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{status must be 'no-active-route'}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_wrong_status",
      status = "supported"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_wrong_handoff_kind {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{handoff_kind must be 'runtime-offload'}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "custom-riscv-isa",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_wrong_handoff_kind",
      status = "no-active-route"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_fallback_role {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{role must be 'direct variant' or 'dispatch case'}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "dispatch fallback",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_fallback_role",
      status = "no-active-route"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_missing_runtime_abi {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{requires attribute 'runtime_abi'}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_missing_runtime_abi",
      status = "no-active-route"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_wrong_required_capabilities {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @portable {id = "portable", kind = "fallback"}
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{required_capabilities must match selected variant requires metadata}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@portable],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_wrong_required_capabilities",
      status = "no-active-route"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @offload_boundary_executable_claim {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      tcrv_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    // expected-error@+1 {{handoff_reason must not claim executable offload runtime}}
    tcrv_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      handoff_reason = "hardware execution and performance evidence produced",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_boundary_executable_claim",
      status = "no-active-route"
    }
  }
}
