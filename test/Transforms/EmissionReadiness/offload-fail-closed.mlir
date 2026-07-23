// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-emission-plans | FileCheck %s --check-prefix=CHECK

module {
  weft.exec.kernel @offload_selected_missing_boundary {
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      weft_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      weft_offload.handoff_kind = "runtime-offload"
    } {
    }
    weft.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
  }
}

// CHECK-LABEL: weft.exec.kernel @offload_selected_missing_boundary
// CHECK: weft_offload.lowering_boundary
// CHECK-SAME: handoff_reason = "family-constructed delegation plan; no executable external implementation is currently bound"
// CHECK-SAME: selected_variant = @offload_runtime_first_slice
// CHECK-SAME: status = "no-active-route"
// CHECK: weft.exec.diagnostic
// CHECK-SAME: message = "the Offload extension currently has no active executable lowering or target artifact route"
// CHECK-SAME: status = "unsupported"

// -----

module {
  // expected-error@+1 {{selected owner construction for origin 'offload-plugin' rejected selected variant legality: Weft-RV runtime-offload extension plugin first slice failed: materialized runtime-offload variant @offload_runtime_first_slice requires non-empty string 'weft_offload.runtime_abi' metadata}}
  weft.exec.kernel @offload_selected_missing_runtime_abi {
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      weft_offload.handoff_kind = "runtime-offload"
    } {
    }
    weft.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
    weft_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_selected_missing_runtime_abi",
      status = "no-active-route"
    }
  }
}

// -----

module {
  // expected-error@+1 {{selected owner construction for origin 'offload-plugin' rejected selected variant legality: Weft-RV runtime-offload extension plugin first slice failed: capability id 'offload.runtime' kind must be 'runtime-offload'}}
  weft.exec.kernel @offload_custom_isa_misclassification {
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.variant @offload_runtime_first_slice attributes {
      origin = "offload-plugin",
      requires = [@offload_runtime],
      weft_offload.runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      weft_offload.handoff_kind = "runtime-offload"
    } {
    }
    weft.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
    weft_offload.lowering_boundary {
      handoff_kind = "runtime-offload",
      origin = "offload-plugin",
      required_capabilities = [@offload_runtime],
      role = "direct variant",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      selected_variant = @offload_runtime_first_slice,
      source_kernel = "offload_custom_isa_misclassification",
      status = "no-active-route"
    }
  }
}

// -----

module {
  // expected-error@+1 {{selected owner construction cannot resolve unknown origin 'offload-unregistered-plugin'}}
  weft.exec.kernel @unknown_offload_origin_generic_registry_failure {
    weft.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    weft.exec.variant @foreign_offload attributes {
      origin = "offload-unregistered-plugin",
      requires = [@offload_runtime]
    } {
    }
  }
}
