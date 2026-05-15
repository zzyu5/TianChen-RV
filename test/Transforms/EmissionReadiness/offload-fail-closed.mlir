// RUN: not tcrv-opt %s --split-input-file --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=CHECK

module {
  tcrv.exec.kernel @offload_selected_missing_boundary {
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
    tcrv.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
  }
}

// CHECK: selected lowering-boundary validation failed before plugin emission routing
// CHECK-SAME: selected path @offload_runtime_first_slice as direct variant requires one materialized plugin lowering boundary

// -----

module {
  tcrv.exec.kernel @offload_selected_missing_runtime_abi {
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
      tcrv_offload.handoff_kind = "runtime-offload"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
    tcrv_offload.lowering_boundary {
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

// CHECK: selected runtime-offload variant @offload_runtime_first_slice failed plugin legality before emission planning
// CHECK-SAME: tcrv_offload.runtime_abi

// -----

module {
  tcrv.exec.kernel @offload_custom_isa_misclassification {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "custom-isa",
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
    tcrv.exec.diagnostic {
      message = "static variant selected by generic cost and capability planning",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @offload_runtime_first_slice
    }
    tcrv_offload.lowering_boundary {
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

// CHECK: selected runtime-offload variant @offload_runtime_first_slice failed plugin legality before emission planning
// CHECK-SAME: kind must be 'runtime-offload'

// -----

module {
  tcrv.exec.kernel @unknown_offload_origin_generic_registry_failure {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.variant @foreign_offload attributes {
      origin = "offload-unregistered-plugin",
      requires = [@offload_runtime]
    } {
    }
  }
}

// CHECK: TianChen-RV variant emission plan collection failed
// CHECK-SAME: variant @foreign_offload
// CHECK: unknown origin plugin 'offload-unregistered-plugin'
