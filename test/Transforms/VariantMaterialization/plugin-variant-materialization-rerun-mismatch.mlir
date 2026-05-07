// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rerun_mismatch {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      policy = "hand_authored_wrong_policy",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// CHECK: error: TianChen-RV variant materialization failed for proposal 'scalar_fallback_first_slice' from origin plugin 'scalar-plugin': existing direct variant @scalar_fallback_first_slice does not exactly match the current plugin proposal: policy attribute differs
