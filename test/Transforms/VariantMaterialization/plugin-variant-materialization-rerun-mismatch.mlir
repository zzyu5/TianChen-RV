// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rerun_mismatch attributes {problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      policy = "hand_authored_wrong_policy",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// CHECK: error: Weft-RV variant materialization failed for proposal 'scalar_fallback_first_slice' from origin plugin 'scalar-plugin': existing direct variant @scalar_fallback_first_slice does not exactly match the current plugin proposal: policy attribute differs
