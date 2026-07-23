// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-plugin-variants

module {
  // expected-error @+1 {{Weft-RV plugin variant materialization for source kernel @missing_domain requires non-empty string attribute 'construction_domain'}}
  weft.exec.kernel @missing_domain attributes {problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// -----

module {
  // expected-error @+1 {{Weft-RV plugin variant materialization for source kernel @unknown_domain declares construction domain 'unknown-domain', but no enabled extension plugin declares that domain}}
  weft.exec.kernel @unknown_domain attributes {construction_domain = "unknown-domain", problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
