// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-plugin-variants

module {
  // expected-error @+1 {{Weft-RV source target/domain binding for kernel @missing_domain requires an explicit module-level target/profile reference}}
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
  weft.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }
  weft.exec.target @unknown_profile {id = "unknown.profile", target_kind = "profile", construction_domain = "unknown-domain", capability_providers = [@scalar_fallback]}
  // expected-error @+1 {{Weft-RV plugin variant materialization for source kernel @unknown_domain declares construction domain 'unknown-domain', but no enabled extension plugin declares that domain}}
  weft.exec.kernel @unknown_domain attributes {target = @unknown_profile, problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}
