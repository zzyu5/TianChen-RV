// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --verify-diagnostics

// Source-problem facts belong only to exact kernel.problem. Their legacy
// presence on the target capability fails closed, including an empty value.
module {
  weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_signedness = "unsigned"}
  weft.exec.target @legacy_signedness_profile {id = "legacy.signedness.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  // expected-error@+1 {{must not carry source-problem property 'ime_signedness'}}
  weft.exec.kernel @legacy_signedness attributes {target = @legacy_signedness_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// -----

module {
  weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_matmul_shape = ""}
  weft.exec.target @legacy_shape_profile {id = "legacy.shape.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  // expected-error@+1 {{must not carry source-problem property 'ime_matmul_shape'}}
  weft.exec.kernel @legacy_shape attributes {target = @legacy_shape_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// -----

module {
  weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_weight_format = "q4_0"}
  weft.exec.target @legacy_format_profile {id = "legacy.format.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  // expected-error@+1 {{must not carry source-problem property 'ime_weight_format'}}
  weft.exec.kernel @legacy_format attributes {target = @legacy_format_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// -----

module {
  weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_slide = "1"}
  weft.exec.target @legacy_slide_profile {id = "legacy.slide.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  // expected-error@+1 {{must not carry source-problem property 'ime_slide'}}
  weft.exec.kernel @legacy_slide attributes {target = @legacy_slide_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}
