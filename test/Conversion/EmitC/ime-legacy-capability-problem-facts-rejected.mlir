// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --verify-diagnostics

// Source-problem facts belong only to exact kernel.problem. Their legacy
// presence on the target capability fails closed, including an empty value.
module {
  // expected-error@+1 {{must not carry source-problem property 'ime_signedness'}}
  weft.exec.kernel @legacy_signedness attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_signedness = "unsigned"}
  }
}

// -----

module {
  // expected-error@+1 {{must not carry source-problem property 'ime_matmul_shape'}}
  weft.exec.kernel @legacy_shape attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_matmul_shape = ""}
  }
}

// -----

module {
  // expected-error@+1 {{must not carry source-problem property 'ime_weight_format'}}
  weft.exec.kernel @legacy_format attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_weight_format = "q4_0"}
  }
}

// -----

module {
  // expected-error@+1 {{must not carry source-problem property 'ime_slide'}}
  weft.exec.kernel @legacy_slide attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256", ime_slide = "1"}
  }
}
