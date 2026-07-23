// RUN: weft-opt %s --split-input-file --weft-verify-plugin-variant-legality --verify-diagnostics

// Variant metadata must not mirror facts already owned by exact P.
module {
  // expected-error@+1 {{must not mirror canonical problem fact 'ime.signedness'}}
  weft.exec.kernel @legacy_variant_signedness attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256"}
    weft.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime], ime.signedness = "signed"} {}
  }
}

// -----

module {
  // expected-error@+1 {{must not mirror canonical problem fact 'ime.slide'}}
  weft.exec.kernel @legacy_variant_slide attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256"}
    weft.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime], ime.slide = "1"} {}
  }
}

// -----

module {
  // expected-error@+1 {{must not mirror canonical problem fact 'ime.weight_format'}}
  weft.exec.kernel @legacy_variant_format attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available", march = "rv64gcv_xsmtvdotii", vlen_bits = "256"}
    weft.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime], ime.weight_format = "q4_0"} {}
  }
}
