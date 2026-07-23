// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --verify-diagnostics

// P-projection fail-closed (I7) for sliding-window semantics. These complement the
// op-VERIFIER negatives in test/Dialect/IME/mma-slide.mlir: here the rejection
// happens before proposal, so an out-of-envelope sliding problem never reaches a
// materialized boundary op. Slide is an exact-P geometry fact, with a CLOSED envelope:
// only signed slide 1/2/3 over the single-fragment MAC is modeled.

// Sliding MAC is signed-only: the unsigned/mixed-sign siblings fail closed.
module {
  // expected-error@+1 {{sliding int8 MAC is only modeled for signed x signed input}}
  weft.exec.kernel @ime_slide_unsigned attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_sliding_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness unsigned>, rhs_signedness = #weft<integer_signedness unsigned>, m = 4 : i64, n = 4 : i64, k = 8 : i64, slide = 1 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
  }
}

// -----

// Sliding construction is a single target fragment only; tiled sliding fails closed.
module {
  // expected-error@+1 {{sliding int8 MAC geometry must equal the target-projected MAC fragment}}
  weft.exec.kernel @ime_slide_matmul attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_sliding_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 8 : i64, n = 8 : i64, k = 16 : i64, slide = 1 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
  }
}

// -----

// The slide envelope is CLOSED to {1,2,3}: a slide value outside it (here 9) is
// not the documented vmadot1/2/3 family and fails in the canonical P verifier.
module {
  weft.exec.kernel @ime_slide_bad attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    // expected-error@+1 {{requires bounded source slide geometry in {1,2,3}; got 9}}
    weft.exec.int8_sliding_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64, slide = 9 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
  }
}
