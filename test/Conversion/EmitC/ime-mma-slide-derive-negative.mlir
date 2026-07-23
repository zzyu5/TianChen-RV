// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --verify-diagnostics

// DERIVE-level fail-closed (I7) for the sliding-window FACT. These complement the
// op-VERIFIER negatives in test/Dialect/IME/mma-slide.mlir: here the rejection
// happens earlier, in deriveIMEMatmulCapability, so an out-of-envelope ime_slide
// request never even reaches a materialized boundary op. The slide is a SHAPE/
// window fact of the SAME spacemit.ime capability, with a CLOSED envelope:
// only signed slide 1/2/3 over the single-fragment MAC is modeled.

// ime_slide is signed-only: the unsigned/mixed-sign slide siblings have no
// emitter, so requesting both fails closed at derive time.
module {
  // expected-error@+1 {{property 'ime_slide' (sliding-window) is only modeled for the signed form (vmadot1/2/3); the unsigned/mixed-sign slide siblings have no emitter}}
  weft.exec.kernel @ime_slide_unsigned attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness unsigned>, rhs_signedness = #weft<integer_signedness unsigned>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_signedness = "unsigned",
      ime_slide = "1"
    }
  }
}

// -----

// ime_slide is the single-fragment boundary only: it is NOT modeled together with
// the tiled whole-matrix shape, so requesting both fails closed at derive time.
module {
  // expected-error@+1 {{property 'ime_slide' (sliding-window) is not modeled together with the tiled whole-matrix shape 'ime_matmul_shape'; the slide boundary is the single-fragment weft.ime.mma_slide only}}
  weft.exec.kernel @ime_slide_matmul attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 8 : i64, n = 8 : i64, k = 16 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "8x8x16",
      ime_slide = "1"
    }
  }
}

// -----

// The slide envelope is CLOSED to {1,2,3}: a slide value outside it (here 9) is
// not the documented vmadot1/2/3 family and fails closed at derive time.
module {
  // expected-error@+1 {{property 'ime_slide' = '9' is outside the validated IME1 slide envelope (only '1' => vmadot1, '2' => vmadot2, '3' => vmadot3 are modeled)}}
  weft.exec.kernel @ime_slide_bad attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_slide = "9"
    }
  }
}
