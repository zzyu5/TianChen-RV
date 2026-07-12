// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-verify-plugin-variant-legality

module {
  // expected-error@+1 {{Weft-RV variant legality verification failed for variant @unknown_path in kernel @legality_unknown_origin: unknown origin plugin 'missing-plugin'}}
  weft.exec.kernel @legality_unknown_origin {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @unknown_path attributes {
      origin = "missing-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// -----

module {
  // expected-error@+1 {{materialized RVV variant requires explicit typed RVV extension-family body}}
  weft.exec.kernel @legality_rvv_missing_typed_body {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @rvv_missing_typed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
