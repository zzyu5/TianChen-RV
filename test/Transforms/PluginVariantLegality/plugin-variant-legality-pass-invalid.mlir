// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-verify-plugin-variant-legality

module {
  // expected-error@+1 {{TianChen-RV variant legality verification failed for variant @unknown_path in kernel @legality_unknown_origin: unknown origin plugin 'missing-plugin'}}
  tcrv.exec.kernel @legality_unknown_origin {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @unknown_path attributes {
      origin = "missing-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// -----

module {
  // expected-error@+1 {{metadata-only RVV first-slice route has been deleted}}
  tcrv.exec.kernel @legality_rvv_deleted_metadata_route {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_deleted_metadata_path attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
