// RUN: tcrv-opt %s --tcrv-disable-builtin-plugins --verify-diagnostics --tcrv-verify-plugin-variant-legality

module {
  // expected-error@+1 {{TianChen-RV variant legality verification failed for variant @scalar_fallback_first_slice in kernel @empty_registry_scalar: unknown origin plugin 'scalar-plugin'}}
  tcrv.exec.kernel @empty_registry_scalar {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}
