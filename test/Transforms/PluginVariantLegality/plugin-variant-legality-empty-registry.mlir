// RUN: weft-opt %s --weft-disable-builtin-plugins --verify-diagnostics --weft-verify-plugin-variant-legality

module {
  // expected-error@+1 {{Weft-RV variant legality verification failed for variant @scalar_fallback_first_slice in kernel @empty_registry_scalar: unknown origin plugin 'scalar-plugin'}}
  weft.exec.kernel @empty_registry_scalar {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}
