// RUN: not weft-opt %s --weft-disable-builtin-plugins --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @empty_registry {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: error: Weft-RV plugin variant materialization for kernel @empty_registry requires at least one enabled extension plugin in the registry
