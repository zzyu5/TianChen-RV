// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @empty_registry {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV plugin variant materialization for kernel @empty_registry requires at least one enabled extension plugin in the registry
