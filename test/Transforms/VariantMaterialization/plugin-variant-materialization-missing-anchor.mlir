// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @missing_capability_anchor {
  }
}

// CHECK: error: Weft-RV plugin variant materialization for kernel @missing_capability_anchor requires at least one capability provider in the kernel capability scope
