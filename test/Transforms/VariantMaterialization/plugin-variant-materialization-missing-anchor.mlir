// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @missing_capability_anchor {
  }
}

// CHECK: error: TianChen-RV plugin variant materialization for kernel @missing_capability_anchor requires at least one capability provider in the kernel capability scope
