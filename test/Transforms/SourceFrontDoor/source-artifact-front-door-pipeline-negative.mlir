// RUN: not tcrv-opt %s --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --implicit-check-not="rvv-generic-binary-add-emitc-route" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""

// The default source-artifact pipeline no longer runs the RVV source
// recognizer. Stale RVV source metadata therefore cannot become selected-body
// or target artifact authority.

module {
  func.func @stale_rvv_marker(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    return
  }
}

// CHECK: TianChen-RV execution plan coherence check failed for kernel <missing>
// CHECK-SAME: requires at least one tcrv.exec.kernel
