// RUN: not tcrv-opt %s --tcrv-source-seed-artifact-front-door-pipeline 2>&1 | FileCheck %s --implicit-check-not=tcrv_toy.compute_skeleton

module {
  func.func @toy_seed() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// CHECK: TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel
