// RUN: not weft-opt %s --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --implicit-check-not=weft_toy.compute_skeleton

module {
  func.func @toy_source() attributes {weft_toy.lowering_seed = "template_compute"} {
    return
  }
}

// CHECK: Weft-RV execution plan coherence check failed for kernel <missing>: requires at least one weft.exec.kernel
