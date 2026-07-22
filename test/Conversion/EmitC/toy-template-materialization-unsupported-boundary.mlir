// RUN: not weft-opt %s 2>&1 | FileCheck %s

// The metadata-only boundary surface is retired. Formula construction must
// produce weft_toy.compute_skeleton directly; no route may revive this op.
module {
  weft_toy.lowering_boundary
}

// CHECK: custom op 'weft_toy.lowering_boundary' is unknown
