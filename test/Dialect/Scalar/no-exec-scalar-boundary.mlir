// RUN: not weft-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'weft.exec.scalar_lowering_boundary' is unknown
weft.exec.scalar_lowering_boundary {
  origin = "scalar-plugin",
  role = "direct variant",
  status = "no-active-route"
}
