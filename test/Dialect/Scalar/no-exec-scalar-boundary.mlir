// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'tcrv.exec.scalar_lowering_boundary' is unknown
tcrv.exec.scalar_lowering_boundary {
  origin = "scalar-plugin",
  role = "direct variant",
  status = "no-active-route"
}
