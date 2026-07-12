// RUN: not weft-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'weft.exec.matmul' is unknown
weft.exec.matmul @not_a_core_op
