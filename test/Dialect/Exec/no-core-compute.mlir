// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

// CHECK: custom op 'tcrv.exec.matmul' is unknown
tcrv.exec.matmul @not_a_core_op
