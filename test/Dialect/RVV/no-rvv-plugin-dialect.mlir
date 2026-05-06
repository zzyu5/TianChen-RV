// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

// CHECK: type created with unregistered dialect
// CHECK: tcrv_rvv
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vl
}
