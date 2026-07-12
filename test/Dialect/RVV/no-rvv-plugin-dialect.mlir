// RUN: not weft-opt %s --weft-disable-builtin-plugins 2>&1 | FileCheck %s

// CHECK: type created with unregistered dialect
// CHECK: weft_rvv
module {
  %token = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vl
}
