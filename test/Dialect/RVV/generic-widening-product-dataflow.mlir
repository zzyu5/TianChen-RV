// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_widening_product_dataflow_valid
  tcrv.exec.kernel @rvv_generic_widening_product_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i8, "mf4">
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i8, "mf4">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
      // CHECK: tcrv_rvv.widening_product
      // CHECK-SAME: kind = "signed_widening_product"
      // CHECK-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      // CHECK: tcrv_rvv.store
      // CHECK-SAME: !tcrv_rvv.vector<i16, "mf2">
      tcrv_rvv.store %out_ptr, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_product_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{currently supports only kind "signed_widening_product" or "unsigned_widening_product" for the bounded Stage 2 low-precision widening-product typed surface}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "metadata_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_product_reject_relation {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{currently supports only product_relation "signed-i8mf4xi8mf4-to-i16mf2", "unsigned-u8mf4xu8mf4-to-u16mf2", or "signed-i8m2xi8m2-to-i16m4" (the deferred-wide max-legal-LMUL rung) for the bounded Stage 2 low-precision widening-product typed surface}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i32m1"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_widening_product_unsigned_u8_surface_valid
  tcrv.exec.kernel @rvv_generic_widening_product_unsigned_u8_surface_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<ui8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<ui8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // CHECK: tcrv_rvv.widening_product
      // CHECK-SAME: kind = "unsigned_widening_product"
      // CHECK-SAME: product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui16, "mf2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_product_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{requires lhs and rhs source vectors to have type !tcrv_rvv.vector<i8, "mf4"> for the bounded signed low-precision widening-product route}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_product_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{requires result vector to have type !tcrv_rvv.vector<i16, "mf2"> for the bounded signed low-precision widening-product route}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_product_reject_result_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i8, "mf4">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires enclosing tcrv_rvv.with_vl result config to be SEW16 LMUL mf2 for the bounded signed low-precision widening-product route}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
    } : !tcrv_rvv.vl
  }
}
