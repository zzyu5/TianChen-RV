// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_widening_product_dataflow_valid
  weft.exec.kernel @rvv_generic_widening_product_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i8, "mf4">
      %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i8, "mf4">
      %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
      // CHECK: weft_rvv.widening_product
      // CHECK-SAME: kind = "signed_widening_product"
      // CHECK-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      // CHECK: weft_rvv.store
      // CHECK-SAME: !weft_rvv.vector<i16, "mf2">
      weft_rvv.store %out_ptr, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_product_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{currently supports only kind "signed_widening_product" or "unsigned_widening_product" for the bounded Stage 2 low-precision widening-product typed surface}}
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "metadata_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_product_reject_relation {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{currently supports only product_relation "signed-i8mf4xi8mf4-to-i16mf2", "unsigned-u8mf4xu8mf4-to-u16mf2", or "signed-i8m2xi8m2-to-i16m4" (the deferred-wide max-legal-LMUL rung) for the bounded Stage 2 low-precision widening-product typed surface}}
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i32m1"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_widening_product_unsigned_u8_surface_valid
  weft.exec.kernel @rvv_generic_widening_product_unsigned_u8_surface_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<ui8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<ui8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // CHECK: weft_rvv.widening_product
      // CHECK-SAME: kind = "unsigned_widening_product"
      // CHECK-SAME: product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<ui16, "mf2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_product_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{requires lhs and rhs source vectors to have type !weft_rvv.vector<i8, "mf4"> for the bounded signed low-precision widening-product route}}
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_product_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{requires result vector to have type !weft_rvv.vector<i16, "mf2"> for the bounded signed low-precision widening-product route}}
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_product_reject_result_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i8, "mf4">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires enclosing weft_rvv.with_vl result config to be SEW16 LMUL mf2 for the bounded signed low-precision widening-product route}}
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
    } : !weft_rvv.vl
  }
}
