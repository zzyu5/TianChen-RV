// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening product (the low-precision contraction
// beachhead). The typed weft_rvv.widening_product widens two FRACTIONAL-LMUL i8
// source multiplicands (i8/mf4 -> vint8mf4_t, the new TypeConverter rung) into a
// one-step-wider i16/mf2 product via __riscv_vwmul_vv_i16mf2. The vwmul
// dtype/lmul derive from the RESULT vector type (i16/mf2), NOT the source.
// Structure-level CHECKs; byte identity to the legacy oracle is pinned by the
// Target/RVV artifact fixture.

module {
  weft.exec.kernel @rvv_widening_product_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-product:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-product:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int16_t *", ownership = "target-export-abi-owned", purpose = "widening-product:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-product:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_widening_product_kernel_rvv_widening_product(%arg0: !emitc.ptr<!emitc.opaque<"const int8_t">>, %arg1: !emitc.ptr<!emitc.opaque<"const int8_t">>, %arg2: !emitc.ptr<!emitc.opaque<"int16_t">>, %arg3: !emitc.opaque<"size_t">)
// CHECK: call_opaque "__riscv_vsetvl_e16mf2"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e16mf2"
// The fractional-LMUL i8 source loads land vint8mf4_t (the new TypeConverter rung).
// CHECK: %[[LHS:.*]] = call_opaque "__riscv_vle8_v_i8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf4_t">
// CHECK: %[[RHS:.*]] = call_opaque "__riscv_vle8_v_i8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint8mf4_t">
// The widened product: i8/mf4 sources -> i16/mf2 result (intrinsic from the RESULT type).
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16mf2"(%[[LHS]], %[[RHS]], %[[BODYVL]]) : (!emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// CHECK: call_opaque "__riscv_vse16_v_i16mf2"(%{{.*}}, %[[PROD]], %[[BODYVL]])
// CHECK: return
