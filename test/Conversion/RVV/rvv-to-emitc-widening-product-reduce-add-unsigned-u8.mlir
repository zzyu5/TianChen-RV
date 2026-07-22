// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — UNSIGNED widening-product-reduce-add. The typed body chains an
// unsigned widening product (ui8/mf4 x ui8/mf4 -> ui16/mf2 via vwmulu) directly
// into an unsigned widening reduction (ui16/mf2 -> ui32/m1 via vwredsumu),
// carrying the scalar accumulator through the output cell (out[0] seeded from
// acc[0] pre-loop, read-modify-written each chunk). Both the product intrinsic
// (vwmulu) and the reduction intrinsic (vwredsumu) plus the unsigned seed splat
// (vmv_v_x_u32m1) are keyed on the unsigned source signedness from the typed
// body. Structure-level CHECKs; byte identity to the legacy oracle is pinned by
// the Target/RVV artifact fixture.

module {
  weft.exec.kernel @rvv_unsigned_product_reduce_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_unsigned_product_reduce attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "unsigned-product-reduce:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "unsigned-product-reduce:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "unsigned-product-reduce:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "uint32_t *", ownership = "target-export-abi-owned", purpose = "unsigned-product-reduce:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "unsigned-product-reduce:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<ui16, "mf2">
        %reduced = weft_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "unsigned_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<ui16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<ui32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_unsigned_product_reduce_kernel_rvv_unsigned_product_reduce(
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop u32 seed: out[0] = acc[0] (acc is const uint32_t*).
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_u32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_u32m1"(%arg3,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The ui8/mf4 source loads land vuint8mf4_t (unsigned rung).
// CHECK: %[[LHS:.*]] = call_opaque "__riscv_vle8_v_u8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf4_t">
// CHECK: %[[RHS:.*]] = call_opaque "__riscv_vle8_v_u8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf4_t">
// The unsigned widened product: ui8/mf4 sources -> ui16/mf2 result via vwmulu.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmulu_vv_u16mf2"(%[[LHS]], %[[RHS]], %[[BODYVL]]) : (!emitc.opaque<"vuint8mf4_t">, !emitc.opaque<"vuint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16mf2_t">
// In-loop u32 running seed.
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_u32m1"
// The unsigned widening reduce: ui16mf2 product -> ui32m1 accumulator via vwredsumu.
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsumu_vs_u16mf2_u32m1"(%[[PROD]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vuint16mf2_t">, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_u32m1"(%arg3, %[[RED]],
// CHECK: return
