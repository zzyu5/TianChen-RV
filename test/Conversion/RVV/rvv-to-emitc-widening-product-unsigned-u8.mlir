// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — UNSIGNED low-precision widening product. The typed
// tcrv_rvv.widening_product (kind=unsigned_widening_product) widens two
// FRACTIONAL-LMUL ui8 source multiplicands (ui8/mf4 -> vuint8mf4_t, the new
// unsigned TypeConverter rung) into a one-step-wider ui16/mf2 product via
// __riscv_vwmulu_vv_u16mf2 (the vwmulu unsigned intrinsic, keyed on the source
// signedness from the typed body). Structure-level CHECKs; byte identity to the
// legacy oracle is pinned by the Target/RVV artifact fixture.

module {
  tcrv.exec.kernel @rvv_unsigned_widening_product_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_unsigned_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "unsigned-widening-product:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "unsigned-widening-product:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "uint16_t *", ownership = "target-export-abi-owned", purpose = "unsigned-widening-product:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "unsigned-widening-product:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_unsigned_widening_product, sew = 16 : i64, source_kernel = "rvv_unsigned_widening_product_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui16, "mf2">
        tcrv_rvv.store %out, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<ui16, "mf2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_unsigned_widening_product_kernel_rvv_unsigned_widening_product(%arg0: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg1: !emitc.ptr<!emitc.opaque<"const uint8_t">>, %arg2: !emitc.ptr<!emitc.opaque<"uint16_t">>, %arg3: !emitc.opaque<"size_t">)
// CHECK: call_opaque "__riscv_vsetvl_e16mf2"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e16mf2"
// The fractional-LMUL ui8 source loads land vuint8mf4_t (the unsigned rung).
// CHECK: %[[LHS:.*]] = call_opaque "__riscv_vle8_v_u8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf4_t">
// CHECK: %[[RHS:.*]] = call_opaque "__riscv_vle8_v_u8mf4"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint8mf4_t">
// The widened UNSIGNED product: ui8/mf4 sources -> ui16/mf2 result via vwmulu.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmulu_vv_u16mf2"(%[[LHS]], %[[RHS]], %[[BODYVL]]) : (!emitc.opaque<"vuint8mf4_t">, !emitc.opaque<"vuint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint16mf2_t">
// CHECK: call_opaque "__riscv_vse16_v_u16mf2"(%{{.*}}, %[[PROD]], %[[BODYVL]])
// CHECK: return
