// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC

module {
  weft.exec.kernel @rvv_unsigned_u8_widening_product_route {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_unsigned_u8_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "uint16_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "mf4">
        %product = weft_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vector<ui8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<ui16, "mf2">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<ui16, "mf2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const uint8_t *"
// PLAN-SAME: {c_name = "out", c_type = "uint16_t *"
// PLAN-SAME: status = "supported"

// EMITC: emitc.include <"riscv_vector.h">
// EMITC: emitc.func @weft_emitc_rvv_unsigned_u8_widening_product_route_rvv_unsigned_u8_widening_product
// EMITC-SAME: !emitc.ptr<!emitc.opaque<"const uint8_t">>
// EMITC-SAME: !emitc.ptr<!emitc.opaque<"uint16_t">>
// EMITC: call_opaque "__riscv_vle8_v_u8mf4"
// EMITC-SAME: !emitc.opaque<"vuint8mf4_t">
// EMITC: call_opaque "__riscv_vwmulu_vv_u16mf2"
// EMITC-SAME: !emitc.opaque<"vuint16mf2_t">
// EMITC: call_opaque "__riscv_vse16_v_u16mf2"
// EMITC-SAME: !emitc.opaque<"uint16_t">
