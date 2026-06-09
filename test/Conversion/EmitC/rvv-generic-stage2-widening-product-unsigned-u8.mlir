// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC

module {
  tcrv.exec.kernel @rvv_unsigned_u8_widening_product_route {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_unsigned_u8_widening_product attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const uint8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "uint16_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_unsigned_u8_widening_product, sew = 16 : i64, source_kernel = "rvv_unsigned_u8_widening_product_route", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "mf4">
        %product = tcrv_rvv.widening_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_widening_product", product_relation = "unsigned-u8mf4xu8mf4-to-u16mf2"} : !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vector<ui8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui16, "mf2">
        tcrv_rvv.store %out, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<ui16, "mf2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_product_u8_u16.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:unsigned-e8mf4,result:unsigned-e16mf2,mask:b32"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_relation", value = "unsigned-u8mf4xu8mf4-to-u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.widening_product_intrinsic", value = "__riscv_vwmulu_vv_u16mf2"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.kind", value = "unsigned-u8mf4xu8mf4-to-u16mf2-widening-product.v1"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_dtype", value = "u8"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.source_signedness", value = "unsigned"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.product_dtype", value = "u16"}
// PLAN-SAME: {key = "tcrv_rvv.low_precision_primitive.result_dtype", value = "u16"}
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const uint8_t *"
// PLAN-SAME: {c_name = "out", c_type = "uint16_t *"
// PLAN-SAME: status = "supported"

// EMITC: emitc.include <"riscv_vector.h">
// EMITC: emitc.func @tcrv_emitc_rvv_unsigned_u8_widening_product_route_rvv_unsigned_u8_widening_product
// EMITC-SAME: !emitc.ptr<!emitc.opaque<"const uint8_t">>
// EMITC-SAME: !emitc.ptr<!emitc.opaque<"uint16_t">>
// EMITC: call_opaque "__riscv_vle8_v_u8mf4"
// EMITC-SAME: !emitc.opaque<"vuint8mf4_t">
// EMITC: call_opaque "__riscv_vwmulu_vv_u16mf2"
// EMITC-SAME: !emitc.opaque<"vuint16mf2_t">
// EMITC: call_opaque "__riscv_vse16_v_u16mf2"
// EMITC-SAME: !emitc.opaque<"uint16_t">
