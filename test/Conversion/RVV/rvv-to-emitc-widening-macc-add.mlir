// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening multiply-accumulate. The fused
// weft_rvv.widening_macc widens two i16/mf2 source multiplicands and accumulates
// into the i32/m1 accumulator vector via __riscv_vwmacc_vv_i32m1. Like the plain
// vmacc, the C call order is (accumulator, lhs, rhs, vl) -- vwmacc is a 3-read
// fused op that read-modify-writes the accumulator, so the accumulator vector is
// the FIRST argument. The intrinsic dtype/lmul derive from the RESULT (i32/m1).
// Structure-level CHECKs; byte identity is pinned by the Target/RVV fixture.

module {
  weft.exec.kernel @rvv_widening_macc_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_widening_macc attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc_ptr = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-macc:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_widening_macc, sew = 32 : i64, source_kernel = "rvv_widening_macc_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %acc = weft_rvv.load %acc_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.widening_macc %a, %b, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_widening_macc_kernel_rvv_widening_macc(
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// i16/mf2 lhs/rhs multiplicands + i32/m1 accumulator vector loads.
// CHECK: %[[LHS:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// CHECK: %[[RHS:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// CHECK: %[[ACC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// vwmacc: accumulator FIRST, then lhs, rhs, vl (the result type drives the intrinsic).
// CHECK: %[[SUM:.*]] = call_opaque "__riscv_vwmacc_vv_i32m1"(%[[ACC]], %[[LHS]], %[[RHS]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint16mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[SUM]], %[[BODYVL]])
// CHECK: return
