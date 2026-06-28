// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening multiply-accumulate. The fused
// tcrv_rvv.widening_macc widens two i16/mf2 source multiplicands and accumulates
// into the i32/m1 accumulator vector via __riscv_vwmacc_vv_i32m1. Like the plain
// vmacc, the C call order is (accumulator, lhs, rhs, vl) -- vwmacc is a 3-read
// fused op that read-modify-writes the accumulator, so the accumulator vector is
// the FIRST argument. The intrinsic dtype/lmul derive from the RESULT (i32/m1).
// Structure-level CHECKs; byte identity is pinned by the Target/RVV fixture.

module {
  tcrv.exec.kernel @rvv_widening_macc_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_widening_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "widening-macc:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-macc:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_widening_macc, sew = 32 : i64, source_kernel = "rvv_widening_macc_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %acc = tcrv_rvv.load %acc_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.widening_macc %a, %b, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_widening_macc_kernel_rvv_widening_macc(
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
