// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening CONVERSION i32/m1 -> i64/m2. The typed
// weft_rvv.widening_convert sign-extends the loaded i32/m1 vector into a
// one-step-wider i64/m2 vector via __riscv_vwcvt_x_x_v_i64m2. The vwcvt
// dtype/lmul derive from the RESULT vector type (i64/m2), NOT the source; the
// setvl/store run on the DEST config (e64m2), the load on the SOURCE (vle32
// i32m1). Structure-level CHECKs; byte identity to the legacy oracle is pinned
// by the Target/RVV explicit/pre-realized widen-i32-to-i64 artifact fixtures.

module {
  weft.exec.kernel @rvv_widen_i32_to_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_widen_i32_to_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widen-i32-to-i64:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "widen-i32-to-i64:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widen-i32-to-i64:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
        weft_rvv.store %out, %widened, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_widen_i32_to_i64_kernel_rvv_widen_i32_to_i64(%arg0: !emitc.ptr<!emitc.opaque<"const int32_t">>, %arg1: !emitc.ptr<!emitc.opaque<"int64_t">>, %arg2: !emitc.opaque<"size_t">)
// The setvl/loop run on the DEST config (e64m2), not the narrow source.
// CHECK: call_opaque "__riscv_vsetvl_e64m2"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e64m2"
// The i32/m1 source load (load width from the SOURCE type).
// CHECK: %[[SRC:.*]] = call_opaque "__riscv_vle32_v_i32m1"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// The widening convert: i32/m1 source -> i64/m2 result (intrinsic from the RESULT type).
// CHECK: %[[WIDE:.*]] = call_opaque "__riscv_vwcvt_x_x_v_i64m2"(%[[SRC]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint64m2_t">
// CHECK: call_opaque "__riscv_vse64_v_i64m2"(%{{.*}}, %[[WIDE]], %[[BODYVL]])
// CHECK: return
