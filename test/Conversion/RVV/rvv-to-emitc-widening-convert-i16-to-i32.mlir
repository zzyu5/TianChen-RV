// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening CONVERSION i16/mf2 -> i32/m1. The typed
// weft_rvv.widening_convert sign-extends the loaded narrow i16/mf2 vector into a
// one-step-wider i32/m1 vector via __riscv_vwcvt_x_x_v_i32m1. The vwcvt
// dtype/lmul derive from the RESULT vector type (i32/m1), NOT the source; the
// setvl/store run on the DEST config (e32m1), the load on the SOURCE (vle16
// i16mf2). Structure-level CHECKs; byte identity to the legacy oracle is pinned
// by the Target/RVV pre-realized widen-i16-to-i32 artifact fixture.

module {
  weft.exec.kernel @rvv_widen_i16_to_i32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_widen_i16_to_i32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widen-i16-to-i32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "widen-i16-to-i32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widen-i16-to-i32:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %widened = weft_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %widened, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_widen_i16_to_i32_kernel_rvv_widen_i16_to_i32(%arg0: !emitc.ptr<!emitc.opaque<"const int16_t">>, %arg1: !emitc.ptr<!emitc.opaque<"int32_t">>, %arg2: !emitc.opaque<"size_t">)
// The setvl/loop run on the DEST config (e32m1), not the narrow source.
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The narrow source load lands vint16mf2_t (load width from the SOURCE type).
// CHECK: %[[SRC:.*]] = call_opaque "__riscv_vle16_v_i16mf2"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int16_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// The widening convert: i16/mf2 source -> i32/m1 result (intrinsic from the RESULT type).
// CHECK: %[[WIDE:.*]] = call_opaque "__riscv_vwcvt_x_x_v_i32m1"(%[[SRC]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[WIDE]], %[[BODYVL]])
// CHECK: return
