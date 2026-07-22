// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass runs a real MLIR DialectConversion
// (TypeConverter + ConversionTarget + OpConversionPattern + applyPartialConversion)
// over the typed generic weft_rvv unit-stride elementwise add beachhead body and
// builds structured MLIR EmitC directly on the typed SSA Values -- no string plan,
// no operand re-parser. This test asserts the STRUCTURE of that emitc (the for
// loop, the typed call_opaque intrinsics with the mangled __riscv_v... callees,
// the emitc.add pointer arithmetic and emitc.sub remaining-AVL, and the
// vint32m1_t/size_t emitc types), not re-parsed prose. Byte-equivalence of the
// rendered C to the hardware-validated golden is checked by piping this pass
// through mlir-translate --mlir-to-cpp in the e2e harness.

module {
  weft.exec.kernel @explicit_selected_body_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// No weft_rvv body op and no unrealized cast survive: the conversion is complete.
// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// Standalone EmitC module: the headers the RVV intrinsic body needs.
// CHECK: emitc.include <"stddef.h">
// CHECK: emitc.include <"stdint.h">
// CHECK: emitc.include <"riscv_vector.h">

// Function name = weft_emitc_<kernel>_<variant>, extern "C", ABI params derived
// from the runtime_abi_value c_type spellings (pointers + size_t). The two
// const-int32 input pointers, the int32 output pointer and the size_t count all
// appear; %[[N]] is the size_t AVL parameter (last argument).
// CHECK: emitc.func @weft_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"const int32_t">>
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK-SAME: %[[N:[a-zA-Z0-9_]+]]: !emitc.opaque<"size_t">
// CHECK-SAME: specifiers = ["extern", "\22C\22"]

// Pre-loop full-chunk setvl over the runtime AVL. Inside emitc.func the emitc
// ops print without the dialect prefix (default-dialect elision).
// CHECK: %[[VLMAX:.*]] = call_opaque "__riscv_vsetvl_e32m1"(%[[N]]) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[ZERO:.*]] = literal "0" : !emitc.opaque<"size_t">

// for (size_t i = 0; i < n; i += vlmax)
// CHECK: for %[[I:[a-zA-Z0-9_]+]] = %[[ZERO]] to %[[N]] step %[[VLMAX]]

// Remaining AVL = n - i, then the body-VL setvl.
// CHECK: %[[REM:.*]] = sub %[[N]], %[[I]] : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"(%[[REM]]) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// lhs load: pointer = base + i, then __riscv_vle32_v_i32m1 -> vint32m1_t.
// CHECK: %[[LHSPTR:.*]] = add %{{.*}}, %[[I]] : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int32_t">>
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"(%[[LHSPTR]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// rhs load.
// CHECK: %[[RHSPTR:.*]] = add %{{.*}}, %[[I]] : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int32_t">>
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"(%[[RHSPTR]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// add: __riscv_vadd_vv_i32m1 over the two typed vectors + body VL.
// CHECK: %[[SUM:.*]] = call_opaque "__riscv_vadd_vv_i32m1"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// store: pointer = out + i, then __riscv_vse32_v_i32m1 (void result).
// CHECK: %[[OUTPTR:.*]] = add %{{.*}}, %[[I]] : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%[[OUTPTR]], %[[SUM]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()

// CHECK: return
