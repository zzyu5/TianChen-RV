// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass runs the same real MLIR DialectConversion as
// the i32/m1 add beachhead, here over an i64/m1 (SEW=64) elementwise add body.
// This pins the SEW=64 extension of the TypeConverter / intrinsic mangler: the
// !tcrv_rvv.vector<i64, "m1"> maps to emitc.opaque<"vint64m1_t"> and the load/
// binary/store/setvl callees mangle to the e64/i64m1 intrinsic family. The i64
// scalar element C types (const int64_t* / int64_t*) are derived from the
// runtime_abi_value c_type spellings. Byte-equivalence of the rendered C to the
// hardware-validated legacy oracle is checked by the e2e harness (PASS op=i64_add).

module {
  tcrv.exec.kernel @explicit_selected_body_i64_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_i64_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i64_add, sew = 64 : i64, source_kernel = "explicit_selected_body_i64_add_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// The conversion is complete: no tcrv_rvv body op and no unrealized cast survive.
// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.include <"riscv_vector.h">

// extern "C" function over the i64 ABI: const int64_t* inputs, int64_t* output,
// size_t count (last argument).
// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_i64_add_kernel_explicit_selected_body_rvv_i64_add(
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"const int64_t">>
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"int64_t">>
// CHECK-SAME: %[[N:[a-zA-Z0-9_]+]]: !emitc.opaque<"size_t">
// CHECK-SAME: specifiers = ["extern", "\22C\22"]

// SEW=64 setvl mangles to e64m1.
// CHECK: %[[VLMAX:.*]] = call_opaque "__riscv_vsetvl_e64m1"(%[[N]]) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// CHECK: for %[[I:[a-zA-Z0-9_]+]] = %{{.*}} to %[[N]] step %[[VLMAX]]
// CHECK: %[[REM:.*]] = sub %[[N]], %[[I]] : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e64m1"(%[[REM]]) : (!emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// i64 loads map to vle64/i64m1 over vint64m1_t.
// CHECK: %[[LHSPTR:.*]] = add %{{.*}}, %[[I]] : (!emitc.ptr<!emitc.opaque<"const int64_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int64_t">>
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vle64_v_i64m1"(%[[LHSPTR]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int64_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint64m1_t">
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vle64_v_i64m1"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int64_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint64m1_t">

// add mangles to vadd/i64m1 over the typed vint64m1_t vectors.
// CHECK: %[[SUM:.*]] = call_opaque "__riscv_vadd_vv_i64m1"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]]) : (!emitc.opaque<"vint64m1_t">, !emitc.opaque<"vint64m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint64m1_t">

// store mangles to vse64/i64m1.
// CHECK: call_opaque "__riscv_vse64_v_i64m1"(%{{.*}}, %[[SUM]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"int64_t">>, !emitc.opaque<"vint64m1_t">, !emitc.opaque<"size_t">) -> ()

// CHECK: return
