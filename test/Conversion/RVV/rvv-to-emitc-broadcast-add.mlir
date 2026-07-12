// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv RHS
// broadcast-load beachhead body via a real MLIR DialectConversion.
// weft_rvv.broadcast_load reads the first RHS element (base[0]) via an
// emitc.subscript + emitc.load, then splats it with __riscv_vmv_v_x_i32m1.
// Byte-equivalence to the legacy materializer C is pinned by the e2e diff;
// this test asserts the emitc STRUCTURE.

module {
  weft.exec.kernel @explicit_selected_body_broadcast_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_i32_broadcast_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i32_broadcast_add, sew = 32 : i64, source_kernel = "explicit_selected_body_broadcast_add_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.broadcast_load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_broadcast_add_kernel_explicit_selected_body_rvv_i32_broadcast_add(
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"

// lhs unit-stride load.
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// rhs broadcast-load: subscript base[0] -> load the scalar -> vmv_v_x splat.
// CHECK: %[[ZEROIDX:.*]] = literal "0" : index
// CHECK: %[[SUB:.*]] = subscript %{{.*}}[%[[ZEROIDX]]] : (!emitc.ptr<!emitc.opaque<"const int32_t">>, index) -> !emitc.lvalue<!emitc.opaque<"const int32_t">>
// CHECK: %[[SCALAR:.*]] = load %[[SUB]] : <!emitc.opaque<"const int32_t">>
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"(%[[SCALAR]], %[[BODYVL]]) : (!emitc.opaque<"const int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// add over the loaded lhs and the broadcast rhs.
// CHECK: %[[SUM:.*]] = call_opaque "__riscv_vadd_vv_i32m1"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]])

// CHECK: call_opaque "__riscv_vse32_v_i32m1"
// CHECK: return
