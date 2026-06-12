// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv RHS
// broadcast-load beachhead body via a real MLIR DialectConversion.
// tcrv_rvv.broadcast_load reads the first RHS element (base[0]) via an
// emitc.subscript + emitc.load, then splats it with __riscv_vmv_v_x_i32m1.
// Byte-equivalence to the legacy materializer C is pinned by the e2e diff;
// this test asserts the emitc STRUCTURE.

module {
  tcrv.exec.kernel @explicit_selected_body_broadcast_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_i32_broadcast_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-broadcast:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i32_broadcast_add, sew = 32 : i64, source_kernel = "explicit_selected_body_broadcast_add_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.broadcast_load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_broadcast_add_kernel_explicit_selected_body_rvv_i32_broadcast_add(
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
