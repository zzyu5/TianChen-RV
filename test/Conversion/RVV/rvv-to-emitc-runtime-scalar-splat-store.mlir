// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv runtime
// scalar splat-store beachhead body via a real MLIR DialectConversion. The RHS
// scalar runtime ABI value becomes a scalar function parameter; weft_rvv.splat
// lowers to the __riscv_vmv_v_x_i32m1 scalar-splat intrinsic; the store lowers
// to __riscv_vse32_v_i32m1. Byte-equivalence of the rendered C to the legacy
// materializer output is pinned by the e2e diff; this test asserts STRUCTURE.

module {
  weft.exec.kernel @explicit_selected_body_runtime_scalar_splat_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_runtime_scalar_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_runtime_scalar_splat_store, sew = 32 : i64, source_kernel = "explicit_selected_body_runtime_scalar_splat_store_kernel", status = "selected-lowering-boundary"} {
        %broadcast = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %broadcast, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// The conversion is complete: no weft_rvv op and no unrealized cast survive.
// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.include <"riscv_vector.h">

// The RHS scalar ABI value (int32_t, not a pointer) is the first function
// parameter; the int32 output pointer and the size_t count follow.
// CHECK: emitc.func @weft_emitc_explicit_selected_body_runtime_scalar_splat_store_kernel_explicit_selected_body_rvv_runtime_scalar_splat_store(
// CHECK-SAME: %[[SCALAR:[a-zA-Z0-9_]+]]: !emitc.opaque<"int32_t">
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK-SAME: %[[N:[a-zA-Z0-9_]+]]: !emitc.opaque<"size_t">
// CHECK-SAME: specifiers = ["extern", "\22C\22"]

// CHECK: for %[[I:[a-zA-Z0-9_]+]] = %{{.*}} to %[[N]] step %{{.*}}
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"

// splat: __riscv_vmv_v_x_i32m1(scalar, vl) over the scalar parameter directly.
// CHECK: %[[SPLAT:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"(%[[SCALAR]], %[[BODYVL]]) : (!emitc.opaque<"int32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// store: pointer = out + i, then __riscv_vse32_v_i32m1 (void result).
// CHECK: %[[OUTPTR:.*]] = add %{{.*}}, %[[I]] : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%[[OUTPTR]], %[[SPLAT]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()

// CHECK: return
