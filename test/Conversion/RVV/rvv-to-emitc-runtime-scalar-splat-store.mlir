// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv runtime
// scalar splat-store beachhead body via a real MLIR DialectConversion. The RHS
// scalar runtime ABI value becomes a scalar function parameter; tcrv_rvv.splat
// lowers to the __riscv_vmv_v_x_i32m1 scalar-splat intrinsic; the store lowers
// to __riscv_vse32_v_i32m1. Byte-equivalence of the rendered C to the legacy
// materializer output is pinned by the e2e diff; this test asserts STRUCTURE.

module {
  tcrv.exec.kernel @explicit_selected_body_runtime_scalar_splat_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_runtime_scalar_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-runtime-scalar-splat-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_runtime_scalar_splat_store, sew = 32 : i64, source_kernel = "explicit_selected_body_runtime_scalar_splat_store_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// The conversion is complete: no tcrv_rvv op and no unrealized cast survive.
// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.include <"riscv_vector.h">

// The RHS scalar ABI value (int32_t, not a pointer) is the first function
// parameter; the int32 output pointer and the size_t count follow.
// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_runtime_scalar_splat_store_kernel_explicit_selected_body_rvv_runtime_scalar_splat_store(
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
