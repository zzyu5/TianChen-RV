// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// base-memory BYTE-strided load body via a real MLIR DialectConversion. Unlike
// the elementwise strided family (which scales an ELEMENT stride by 4 via
// ptrdiff_t), this rung receives a runtime BYTE stride (source-byte-stride ABI
// role) and computes the element pointer in BYTE space:
//   (const uint8_t*) base + i * stride_bytes, cast back to (const int32_t*),
//   then __riscv_vlse32_v_i32m1(ptr, stride_bytes /*AS-IS*/, vl).
// The weft_rvv.move{copy} is a passthrough; the store is unit-stride. The
// byte-vs-element addressing is selected from the typed stride ABI role.

module {
  weft.exec.kernel @explicit_selected_body_strided_load_unit_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_strided_load_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-load-unit-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-load-unit-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-load-unit-store:n", role = "runtime-element-count"} : index
      %stride_bytes = weft_rvv.runtime_abi_value {c_name = "stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-load-unit-store:stride-bytes", role = "source-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %loaded = weft_rvv.strided_load %src, %stride_bytes, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %moved = weft_rvv.move %loaded, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %moved, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_strided_load_unit_store_kernel_explicit_selected_body_rvv_strided_load_unit_store(

// byte-space pointer: cast base to const uint8_t*, add i*stride_bytes, cast back.
// CHECK: %[[BYTEBASE:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const int32_t">> to !emitc.ptr<!emitc.opaque<"const uint8_t">>
// CHECK: %[[OFF:.*]] = mul %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[BYTEPTR:.*]] = add %[[BYTEBASE]], %[[OFF]] : (!emitc.ptr<!emitc.opaque<"const uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint8_t">>
// CHECK: %[[PTR:.*]] = cast %[[BYTEPTR]] : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int32_t">>

// vlse with the runtime byte stride passed AS-IS (no *4 ptrdiff_t scaling).
// CHECK: %[[VEC:.*]] = call_opaque "__riscv_vlse32_v_i32m1"(%[[PTR]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// unit-stride store of the loaded vector.
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[VEC]], %{{.*}})
// CHECK: return
