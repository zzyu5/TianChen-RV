// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// base-memory unit-load to BYTE-strided-store body via a real MLIR
// DialectConversion:
//   weft_rvv.load          -> __riscv_vle32_v_i32m1 (unit-stride source load)
//   weft_rvv.move{copy}    -> passthrough (no call)
//   weft_rvv.strided_store -> byte-space dst pointer + __riscv_vsse32_v_i32m1
//                             with the runtime byte stride passed AS-IS.
// The destination-byte-stride ABI role selects the byte-space addressing.

module {
  weft.exec.kernel @explicit_selected_body_unit_load_strided_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unit_load_strided_store, sew = 32 : i64, source_kernel = "explicit_selected_body_unit_load_strided_store_kernel", status = "selected-lowering-boundary"} {
        %loaded = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %moved = weft_rvv.move %loaded, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.strided_store %dst, %moved, %dst_stride_bytes, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_unit_load_strided_store_kernel_explicit_selected_body_rvv_unit_load_strided_store(
// CHECK: %[[VEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// byte-space dst pointer + vsse with the byte stride AS-IS.
// CHECK: %[[BYTEBASE:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"int32_t">> to !emitc.ptr<!emitc.opaque<"uint8_t">>
// CHECK: %[[PTR:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vsse32_v_i32m1"(%[[PTR]], %{{.*}}, %[[VEC]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"size_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
