// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// base-memory unit-load to BYTE-strided-store body via a real MLIR
// DialectConversion:
//   tcrv_rvv.load          -> __riscv_vle32_v_i32m1 (unit-stride source load)
//   tcrv_rvv.move{copy}    -> passthrough (no call)
//   tcrv_rvv.strided_store -> byte-space dst pointer + __riscv_vsse32_v_i32m1
//                             with the runtime byte stride passed AS-IS.
// The destination-byte-stride ABI role selects the byte-space addressing.

module {
  tcrv.exec.kernel @explicit_selected_body_unit_load_strided_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_unit_load_strided_store, sew = 32 : i64, source_kernel = "explicit_selected_body_unit_load_strided_store_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.strided_store %dst, %moved, %dst_stride_bytes, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, index, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_unit_load_strided_store_kernel_explicit_selected_body_rvv_unit_load_strided_store(
// CHECK: %[[VEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// byte-space dst pointer + vsse with the byte stride AS-IS.
// CHECK: %[[BYTEBASE:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"int32_t">> to !emitc.ptr<!emitc.opaque<"uint8_t">>
// CHECK: %[[PTR:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vsse32_v_i32m1"(%[[PTR]], %{{.*}}, %[[VEC]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"size_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
