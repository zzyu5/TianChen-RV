// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// computed-mask byte-strided store body (Stage 3 换心) via a real MLIR
// DialectConversion:
//   tcrv_rvv.load x3        -> __riscv_vle32_v_i32m1 (cmp_lhs, cmp_rhs, source)
//   tcrv_rvv.compare        -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   tcrv_rvv.masked_strided_store -> __riscv_vsse32_v_i32m1_m (masked strided store)
// The destination pointer is computed in BYTE space from the runtime byte
// stride: (int32_t *)((uint8_t *)dst + i * stride). The masked strided store
// consumes the COMPARE mask: (mask, ptr, byte_stride, value, vl). Asserts
// STRUCTURE; byte-equivalence pinned by the e2e diff.

module {
  tcrv.exec.kernel @computed_mask_strided_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_computed_mask_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %stride = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_computed_mask_strided_store, sew = 32 : i64, source_kernel = "computed_mask_strided_store_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %m = tcrv_rvv.compare %a, %b, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        tcrv_rvv.masked_strided_store %dst, %m, %payload, %stride, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-strided-store", stride_unit = "byte"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, index, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_computed_mask_strided_store_kernel_rvv_computed_mask_strided_store(

// CHECK: %[[A:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[B:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[PAYLOAD:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// compare produces the predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%[[A]], %[[B]], %{{.*}})

// byte-space destination pointer: cast dst to uint8_t*, + i*stride, cast back to
// int32_t*.
// CHECK: %[[BYTEBASE:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"int32_t">> to !emitc.ptr<!emitc.opaque<"uint8_t">>
// CHECK: %[[BYTEPTR:.*]] = add %[[BYTEBASE]], %{{.*}} : (!emitc.ptr<!emitc.opaque<"uint8_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"uint8_t">>
// CHECK: %[[PTR:.*]] = cast %[[BYTEPTR]] : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"int32_t">>

// masked byte-strided store consuming the COMPARE mask:
// (mask, ptr, byte_stride, value, vl).
// CHECK: call_opaque "__riscv_vsse32_v_i32m1_m"(%[[MASK]], %[[PTR]], %{{.*}}, %[[PAYLOAD]], %{{.*}})
// CHECK: return
