// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// computed-mask indexed scatter-store body (Stage 3 换心) via a real MLIR
// DialectConversion:
//   tcrv_rvv.load x3       -> __riscv_vle32_v_i32m1 (cmp_lhs, cmp_rhs, source)
//   tcrv_rvv.index_load    -> __riscv_vle32_v_u32m1 (raw element index vector)
//                             + __riscv_vmul_vx_u32m1 (element->byte scale, EARLY)
//   tcrv_rvv.compare       -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   tcrv_rvv.masked_indexed_store -> __riscv_vsoxei32_v_i32m1_m (masked scatter)
// The index_load + its byte scale are emitted EARLY (right after the first
// compare-LHS load), matching the legacy string-plan byte order so the rendered
// C is byte-identical to the legacy oracle. The masked scatter consumes the
// COMPARE mask and the pre-scaled byte offsets: (mask, dst, byte_offsets, value,
// vl). Asserts STRUCTURE; byte-equivalence pinned by the e2e diff.

module {
  tcrv.exec.kernel @computed_mask_indexed_scatter_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_computed_mask_indexed_scatter_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_computed_mask_indexed_scatter_store, sew = 32 : i64, source_kernel = "computed_mask_indexed_scatter_store_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %idx = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %m = tcrv_rvv.compare %a, %b, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        tcrv_rvv.masked_indexed_store %dst, %idx, %m, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", memory_form = "masked-indexed-store", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_computed_mask_indexed_scatter_store_kernel_rvv_computed_mask_indexed_scatter_store(

// compare-LHS load, then the index_load + byte-scale EARLY (index-early order).
// CHECK: %[[A:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[IDX:.*]] = call_opaque "__riscv_vle32_v_u32m1"
// CHECK: %[[BYTES:.*]] = call_opaque "__riscv_vmul_vx_u32m1"(%[[IDX]], %{{.*}}, %{{.*}})

// compare produces the predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"

// masked indexed (ordered) scatter store: (mask, dst, byte_offsets, value, vl).
// CHECK: call_opaque "__riscv_vsoxei32_v_i32m1_m"(%[[MASK]], %{{.*}}, %[[BYTES]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
