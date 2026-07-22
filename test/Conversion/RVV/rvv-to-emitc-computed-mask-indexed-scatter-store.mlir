// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// computed-mask indexed scatter-store body (Stage 3 换心) via a real MLIR
// DialectConversion:
//   weft_rvv.load x3       -> __riscv_vle32_v_i32m1 (cmp_lhs, cmp_rhs, source)
//   weft_rvv.index_load    -> __riscv_vle32_v_u32m1 (raw element index vector)
//                             + __riscv_vmul_vx_u32m1 (element->byte scale, EARLY)
//   weft_rvv.compare       -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   weft_rvv.masked_indexed_store -> __riscv_vsoxei32_v_i32m1_m (masked scatter)
// The index_load + its byte scale are emitted EARLY (right after the first
// compare-LHS load), matching the legacy string-plan byte order so the rendered
// C is byte-identical to the legacy oracle. The masked scatter consumes the
// COMPARE mask and the pre-scaled byte offsets: (mask, dst, byte_offsets, value,
// vl). Asserts STRUCTURE; byte-equivalence pinned by the e2e diff.

module {
  weft.exec.kernel @computed_mask_indexed_scatter_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_computed_mask_indexed_scatter_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %payload = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %idx = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %m = weft_rvv.compare %a, %b, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        weft_rvv.masked_indexed_store %dst, %idx, %m, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", memory_form = "masked-indexed-store", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_computed_mask_indexed_scatter_store_kernel_rvv_computed_mask_indexed_scatter_store(

// compare-LHS load, then the index_load + byte-scale EARLY (index-early order).
// CHECK: %[[A:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[IDX:.*]] = call_opaque "__riscv_vle32_v_u32m1"
// CHECK: %[[BYTES:.*]] = call_opaque "__riscv_vmul_vx_u32m1"(%[[IDX]], %{{.*}}, %{{.*}})

// compare produces the predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"

// masked indexed (ordered) scatter store: (mask, dst, byte_offsets, value, vl).
// CHECK: call_opaque "__riscv_vsoxei32_v_i32m1_m"(%[[MASK]], %{{.*}}, %[[BYTES]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
