// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// base-memory indexed-scatter body via a real MLIR DialectConversion:
//   weft_rvv.load          -> __riscv_vle32_v_i32m1 (unit-stride source load)
//   weft_rvv.index_load    -> __riscv_vle32_v_u32m1 (index buffer load)
//   weft_rvv.move{copy}    -> passthrough (no call)
//   weft_rvv.indexed_store -> __riscv_vmul_vx_u32m1 (element->byte index scale)
//                             + __riscv_vsoxei32_v_i32m1 (ordered indexed scatter)
// Asserts STRUCTURE; byte-equivalence to the legacy C is pinned by the e2e diff.

module {
  weft.exec.kernel @explicit_selected_body_indexed_scatter_unit_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_indexed_scatter_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %loaded = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %indices = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %moved = weft_rvv.move %loaded, %vl {kind = "copy"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.indexed_store %dst, %indices, %moved, %vl {index_eew = 32 : i64, index_uniqueness = "unique", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_indexed_scatter_unit_load_kernel_explicit_selected_body_rvv_indexed_scatter_unit_load(
// CHECK: %[[SRCVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[IDX:.*]] = call_opaque "__riscv_vle32_v_u32m1"
// CHECK: %[[BYTES:.*]] = call_opaque "__riscv_vmul_vx_u32m1"(%[[IDX]], %{{.*}}) : (!emitc.opaque<"vuint32m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint32m1_t">
// CHECK: call_opaque "__riscv_vsoxei32_v_i32m1"(%{{.*}}, %[[BYTES]], %[[SRCVEC]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
