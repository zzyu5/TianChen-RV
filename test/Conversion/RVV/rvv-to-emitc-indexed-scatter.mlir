// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// base-memory indexed-scatter body via a real MLIR DialectConversion:
//   tcrv_rvv.load          -> __riscv_vle32_v_i32m1 (unit-stride source load)
//   tcrv_rvv.index_load    -> __riscv_vle32_v_u32m1 (index buffer load)
//   tcrv_rvv.move{copy}    -> passthrough (no call)
//   tcrv_rvv.indexed_store -> __riscv_vmul_vx_u32m1 (element->byte index scale)
//                             + __riscv_vsoxei32_v_i32m1 (ordered indexed scatter)
// Asserts STRUCTURE; byte-equivalence to the legacy C is pinned by the e2e diff.

module {
  tcrv.exec.kernel @explicit_selected_body_indexed_scatter_unit_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_indexed_scatter_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_indexed_scatter_unit_load, sew = 32 : i64, source_kernel = "explicit_selected_body_indexed_scatter_unit_load_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.indexed_store %dst, %indices, %moved, %vl {index_eew = 32 : i64, index_uniqueness = "unique", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_indexed_scatter_unit_load_kernel_explicit_selected_body_rvv_indexed_scatter_unit_load(
// CHECK: %[[SRCVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[IDX:.*]] = call_opaque "__riscv_vle32_v_u32m1"
// CHECK: %[[BYTES:.*]] = call_opaque "__riscv_vmul_vx_u32m1"(%[[IDX]], %{{.*}}) : (!emitc.opaque<"vuint32m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint32m1_t">
// CHECK: call_opaque "__riscv_vsoxei32_v_i32m1"(%{{.*}}, %[[BYTES]], %[[SRCVEC]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
