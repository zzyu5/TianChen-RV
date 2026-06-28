// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// base-memory indexed-gather body via a real MLIR DialectConversion:
//   tcrv_rvv.index_load  -> __riscv_vle32_v_u32m1 (unit-stride index buffer load)
//   tcrv_rvv.indexed_load -> __riscv_vmul_vx_u32m1 (element->byte index scale)
//                            + __riscv_vloxei32_v_i32m1 (ordered indexed gather)
//   tcrv_rvv.move{copy}  -> passthrough (no call)
//   tcrv_rvv.store       -> __riscv_vse32_v_i32m1 (unit-stride store)
// The !tcrv_rvv.index_vector<i32,"m1"> converts to vuint32m1_t. Byte-equivalence
// to the legacy materializer C is pinned by the e2e diff; this asserts STRUCTURE.

module {
  tcrv.exec.kernel @explicit_selected_body_indexed_gather_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_indexed_gather_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %data = tcrv_rvv.runtime_abi_value {c_name = "data", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:data", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-indexed-gather-unit-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_indexed_gather_unit_store, sew = 32 : i64, source_kernel = "explicit_selected_body_indexed_gather_unit_store_kernel", status = "selected-lowering-boundary"} {
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %loaded = tcrv_rvv.indexed_load %data, %indices, %vl {index_eew = 32 : i64, offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %moved, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_indexed_gather_unit_store_kernel_explicit_selected_body_rvv_indexed_gather_unit_store(

// index_load: unit-stride index buffer pointer + vle32_v_u32m1 -> vuint32m1_t.
// CHECK: %[[IPTR:.*]] = add %{{.*}} : (!emitc.ptr<!emitc.opaque<"const uint32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const uint32_t">>
// CHECK: %[[IDX:.*]] = call_opaque "__riscv_vle32_v_u32m1"(%[[IPTR]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"const uint32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint32m1_t">

// indexed_load: byte-scale the index vector, then ordered indexed gather.
// CHECK: %[[BYTES:.*]] = call_opaque "__riscv_vmul_vx_u32m1"(%[[IDX]], %{{.*}}) : (!emitc.opaque<"vuint32m1_t">, !emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vuint32m1_t">
// CHECK: %[[GATHERED:.*]] = call_opaque "__riscv_vloxei32_v_i32m1"(%{{.*}}, %[[BYTES]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"vuint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// unit-stride store of the gathered vector (move{copy} is a passthrough).
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[GATHERED]], %{{.*}})
// CHECK: return
