// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 indexed gather
// memory movement slice. The RVV plugin must realize the data/index/output
// ABI operands into explicit index_load/indexed_load/move/unit-store typed
// structure before the provider may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_indexed_gather_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_indexed_gather_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %data = tcrv_rvv.runtime_abi_value {c_name = "data", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-gather-unit-store:data", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-gather-unit-store:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-gather-unit-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-indexed-gather-unit-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_indexed_gather_memory_pre_realized_body %data, %index, %out, %n {index_eew = 32 : i64, lmul = "m1", memory_form = "indexed-load-unit-store", offset_unit = "element", op_kind = "indexed_gather_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_indexed_gather_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-indexed-gather-unit-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-indexed-gather-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_indexed_gather_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_indexed_gather_unit_store
// REALIZED: tcrv_rvv.index_load
// REALIZED-SAME: index_eew = 32 : i64
// REALIZED: tcrv_rvv.indexed_load
// REALIZED-SAME: offset_unit = "element"
// REALIZED: tcrv_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_indexed_gather_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "indexed_gather_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "indexed-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "data,index,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "element-indexed-data-index-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "tcrv_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_data_memory_form", value = "indexed-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-indexed-gather-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_indexed_gather_unit_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_indexed_gather_unit_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-indexed-gather-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: data,index,out,n
// HEADER: void tcrv_emitc_pre_realized_body_indexed_gather_unit_store_kernel_pre_realized_body_rvv_indexed_gather_unit_store(const int32_t *data, const uint32_t *index, int32_t *out, size_t n);
