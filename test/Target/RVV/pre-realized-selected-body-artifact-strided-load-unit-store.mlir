// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided memory
// movement slice. The RVV plugin must realize the source byte-stride ABI operand
// into explicit strided_load/move/unit-store typed structure before the
// provider may construct the EmitC route.
// The direct emission-plan runs above prove that route-entry realization carries
// this pre-realized body through target artifact/header export without a
// separate selected-lowering-boundary materialization pass.

module {
  tcrv.exec.kernel @pre_realized_body_strided_load_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_strided_load_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:n", role = "runtime-element-count"} : index
      %stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:stride-bytes", role = "source-byte-stride"} : index
      tcrv_rvv.typed_strided_memory_pre_realized_body %src, %out, %n, %stride_bytes {lmul = "m1", memory_form = "strided-load-unit-store", op_kind = "strided_load_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_strided_load_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-strided-load-unit-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-strided-load-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_strided_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_strided_load_unit_store
// REALIZED: tcrv_rvv.strided_load
// REALIZED: tcrv_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_strided_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "strided_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "strided-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,out,n,stride_bytes"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:strided_load_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:strided_load_unit_store.v1;src=source-input-buffer:src:runtime-abi-mirror|materialized-strided-load-base|move-source;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;stride_bytes=source-byte-stride:stride_bytes:runtime-abi-mirror|materialized-strided-load-stride|materialized-byte-address|header-mirror"}
// PLAN-SAME: {key = "tcrv_rvv.strided_memory_layout", value = "byte-strided-source-unit-stride-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.source_stride_source", value = "runtime_abi:stride_bytes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "strided-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-strided-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_strided_load_unit_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_strided_load_unit_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-strided-load-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,out,n,stride_bytes
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:strided_load_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:strided_load_unit_store.v1;src=source-input-buffer:src:runtime-abi-mirror|materialized-strided-load-base|move-source;out=output-buffer:out:runtime-abi-mirror|materialized-store-base|header-mirror;n=runtime-element-count:n:runtime-abi-mirror|setvl-avl|loop-control|header-mirror;stride_bytes=source-byte-stride:stride_bytes:runtime-abi-mirror|materialized-strided-load-stride|materialized-byte-address|header-mirror
// HEADER: void tcrv_emitc_pre_realized_body_strided_load_unit_store_kernel_pre_realized_body_rvv_strided_load_unit_store(const int32_t *src, int32_t *out, size_t n, size_t stride_bytes);
