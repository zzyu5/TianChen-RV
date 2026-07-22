// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided memory
// movement slice. The RVV plugin must realize the source byte-stride ABI operand
// into explicit strided_load/move/unit-store typed structure before the
// provider may construct the EmitC route.
// The target artifact/header checks prove the selected-boundary materialized
// body feeds emission planning and export; this base-memory slice is not a
// direct route-entry fixture.

module {
  weft.exec.kernel @pre_realized_body_strided_load_unit_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_strided_load_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:n", role = "runtime-element-count"} : index
      %stride_bytes = weft_rvv.runtime_abi_value {c_name = "stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-load-unit-store:stride-bytes", role = "source-byte-stride"} : index
      weft_rvv.typed_strided_memory_pre_realized_body %src, %out, %n, %stride_bytes {lmul = "m1", memory_form = "strided-load-unit-store", op_kind = "strided_load_unit_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_strided_load_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-strided-load-unit-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-strided-load-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_strided_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_strided_load_unit_store
// REALIZED: weft_rvv.strided_load
// REALIZED: weft_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_strided_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_strided_load_unit_store

// HEADER-DAG: weft.rvv.selected_variant: @pre_realized_body_rvv_strided_load_unit_store
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_strided_load_unit_store_kernel_pre_realized_body_rvv_strided_load_unit_store(const int32_t *src, int32_t *out, size_t n, size_t stride_bytes);





