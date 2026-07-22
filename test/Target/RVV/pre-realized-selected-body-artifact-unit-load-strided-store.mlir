// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided destination
// store slice. The RVV plugin must realize destination byte-stride ABI facts
// into explicit load/move/strided_store typed structure before the provider
// may construct the EmitC route.

module {
  weft.exec.kernel @pre_realized_body_unit_load_strided_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_unit_load_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:src", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_unit_load_strided_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-unit-load-strided-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_strided_store_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_unit_load_strided_store
// REALIZED: weft_rvv.load
// REALIZED: weft_rvv.move
// REALIZED-SAME: kind = "copy"
// REALIZED: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_strided_store_memory_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_unit_load_strided_store

// HEADER-DAG: weft.rvv.selected_variant: @pre_realized_body_rvv_unit_load_strided_store
// HEADER-DAG: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_unit_load_strided_store_kernel_pre_realized_body_rvv_unit_load_strided_store(const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);





