// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 strided add. The
// RVV plugin must realize the explicit stride ABI operands into typed
// strided_load/binary/strided_store structure before the provider route is
// allowed to construct the strided_add EmitC route.

module {
  weft.exec.kernel @pre_realized_body_strided_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_strided_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:rhs-stride", role = "rhs-input-stride"} : index
      %out_stride = weft_rvv.runtime_abi_value {c_name = "out_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-add:out-stride", role = "output-stride"} : index
      weft_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n, %lhs_stride, %rhs_stride, %out_stride {lmul = "m1", memory_form = "strided-load-store", op_kind = "add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_strided_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-strided-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-strided-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: weft_rvv.strided_load
// REALIZED: weft_rvv.strided_load
// REALIZED: weft_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: weft_rvv.strided_store
// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_strided_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_strided_add
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_strided_add_kernel_pre_realized_body_rvv_strided_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride, size_t out_stride);



