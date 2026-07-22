// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// dot-product reduction slice with runtime element strides on both i16mf2
// source operands. Stride/dtype/reduction/scalar-result facts must be derived
// from typed RVV body/config/runtime ABI facts by the RVV plugin.

module {
  weft.exec.kernel @pre_realized_strided_dot_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_strided_input_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-strided-input-dot:rhs-stride", role = "rhs-input-stride"} : index
      weft_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "element"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_strided_input_dot {origin = "rvv-plugin", policy = "pre-realized-selected-body-strided-input-dot-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-strided-input-dot-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[LHS:.*]] = weft_rvv.strided_load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: %[[RHS:.*]] = weft_rvv.strided_load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED-NOT: weft_rvv.load {{.*}}!weft_rvv.vector<i16, "mf2">
// REALIZED: %[[SUM:.*]] = weft_rvv.widening_dot_reduce %[[LHS]], %[[RHS]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32"
// REALIZED-SAME: kind = "signed_widening_dot_reduce_add"
// REALIZED-SAME: result_layout = "store-dot-reduction-lane0-to-output-scalar"
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED-NOT: weft_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_strided_input_dot

// HEADER: weft.rvv.selected_variant: @rvv_strided_input_dot
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_strided_dot_kernel_rvv_strided_input_dot(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n, size_t lhs_stride, size_t rhs_stride);
