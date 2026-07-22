// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for the bounded Stage 2 non-i32 slice.
// The RVV plugin must derive SEW64/i64 route facts from the typed body/config.

module {
  weft.exec.kernel @pre_realized_body_i64_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_i64_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:n", role = "runtime-element-count"} : index
      weft_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_i64_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-i64-add-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-i64-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body
// REALIZED: %[[N:[0-9]+]] = weft_rvv.runtime_abi_value {{.*}}c_name = "n"
// REALIZED-SAME: c_type = "size_t"
// REALIZED-SAME: role = "runtime-element-count"
// REALIZED-SAME: : index
// REALIZED-NEXT: %[[VL:[0-9]+]] = weft_rvv.setvl %[[N]] {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_i64_add
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i64, "m1">
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i64, "m1">
// REALIZED: weft_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: !weft_rvv.vector<i64, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i64, "m1">
// REALIZED-NOT: weft_rvv.typed_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_i64_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_i64_add
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_i64_add_kernel_pre_realized_body_rvv_i64_add(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n);
