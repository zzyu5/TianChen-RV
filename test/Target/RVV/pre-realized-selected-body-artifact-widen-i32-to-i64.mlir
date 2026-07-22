// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// conversion slice. The RVV plugin must derive source i32/m1, destination
// i64/m2, memory, policy, route, and ABI facts from typed body/config facts.

module {
  weft.exec.kernel @pre_realized_body_widen_i32_to_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_widen_i32_to_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_widen_i32_to_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-widen-i32-to-i64-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widen-i32-to-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widen_i32_to_i64
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.widening_convert
// REALIZED-SAME: kind = "widen_i32_to_i64"
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i64, "m2">
// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widen_i32_to_i64

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_widen_i32_to_i64
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_widen_i32_to_i64_kernel_pre_realized_body_rvv_widen_i32_to_i64(const int32_t *lhs, int64_t *out, size_t n);




