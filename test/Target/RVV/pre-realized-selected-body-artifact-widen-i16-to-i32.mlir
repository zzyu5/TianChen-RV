// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// conversion slice. The RVV plugin must derive source i16/mf2, destination
// i32/m1, conversion kind, memory, policy, route, and ABI facts from typed
// body/config facts.

module {
  weft.exec.kernel @pre_realized_body_widen_i16_to_i32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_widen_i16_to_i32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i16mf2-to-i32m1", dest_lmul = "m1", dest_sew = 32 : i64, memory_form = "unit-stride-conversion", op_kind = "sign_extend_widen_vf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_widen_i16_to_i32 {origin = "rvv-plugin", policy = "pre-realized-selected-body-widen-i16-to-i32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widen-i16-to-i32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m1"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widen_i16_to_i32
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: weft_rvv.widening_convert
// REALIZED-SAME: kind = "sign_extend_widen_vf2"
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widen_i16_to_i32

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_widen_i16_to_i32
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_widen_i16_to_i32_kernel_pre_realized_body_rvv_widen_i16_to_i32(const int16_t *lhs, int32_t *out, size_t n);







