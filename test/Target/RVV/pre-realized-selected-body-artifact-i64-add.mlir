// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for the bounded Stage 2 non-i32 slice.
// The RVV plugin must derive SEW64/i64 route facts from the typed body/config.

module {
  tcrv.exec.kernel @pre_realized_body_i64_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_i64_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_i64_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-i64-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-i64-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_i64_add
// REALIZED: tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED-NOT: tcrv_rvv.typed_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-selected-body-sew64-lmul-m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_i64_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_i64_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.element_type: i64
// HEADER: void tcrv_emitc_pre_realized_body_i64_add_kernel_pre_realized_body_rvv_i64_add(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n);
