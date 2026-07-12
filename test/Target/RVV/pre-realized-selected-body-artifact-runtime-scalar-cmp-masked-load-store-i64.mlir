// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pr_rt_cmp_mload_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pr_rvv_cmp_mload_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int64_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:rhs_scalar", role = "rhs-scalar-value"} : i64
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body %lhs, %rhs_scalar, %src, %dst, %n {inactive_lane_policy = "preserve-old-destination", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-load-store", op_kind = "runtime_scalar_cmp_masked_load_store", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : (!weft_rvv.runtime_abi_value, i64, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pr_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pr_rvv_cmp_mload_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64-case"}
      weft.exec.fallback @pr_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pr_rvv_cmp_mload_i64
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[OLD:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i64, "m1">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i64, "m1">
// REALIZED: %[[LOADED:.*]] = weft_rvv.masked_load %{{.*}}, %[[MASK]], %[[OLD]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-load"
// REALIZED: weft_rvv.store %{{.*}}, %[[LOADED]], %[[VL]]
// REALIZED-NOT: weft_rvv.masked_store
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_load_store"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "64"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store"}
// PLAN-SAME: target = @pr_rvv_cmp_mload_i64

// HEADER: weft.rvv.selected_variant: @pr_rvv_cmp_mload_i64
// HEADER: weft.rvv.element_type: i64
// HEADER: weft.rvv.sew: 64
// HEADER: weft.rvv.lmul: m1
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1
// HEADER: weft.rvv.c_type_mapping: vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store
// HEADER: void weft_emitc_pr_rt_cmp_mload_i64_kernel_pr_rvv_cmp_mload_i64(const int64_t *lhs, int64_t rhs_scalar, const int64_t *src, int64_t *dst, size_t n);
