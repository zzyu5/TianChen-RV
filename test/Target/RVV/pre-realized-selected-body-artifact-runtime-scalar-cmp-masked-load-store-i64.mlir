// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pr_rt_cmp_mload_i64_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pr_rvv_cmp_mload_i64 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int64_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:rhs_scalar", role = "rhs-scalar-value"} : i64
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body %lhs, %rhs_scalar, %src, %dst, %n {inactive_lane_policy = "preserve-old-destination", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-load-store", op_kind = "runtime_scalar_cmp_masked_load_store", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : (!tcrv_rvv.runtime_abi_value, i64, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pr_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pr_rvv_cmp_mload_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64-case"}
      tcrv.exec.fallback @pr_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-load-store-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pr_rvv_cmp_mload_i64
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[OLD:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i64, "m1">
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !tcrv_rvv.mask<i64, "m1">
// REALIZED: %[[LOADED:.*]] = tcrv_rvv.masked_load %{{.*}}, %[[MASK]], %[[OLD]], %[[VL]]
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: memory_form = "masked-unit-load"
// REALIZED: tcrv_rvv.store %{{.*}}, %[[LOADED]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.masked_store
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_load_store_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_load_store"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store"}
// PLAN-SAME: target = @pr_rvv_cmp_mload_i64

// HEADER: tianchenrv.rvv.selected_variant: @pr_rvv_cmp_mload_i64
// HEADER: tianchenrv.rvv.element_type: i64
// HEADER: tianchenrv.rvv.sew: 64
// HEADER: tianchenrv.rvv.lmul: m1
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-load-store-leaf-profile.v1
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs/source/passthrough:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:masked-load-store
// HEADER: void tcrv_emitc_pr_rt_cmp_mload_i64_kernel_pr_rvv_cmp_mload_i64(const int64_t *lhs, int64_t rhs_scalar, const int64_t *src, int64_t *dst, size_t n);
