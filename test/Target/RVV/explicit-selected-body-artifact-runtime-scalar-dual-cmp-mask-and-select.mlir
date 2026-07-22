// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @explicit_dual_cmp_mask_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_rvv_dual_cmp_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs_a = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_a", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_a = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_a", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:cmp_lhs_b", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_b = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:rhs_scalar_b", role = "rhs-secondary-scalar-value"} : i32
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_rvv_dual_cmp_mask_select, sew = 32 : i64, source_kernel = "explicit_dual_cmp_mask_select_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_a_vec = weft_rvv.load %cmp_lhs_a, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_a_vec = weft_rvv.splat %rhs_scalar_a, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %cmp_lhs_b_vec = weft_rvv.load %cmp_lhs_b, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_b_vec = weft_rvv.splat %rhs_scalar_b, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %true_vec = weft_rvv.load %true_value, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %false_vec = weft_rvv.load %false_value, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask_a = weft_rvv.compare %cmp_lhs_a_vec, %rhs_a_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %mask_b = weft_rvv.compare %cmp_lhs_b_vec, %rhs_b_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %composed = weft_rvv.mask_and %mask_a, %mask_b, %vl {kind = "and"} : !weft_rvv.mask<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %selected = weft_rvv.select %composed, %true_vec, %false_vec, %vl : !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %selected, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_dual_cmp_mask_select_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_rvv_dual_cmp_mask_select {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-case"}
      weft.exec.fallback @explicit_dual_cmp_mask_select_scalar {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-dual-cmp-mask-and-select-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_rvv_dual_cmp_mask_select

// HEADER: weft.rvv.selected_variant: @explicit_rvv_dual_cmp_mask_select
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_dual_cmp_mask_select_kernel_explicit_rvv_dual_cmp_mask_select(const int32_t *cmp_lhs_a, int32_t rhs_scalar_a, const int32_t *cmp_lhs_b, int32_t rhs_scalar_b, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
