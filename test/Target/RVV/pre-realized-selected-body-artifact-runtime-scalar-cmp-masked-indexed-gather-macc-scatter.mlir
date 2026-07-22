// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized family bodies for runtime scalar masked indexed gather, masked
// MAcc, and masked indexed scatter. The RVV plugin-local realization owner must
// fuse them into one explicit composite weft_rvv body before provider route
// facts and target artifact ABI/header mirrors are accepted.

module {
  weft.exec.kernel @pre_realized_composite_masked_indexed_gather_macc_scatter_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.mem_window @abi_cmp_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.mem_window @abi_source_input_buffer {abi_role = "source-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_dot_rhs_input_buffer {abi_role = "dot-rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_index_input_buffer {abi_role = "index-input-buffer", access = "read", binding = "kernel-argument", c_type = "const uint32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.variant @rvv_pre_composite attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, weft_rvv.require_exec_abi_bindings = true} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", exec_binding = @abi_cmp_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:rhs_scalar", role = "rhs-scalar-value"} : i32
      %gather_src = weft_rvv.runtime_abi_value {c_name = "gather_src", c_type = "const int32_t *", exec_binding = @abi_source_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:gather_src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dot_lhs = weft_rvv.runtime_abi_value {c_name = "dot_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:dot_lhs", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %payload = weft_rvv.runtime_abi_value {c_name = "payload", c_type = "const int32_t *", exec_binding = @abi_dot_rhs_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:payload", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", exec_binding = @abi_index_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %scatter_src = weft_rvv.runtime_abi_value {c_name = "scatter_src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:scatter_src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "pre-realized-composite-gather-macc-scatter:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body %cmp_lhs, %rhs_scalar, %gather_src, %index, %dst, %n {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-indexed-gather-load-unit-store", offset_unit = "element", op_kind = "runtime_scalar_cmp_masked_indexed_gather_load_unit_store", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
      weft_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body %cmp_lhs, %rhs_scalar, %dot_lhs, %payload, %acc, %dst, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-macc", op_kind = "runtime_scalar_cmp_masked_macc_add", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
      weft_rvv.typed_runtime_scalar_computed_mask_indexed_scatter_pre_realized_body %cmp_lhs, %rhs_scalar, %scatter_src, %index, %dst, %n {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-indexed-scatter-store", offset_unit = "element", op_kind = "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_composite_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_pre_composite_gather_macc_scatter", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_pre_composite {origin = "rvv-plugin", policy = "pre-realized-composite-gather-macc-scatter-case"}
      weft.exec.fallback @pre_composite_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-composite-gather-macc-scatter-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_indexed_gather_pre_realized_body
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_indexed_scatter_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @rvv_pre_composite
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[THRESHOLD:.*]] = weft_rvv.splat
// REALIZED: %[[PAYLOAD:.*]] = weft_rvv.load
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[OLD_DST:.*]] = weft_rvv.load
// REALIZED: %[[INDICES:.*]] = weft_rvv.index_load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[THRESHOLD]], %[[VL]]
// REALIZED: %[[GATHERED:.*]] = weft_rvv.masked_indexed_load %{{.*}}, %[[INDICES]], %[[MASK]], %[[OLD_DST]], %[[VL]]
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_macc %[[MASK]], %[[GATHERED]], %[[PAYLOAD]], %[[ACC]], %[[VL]]
// REALIZED: weft_rvv.masked_indexed_store %{{.*}}, %[[INDICES]], %[[MASK]], %[[SUM]], %[[VL]]

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_composite

// HEADER: weft.rvv.selected_variant: @rvv_pre_composite
// HEADER: void weft_emitc_pre_realized_composite_masked_indexed_gather_macc_scatter_kernel_rvv_pre_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);






