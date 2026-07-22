// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/weft.exec.case @rvv_explicit_composite/d' %s | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-CASE
// RUN: sed '/weft.exec.fallback @explicit_composite_scalar_fallback/d' %s | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-FALLBACK
// RUN: sed '0,/^      weft.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case"}/s//      weft.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case", runtime_guard_required = true}/' %s | not weft-opt --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-GUARD

// Hand-authored explicit selected-body input for the Stage2 runtime scalar
// compare, masked indexed gather, masked MAcc, and masked indexed scatter
// composite. The target artifact must consume provider-owned route facts and
// mirrors from this realized typed body instead of metadata or route names.

module {
  weft.exec.kernel @explicit_composite_masked_indexed_gather_macc_scatter_kernel {
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
    weft.exec.variant @rvv_explicit_composite attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, weft_rvv.require_exec_abi_bindings = true} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", exec_binding = @abi_cmp_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:rhs_scalar", role = "rhs-scalar-value"} : i32
      %gather_src = weft_rvv.runtime_abi_value {c_name = "gather_src", c_type = "const int32_t *", exec_binding = @abi_source_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:gather_src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %payload = weft_rvv.runtime_abi_value {c_name = "payload", c_type = "const int32_t *", exec_binding = @abi_dot_rhs_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:payload", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %index = weft_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", exec_binding = @abi_index_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:index", role = "index-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %cmp_lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %threshold_vec = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %payload_vec = weft_rvv.load %payload, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %acc_vec = weft_rvv.load %acc, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old_dst = weft_rvv.load %dst, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %indices = weft_rvv.index_load %index, %vl {index_eew = 32 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.index_vector<i32, "m1">
        %mask = weft_rvv.compare %cmp_lhs_vec, %threshold_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %gathered = weft_rvv.masked_indexed_load %gather_src, %indices, %mask, %old_dst, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, memory_form = "masked-indexed-load", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.masked_macc %mask, %gathered, %payload_vec, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.masked_indexed_store %dst, %indices, %mask, %sum, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", memory_form = "masked-indexed-store", offset_unit = "element"} : !weft_rvv.runtime_abi_value, !weft_rvv.index_vector<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_composite_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_composite_gather_macc_scatter", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case"}
      weft.exec.fallback @explicit_composite_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-composite-gather-macc-scatter-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_composite

// HEADER: weft.rvv.selected_variant: @rvv_explicit_composite
// HEADER: void weft_emitc_explicit_composite_masked_indexed_gather_macc_scatter_kernel_rvv_explicit_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);







// MISSING-DISPATCH-CASE: 'weft.exec.dispatch' op requires at least one weft.exec.case

// MISSING-DISPATCH-FALLBACK: 'weft.exec.dispatch' op requires exactly one weft.exec.fallback

// MISSING-RUNTIME-GUARD: 'weft.exec.case' op requires runtime_guard linkage to a dispatch-availability-guard runtime_param when runtime_guard_required=true








