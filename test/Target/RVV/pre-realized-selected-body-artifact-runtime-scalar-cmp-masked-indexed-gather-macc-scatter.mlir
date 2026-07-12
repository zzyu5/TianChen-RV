// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated/s//provider_supported_mirror:rvv-script-derived-pre-composite-gather-macc-scatter/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n/s//cmp_lhs,rhs_scalar,payload,gather_src,acc,index,dst,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;/s//payload=dot-rhs-input-buffer->@stale_dot_rhs_input_buffer;/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXEC-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/dst=output-buffer:dst:abi|olddst|gpass|scatter|hdr/s//dst=output-buffer:dst:abi|olddst|gpass|stale-scatter|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-OPERAND-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.composite_resource.vector_register_budget\", value = \"32\"/s//weft_rvv.composite_resource.vector_register_budget\", value = \"4\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-COMPOSITE-RESOURCE
// RUN: sed '0,/^      %index = weft_rvv.runtime_abi_value/{s/exec_binding = @abi_index_input_buffer, //}' %s | not weft-opt --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-EXEC-BINDING

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
// REALIZED-DAG: weft_rvv.composite_resource.selected_candidate = "rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]"
// REALIZED-DAG: weft_rvv.composite_resource.vl_policy = "runtime-avl-single-setvl"
// REALIZED-DAG: weft_rvv.composite_resource.peak_live_vector_groups = 8 : i64
// REALIZED-DAG: weft_rvv.composite_resource.vector_register_budget = 32 : i64
// REALIZED-DAG: weft_rvv.composite_resource.runtime_abi_order = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n"
// REALIZED-DAG: weft_rvv.composite_resource.target_capability_provider_mirror = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"
// REALIZED-DAG: weft_rvv.composite_resource.target_capability_legality_mirror = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"
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
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_indexed_gather_macc_scatter"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_indexed_load+weft_rvv.masked_macc+weft_rvv.masked_indexed_store"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "runtime-scalar-computed-mask-indexed-gather-macc-scatter"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n"}
// PLAN-SAME: {key = "weft_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@rvv_pre_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-composite-gather-macc-scatter-case"}
// PLAN-SAME: {key = "weft_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-composite-gather-macc-scatter-fallback-envelope"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1;
// PLAN-SAME: gather_src=source-input-buffer:gather_src
// PLAN-SAME: payload=dot-rhs-input-buffer:payload
// PLAN-SAME: acc=accumulator-input-buffer:acc
// PLAN-SAME: index=index-input-buffer:index
// PLAN-SAME: dst=output-buffer:dst
// PLAN-SAME: {key = "weft_rvv.exec_abi_bindings", value = "cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;gather_src=source-input-buffer->@abi_source_input_buffer;payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;index=index-input-buffer->@abi_index_input_buffer;dst=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-indexed-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: {key = "weft_rvv.composite_route_family_plan", value = "rvv-composite-gather-macc-scatter-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.composite_typed_compute_chain", value = "weft_rvv.masked_indexed_load+weft_rvv.masked_macc+weft_rvv.masked_indexed_store"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.selected_candidate", value = "rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.vl_policy", value = "runtime-avl-single-setvl"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.peak_live_vector_groups", value = "8"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.runtime_abi_order", value = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "weft_rvv.composite_resource.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "weft_rvv.indexed_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.indexed_write_side_contract", value = "gather-payload-acc-before-active-indexed-write;destination-before-inactive-tail-preserve"}
// PLAN-SAME: {key = "weft_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "weft_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "weft_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: {key = "weft_rvv.indexed_data_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "weft_rvv.indexed_destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pre_composite

// HEADER: weft.rvv.selected_variant: @rvv_pre_composite
// HEADER-DAG: weft.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n
// HEADER-DAG: weft.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@rvv_pre_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-composite-gather-macc-scatter-case
// HEADER-DAG: weft.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-composite-gather-macc-scatter-fallback-envelope
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated
// HEADER-DAG: weft.rvv.composite_route_family_plan: rvv-composite-gather-macc-scatter-route-family-plan.v1
// HEADER-DAG: weft.rvv.composite_typed_compute_chain: weft_rvv.masked_indexed_load+weft_rvv.masked_macc+weft_rvv.masked_indexed_store
// HEADER-DAG: weft.rvv.composite_resource.selected_candidate: rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]
// HEADER-DAG: weft.rvv.composite_resource.vector_register_budget: 32
// HEADER-DAG: weft.rvv.composite_resource.runtime_abi_order: cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n
// HEADER-DAG: weft.rvv.composite_resource.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1
// HEADER-DAG: weft.rvv.exec_abi_bindings: cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;gather_src=source-input-buffer->@abi_source_input_buffer;payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;index=index-input-buffer->@abi_index_input_buffer;dst=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count
// HEADER-DAG: weft.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER-DAG: weft.rvv.indexed_memory_layout: unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi
// HEADER-DAG: weft.rvv.indexed_write_side_contract: gather-payload-acc-before-active-indexed-write;destination-before-inactive-tail-preserve
// HEADER-DAG: weft.rvv.index_uniqueness: unique
// HEADER: void weft_emitc_pre_realized_composite_masked_indexed_gather_macc_scatter_kernel_rvv_pre_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);

// STALE-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-pre-composite-gather-macc-scatter

// STALE-ABI: composite resource runtime ABI order must mirror realized/provider-derived fact
// STALE-ABI-SAME: cmp_lhs,rhs_scalar,payload,gather_src,acc,index,dst,n

// STALE-EXEC-BINDING: candidate weft_rvv.exec_abi_bindings provenance must mirror selected weft.exec ABI binding summary
// STALE-EXEC-BINDING-SAME: payload=dot-rhs-input-buffer->@stale_dot_rhs_input_buffer

// STALE-OPERAND-BINDING: candidate weft_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-OPERAND-BINDING-SAME: stale-scatter

// STALE-COMPOSITE-RESOURCE: metadata key '{{.*}}composite_resource.vector_register_budget'{{.*}}'32' but was '4'

// MISSING-EXEC-BINDING: requires weft_rvv.runtime_abi_value 'index' with role 'index-input-buffer' to carry exec_binding to a weft.exec ABI declaration
