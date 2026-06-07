// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated/s//provider_supported_mirror:rvv-script-derived-composite-gather-macc-scatter/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n/s//cmp_lhs,gather_src,rhs_scalar,payload,acc,index,dst,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;/s//cmp_lhs=lhs-input-buffer->@stale_cmp_lhs_input_buffer;/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-EXEC-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.exec_abi_bindings/s//tcrv_rvv.exec_abi_bindings_removed/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-EXEC-MIRROR
// RUN: sed '0,/^      %index = tcrv_rvv.runtime_abi_value/{s/exec_binding = @abi_index_input_buffer, //}' %s | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-EXEC-BINDING
// RUN: sed '/tcrv.exec.case @rvv_explicit_composite/d' %s | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-CASE
// RUN: sed '/tcrv.exec.fallback @explicit_composite_scalar_fallback/d' %s | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-FALLBACK
// RUN: sed '0,/^      tcrv.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case"}/s//      tcrv.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case", runtime_guard_required = true}/' %s | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-GUARD
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/selected_dispatch_case_mirror:@rvv_explicit_composite/s//selected_dispatch_case_mirror:@stale_rvv_composite/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DISPATCH-CASE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/selected_dispatch_fallback_mirror:@explicit_composite_scalar_fallback/s//selected_dispatch_fallback_mirror:@stale_scalar_fallback/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DISPATCH-FALLBACK
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.selected_dispatch_case_mirror/s//tcrv_rvv.selected_dispatch_case_mirror_removed/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-CASE-MIRROR
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.selected_dispatch_fallback_mirror/s//tcrv_rvv.selected_dispatch_fallback_mirror_removed/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-DISPATCH-FALLBACK-MIRROR
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.composite_resource.selected_candidate\", value = \"rvv-composite-gather-macc-scatter-resource-candidate.v1\[rt-scmp-indexed-gather-macc-scatter,e32m1,u1\]\"/s//tcrv_rvv.composite_resource.selected_candidate\", value = \"artifact-name-derived-composite-resource\"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-COMPOSITE-RESOURCE
// RUN: sed '/^      tcrv_rvv.with_vl/s/tcrv_rvv.composite_resource.vl_policy = "runtime-avl-single-setvl", //' %s | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-COMPOSITE-RESOURCE

// Hand-authored explicit selected-body input for the Stage2 runtime scalar
// compare, masked indexed gather, masked MAcc, and masked indexed scatter
// composite. The target artifact must consume provider-owned route facts and
// mirrors from this realized typed body instead of metadata or route names.

module {
  tcrv.exec.kernel @explicit_composite_masked_indexed_gather_macc_scatter_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_cmp_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.mem_window @abi_source_input_buffer {abi_role = "source-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_dot_rhs_input_buffer {abi_role = "dot-rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_index_input_buffer {abi_role = "index-input-buffer", access = "read", binding = "kernel-argument", c_type = "const uint32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_explicit_composite attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.require_exec_abi_bindings = true} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", exec_binding = @abi_cmp_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:rhs_scalar", role = "rhs-scalar-value"} : i32
      %gather_src = tcrv_rvv.runtime_abi_value {c_name = "gather_src", c_type = "const int32_t *", exec_binding = @abi_source_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:gather_src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %payload = tcrv_rvv.runtime_abi_value {c_name = "payload", c_type = "const int32_t *", exec_binding = @abi_dot_rhs_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:payload", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", exec_binding = @abi_index_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "explicit-composite-gather-macc-scatter:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_explicit_composite, sew = 32 : i64, source_kernel = "explicit_composite_masked_indexed_gather_macc_scatter_kernel", status = "selected-lowering-boundary", tcrv_rvv.composite_resource.accumulator_layout = "separate-i32-vector-accumulator-input", tcrv_rvv.composite_resource.candidate_set = "rvv-composite-gather-macc-scatter-resource-candidate-set.v1[rt-scmp-indexed-gather-macc-scatter-e32m1-u1]", tcrv_rvv.composite_resource.legality = "legal", tcrv_rvv.composite_resource.legality_scope = "typed-composite-gather-macc-scatter-resource-legality.v1", tcrv_rvv.composite_resource.lmul = "m1", tcrv_rvv.composite_resource.mask_policy = "agnostic", tcrv_rvv.composite_resource.memory_form = "runtime-scalar-computed-mask-indexed-gather-macc-scatter", tcrv_rvv.composite_resource.operation = "runtime_scalar_cmp_masked_indexed_gather_macc_scatter", tcrv_rvv.composite_resource.peak_live_vector_groups = 8 : i64, tcrv_rvv.composite_resource.pipeline_intent = "single-vl-linear-gather-macc-scatter.v1", tcrv_rvv.composite_resource.prefetch_intent = "none", tcrv_rvv.composite_resource.rejection_reason = "none", tcrv_rvv.composite_resource.runtime_abi_order = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n", tcrv_rvv.composite_resource.runtime_avl_source = "runtime_abi:n", tcrv_rvv.composite_resource.selected_candidate = "rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]", tcrv_rvv.composite_resource.selection_reason = "static-bounded-runtime-scalar-computed-mask-indexed-gather-macc-scatter-e32m1-runtime-avl", tcrv_rvv.composite_resource.sew = 32 : i64, tcrv_rvv.composite_resource.tail_policy = "agnostic", tcrv_rvv.composite_resource.target_capability_legality_mirror = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic", tcrv_rvv.composite_resource.target_capability_provider_mirror = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact", tcrv_rvv.composite_resource.unroll_factor = 1 : i64, tcrv_rvv.composite_resource.vector_register_budget = 32 : i64, tcrv_rvv.composite_resource.vl_policy = "runtime-avl-single-setvl", tcrv_rvv.composite_resource.vsetvl_region_count = 1 : i64} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %threshold_vec = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %payload_vec = tcrv_rvv.load %payload, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old_dst = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %threshold_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %gathered = tcrv_rvv.masked_indexed_load %gather_src, %indices, %mask, %old_dst, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", index_eew = 32 : i64, memory_form = "masked-indexed-load", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.masked_macc %mask, %gathered, %payload_vec, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.masked_indexed_store %dst, %indices, %mask, %sum, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", index_eew = 32 : i64, index_uniqueness = "unique", memory_form = "masked-indexed-store", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_composite_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_composite_gather_macc_scatter", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_explicit_composite {origin = "rvv-plugin", policy = "explicit-composite-gather-macc-scatter-case"}
      tcrv.exec.fallback @explicit_composite_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-composite-gather-macc-scatter-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_indexed_gather_macc_scatter"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+tcrv_rvv.masked_indexed_store"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-indexed-gather-macc-scatter"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@rvv_explicit_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-composite-gather-macc-scatter-case"}
// PLAN-SAME: {key = "tcrv_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@explicit_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-composite-gather-macc-scatter-fallback-envelope"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1;
// PLAN-SAME: gather_src=source-input-buffer:gather_src
// PLAN-SAME: payload=dot-rhs-input-buffer:payload
// PLAN-SAME: acc=accumulator-input-buffer:acc
// PLAN-SAME: index=index-input-buffer:index
// PLAN-SAME: dst=output-buffer:dst
// PLAN-SAME: {key = "tcrv_rvv.exec_abi_bindings", value = "cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;gather_src=source-input-buffer->@abi_source_input_buffer;payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;index=index-input-buffer->@abi_index_input_buffer;dst=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/gather-src/payload/acc/passthrough/result:signed-e32m1,rhs_scalar:signed-scalar,index:u32m1,mask:b32,dst:runtime-scalar-masked-indexed-gather-macc-scatter"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-indexed-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.selected_candidate", value = "rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.vl_policy", value = "runtime-avl-single-setvl"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.peak_live_vector_groups", value = "8"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.vector_register_budget", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.runtime_abi_order", value = "cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "tcrv_rvv.composite_resource.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.index_source", value = "runtime_abi:index"}
// PLAN-SAME: {key = "tcrv_rvv.index_eew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.offset_unit", value = "element"}
// PLAN-SAME: {key = "tcrv_rvv.index_uniqueness", value = "unique"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_data_memory_form", value = "masked-indexed-load"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_destination_memory_form", value = "masked-indexed-store"}
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_explicit_composite

// HEADER: tianchenrv.rvv.selected_variant: @rvv_explicit_composite
// HEADER-DAG: tianchenrv.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n
// HEADER-DAG: tianchenrv.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@rvv_explicit_composite;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=explicit-composite-gather-macc-scatter-case
// HEADER-DAG: tianchenrv.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@explicit_composite_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=explicit-composite-gather-macc-scatter-fallback-envelope
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated
// HEADER-DAG: tianchenrv.rvv.composite_resource.selected_candidate: rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]
// HEADER-DAG: tianchenrv.rvv.composite_resource.vector_register_budget: 32
// HEADER-DAG: tianchenrv.rvv.composite_resource.runtime_abi_order: cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n
// HEADER-DAG: tianchenrv.rvv.composite_resource.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1
// HEADER-DAG: tianchenrv.rvv.exec_abi_bindings: cmp_lhs=lhs-input-buffer->@abi_cmp_lhs_input_buffer;rhs_scalar=rhs-scalar-value->@abi_rhs_scalar_value;gather_src=source-input-buffer->@abi_source_input_buffer;payload=dot-rhs-input-buffer->@abi_dot_rhs_input_buffer;acc=accumulator-input-buffer->@abi_accumulator_input_buffer;index=index-input-buffer->@abi_index_input_buffer;dst=output-buffer->@abi_output_buffer;n=runtime-element-count->@abi_runtime_element_count
// HEADER-DAG: tianchenrv.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER-DAG: tianchenrv.rvv.indexed_memory_layout: unit-stride-lhs-runtime-scalar-threshold-indexed-masked-gather-payload-accumulator-macc-indexed-masked-scatter-runtime-abi
// HEADER-DAG: tianchenrv.rvv.index_uniqueness: unique
// HEADER: void tcrv_emitc_explicit_composite_masked_indexed_gather_macc_scatter_kernel_rvv_explicit_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);

// STALE-PROVIDER: candidate tcrv_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-composite-gather-macc-scatter

// STALE-ABI: composite resource runtime ABI order must mirror realized/provider-derived fact
// STALE-ABI-SAME: cmp_lhs,gather_src,rhs_scalar,payload,acc,index,dst,n

// STALE-EXEC-BINDING: candidate tcrv_rvv.exec_abi_bindings provenance must mirror selected tcrv.exec ABI binding summary
// STALE-EXEC-BINDING-SAME: cmp_lhs=lhs-input-buffer->@stale_cmp_lhs_input_buffer

// MISSING-EXEC-MIRROR: candidate metadata must carry tcrv_rvv.exec_abi_bindings provenance
// MISSING-EXEC-MIRROR: tcrv_rvv.exec_abi_bindings_removed

// MISSING-EXEC-BINDING: requires tcrv_rvv.runtime_abi_value 'index' with role 'index-input-buffer' to carry exec_binding to a tcrv.exec ABI declaration

// MISSING-DISPATCH-CASE: 'tcrv.exec.dispatch' op requires at least one tcrv.exec.case

// MISSING-DISPATCH-FALLBACK: 'tcrv.exec.dispatch' op requires exactly one tcrv.exec.fallback

// MISSING-RUNTIME-GUARD: selected dispatch case declares runtime_guard_required=true but does not link runtime_guard to a same-kernel dispatch-availability-guard runtime_param before RVV route construction

// STALE-DISPATCH-CASE: candidate tcrv_rvv.selected_dispatch_case_mirror provenance must mirror selected dispatch case facts
// STALE-DISPATCH-CASE-SAME: selected_dispatch_case_mirror:@stale_rvv_composite

// STALE-DISPATCH-FALLBACK: candidate tcrv_rvv.selected_dispatch_fallback_mirror provenance must mirror selected dispatch fallback facts
// STALE-DISPATCH-FALLBACK-SAME: selected_dispatch_fallback_mirror:@stale_scalar_fallback

// MISSING-DISPATCH-CASE-MIRROR: candidate metadata must carry tcrv_rvv.selected_dispatch_case_mirror provenance

// MISSING-DISPATCH-FALLBACK-MIRROR: candidate metadata must carry tcrv_rvv.selected_dispatch_fallback_mirror provenance

// STALE-COMPOSITE-RESOURCE: candidate tcrv_rvv.composite_resource.selected_candidate provenance must mirror provider-selected composite gather-MAcc-scatter resource selected candidate
// STALE-COMPOSITE-RESOURCE-SAME: artifact-name-derived-composite-resource

// MISSING-COMPOSITE-RESOURCE: requires realized composite resource string fact 'tcrv_rvv.composite_resource.vl_policy' before provider route construction
