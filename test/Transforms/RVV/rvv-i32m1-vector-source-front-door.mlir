// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-vector-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="func.func"
// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-vector-source-front-door --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-rvv-materialize-i32m1-vector-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN

module {
  func.func @vector_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %remaining = arith.subi %n, %i : index
      %mask = vector.create_mask %remaining : vector<4xi1>
      %a = vector.transfer_read %lhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %b = vector.transfer_read %rhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.transfer_write %sum, %out[%i], %mask : vector<4xi32>, memref<?xi32>
    }
    return
  }
}

// BOUNDARY-LABEL: tcrv.exec.kernel @vector_source_kernel
// BOUNDARY: tcrv.exec.capability @rvv
// BOUNDARY-SAME: id = "rvv"
// BOUNDARY-SAME: kind = "isa-vector"
// BOUNDARY: tcrv.exec.capability @scalar_fallback
// BOUNDARY-SAME: id = "scalar.fallback"
// BOUNDARY-SAME: kind = "fallback"
// BOUNDARY-LABEL: tcrv.exec.variant @vector_source_rvv_i32_add
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: requires = [@rvv]
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "lhs"
// BOUNDARY-SAME: c_type = "const int32_t *"
// BOUNDARY-SAME: purpose = "source-arg-0:lhs"
// BOUNDARY-SAME: role = "lhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "rhs"
// BOUNDARY-SAME: purpose = "source-arg-1:rhs"
// BOUNDARY-SAME: role = "rhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "out"
// BOUNDARY-SAME: purpose = "source-arg-2:out"
// BOUNDARY-SAME: role = "output-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "n"
// BOUNDARY-SAME: c_type = "size_t"
// BOUNDARY-SAME: purpose = "source-arg-3:n"
// BOUNDARY-SAME: role = "runtime-element-count"
// BOUNDARY: tcrv_rvv.setvl
// BOUNDARY-SAME: lmul = "m1"
// BOUNDARY-SAME: sew = 32 : i64
// BOUNDARY: tcrv_rvv.with_vl
// BOUNDARY-SAME: lmul = "m1"
// BOUNDARY-SAME: sew = 32 : i64
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_add
// BOUNDARY: tcrv_rvv.i32_store
// BOUNDARY: tcrv.exec.variant @vector_source_scalar_fallback
// BOUNDARY-SAME: fallback_role = "conservative"
// BOUNDARY-SAME: origin = "scalar-plugin"
// BOUNDARY-SAME: requires = [@scalar_fallback]
// BOUNDARY: tcrv.exec.dispatch
// BOUNDARY: tcrv.exec.case @vector_source_rvv_i32_add
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: policy = "source-pattern-selected-rvv-case"
// BOUNDARY: tcrv.exec.fallback @vector_source_scalar_fallback
// BOUNDARY-SAME: fallback_role = "conservative"
// BOUNDARY-SAME: origin = "scalar-plugin"

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: artifact_metadata = [{key = "rvv_emitc_lowerable_route", value = "rvv-i32m1-add-emitc-route"}
// PLAN-SAME: {key = "rvv_arithmetic_op", value = "add"}
// PLAN-SAME: {key = "rvv_source_ops", value = "tcrv_rvv.runtime_abi_value->tcrv_rvv.setvl->tcrv_rvv.with_vl->tcrv_rvv.i32_load->tcrv_rvv.i32_load->tcrv_rvv.i32_arithmetic->tcrv_rvv.i32_store"}
// PLAN-SAME: {key = "rvv_source_roles", value = "runtime_abi->configure->scope->load->load->compute->store"}
// PLAN-SAME: {key = "rvv_source_op_interface", value = "TCRVEmitCLowerableOpInterface"}
// PLAN-SAME: {key = "rvv_construction_protocol", value = "extension-family-construction-protocol.v1"}
// PLAN-SAME: {key = "rvv_extension_archetype", value = "rvv-finite-binary"}
// PLAN-SAME: {key = "rvv_semantic_role_graph", value = "runtime_abi->configure->scope->load->compute->store"}
// PLAN-SAME: {key = "rvv_common_interface_realization", value = "runtime_abi/resource+emitc
// PLAN-SAME: {key = "rvv_typed_role_realization", value = "runtime_abi:tcrv_rvv.runtime_abi_value
// PLAN-SAME: {key = "rvv_emitc_route_mapping", value = "rvv-i32m1-arithmetic-emitc-route-family"}
// PLAN-SAME: {key = "rvv_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_target_artifact|ssh_rvv_required_for_runtime_claims"}
// PLAN-SAME: {key = "rvv_runtime_abi_contract", value = "rvv-i32m1-arithmetic-callable-c-abi-family.v1"}
// PLAN-SAME: {key = "rvv_bundle_component_group", value = "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1"}
// PLAN-SAME: {key = "rvv_object_handoff", value = "materialized-emitc-cpp-rvv-intrinsic-object"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.vl_def", value = "tcrv_rvv.setvl"}
// PLAN-SAME: {key = "tcrv_rvv.vl_scope", value = "tcrv_rvv.with_vl"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_avl_abi_parameter", value = "n"}
// PLAN-SAME: {key = "tcrv_rvv.emitc_loop", value = "emitc.for"}
// PLAN-SAME: {key = "tcrv_rvv.loop_induction", value = "offset"}
// PLAN-SAME: {key = "tcrv_rvv.loop_step", value = "full_chunk_vl"}
// PLAN-SAME: {key = "tcrv_rvv.remaining_avl", value = "n-offset"}
// PLAN-SAME: {key = "tcrv_rvv.pointer_advance", value = "offset"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-i32m1-arithmetic"}
// PLAN-SAME: {key = "tcrv_rvv.multi_vl", value = "supported"}]
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "rvv-i32m1-arithmetic-emitc-route-family"
// PLAN-SAME: message = "RVV selected i32m1 arithmetic route materializes
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: required_capabilities = [@rvv]
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @vector_source_rvv_i32_add
// PLAN: tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic"
// PLAN-SAME: emission_kind = "scalar-fallback-unsupported-emission"
// PLAN-SAME: origin = "scalar-plugin"
// PLAN-SAME: role = "dispatch fallback"
// PLAN-SAME: status = "unsupported"
// PLAN-SAME: target = @vector_source_scalar_fallback

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-rvv-materialize-i32m1-vector-source-front-door
