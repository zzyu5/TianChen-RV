// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not=tcrv_rvv.i32_vmul_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i32 --implicit-check-not=__riscv_vsub --implicit-check-not=__riscv_vmul --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline > %t.post-planning.mlir
// RUN: sed '0,/tcrv.exec.variant @rvv_first_slice attributes {/s//tcrv.exec.variant @rvv_first_slice attributes {tcrv_rvv.lowering_descriptor = "i64-vsub-microkernel.v1", /' %t.post-planning.mlir | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RVV-DESCRIPTOR
// RUN: sed '0,/tcrv.exec.variant @scalar_fallback_first_slice attributes {/s//tcrv.exec.variant @scalar_fallback_first_slice attributes {tcrv_scalar.lowering_descriptor = "i64-vsub-microkernel.v1", tcrv_scalar.element_count = 16 : i64, /' %t.post-planning.mlir | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALAR-DESCRIPTOR --implicit-check-not="out[index] = lhs[index] + rhs[index];"

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.capability @no_rvv_policy {
    id = "generic.build.profile",
    kind = "build-policy",
    provides = ["build.policy.no_rvv"],
    status = "available"
  }

  tcrv.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }

  tcrv.exec.target @frontend_rvv_i64_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    conflicts = ["build.policy.no_rvv"],
    id = "rvv.profile.frontend.i64",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    sew_bits = 64 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_i64_vadd(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "frontend_i64_vadd",
        tcrv_frontend_target = @frontend_rvv_i64_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vadd"
      }
      ins(%lhs, %rhs : memref<?xi64>, memref<?xi64>)
      outs(%out : memref<?xi64>) {
    ^bb0(%a: i64, %b: i64, %old: i64):
      %sum = arith.addi %a, %b : i64
      linalg.yield %sum : i64
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_i64_vadd
// LOWER-SAME: target = @frontend_rvv_i64_profile
// LOWER-SAME: tcrv_frontend_lowering = "i64-vadd"
// LOWER: tcrv.exec.mem_window @abi_lhs_input_buffer
// LOWER-SAME: abi_role = "lhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int64_t *"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER: tcrv.exec.mem_window @abi_rhs_input_buffer
// LOWER-SAME: abi_role = "rhs-input-buffer"
// LOWER-SAME: c_type = "const int64_t *"
// LOWER: tcrv.exec.mem_window @abi_output_buffer
// LOWER-SAME: abi_role = "output-buffer"
// LOWER-SAME: access = "write"
// LOWER-SAME: c_type = "int64_t *"
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"

// PIPE-LABEL: tcrv.exec.kernel @frontend_i64_vadd
// PIPE-SAME: target = @frontend_rvv_i64_profile
// PIPE-SAME: tcrv_frontend_lowering = "i64-vadd"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@frontend_rvv_i64_profile]
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-NOT: tcrv_rvv.lowering_descriptor
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.selected_setvl_suffix = "e64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_lmul = "m1"
// PIPE-SAME: tcrv_rvv.selected_vector_sew = 64 : i64
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_suffix = "i64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_type = "vint64m1_t"
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-NOT: tcrv_scalar.lowering_descriptor
// PIPE: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// PIPE-SAME: abi_role = "dispatch-availability-guard"
// PIPE-SAME: c_name = "rvv_available"
// PIPE-SAME: c_type = "int"
// PIPE: tcrv.exec.dispatch
// PIPE: tcrv.exec.case @rvv_first_slice
// PIPE-SAME: runtime_guard = @abi_dispatch_availability_guard
// PIPE-SAME: runtime_guard_required = true
// PIPE: tcrv.exec.fallback @scalar_fallback_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@frontend_rvv_i64_profile]
// PIPE-SAME: role = "dispatch case"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m1"
// PIPE-SAME: selected_vector_sew = 64 : i64
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_type = "vint64m1_t"
// PIPE-SAME: source_kernel = "frontend_i64_vadd"
// PIPE: tcrv_rvv.i64_vadd_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_rvv_i64_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_suffix = "i64m1"
// PIPE-SAME: source_kernel = "frontend_i64_vadd"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 64 : i64
// PIPE: tcrv_rvv.i64_load
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_add
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_store
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.vl
// PIPE: tcrv_scalar.lowering_boundary
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: role = "dispatch fallback"
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE: tcrv_scalar.i64_vadd_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "rvv-explicit-i64-vadd-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i64-vadd-microkernel-c"
// PIPE-SAME: role = "dispatch case"
// PIPE-SAME: runtime_abi = "rvv-i64-vadd-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "rvv-i64-vadd-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_glue_role = "runtime-callable-i64-vadd-function"
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_family"
// PIPE-SAME: role = "typed-rvv-binary-source"
// PIPE-SAME: value = "i64-vadd"
// PIPE-SAME: {name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: value = "tcrv_rvv.i64_add"
// PIPE-SAME: {name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "scalar-explicit-i64-vadd-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-i64-vadd-microkernel-c"
// PIPE-SAME: role = "dispatch fallback"
// PIPE-SAME: runtime_abi = "scalar-i64-vadd-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_name = "scalar-i64-vadd-runtime-callable-c-function.v1"
// PIPE-SAME: {name = "tcrv_scalar.selected_binary_family"
// PIPE-SAME: role = "typed-scalar-binary-source"
// PIPE-SAME: value = "i64-vadd"
// PIPE-SAME: {name = "tcrv_scalar.emitc_source_op"
// PIPE-SAME: value = "tcrv_scalar.i64_vadd_microkernel"
// PIPE-SAME: {name = "tcrv_scalar.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @scalar_fallback_first_slice

// SOURCE: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// SOURCE: /* Scope: one selected RVV i64-vadd dispatch case plus one scalar i64-vadd dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_i64_vadd */
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.selected_binary_family, value=i64-vadd, role=typed-rvv-binary-source
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_source_op, value=tcrv_rvv.i64_add, role=typed-rvv-emitc-source-op
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_lowerable_op_interface, value=TCRVEmitCLowerableOpInterface
// SOURCE: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.selected_binary_family, value=i64-vadd, role=typed-scalar-binary-source
// SOURCE: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.emitc_source_op, value=tcrv_scalar.i64_vadd_microkernel, role=typed-scalar-emitc-source-op
// SOURCE: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.emitc_lowerable_op_interface, value=TCRVEmitCLowerableOpInterface
// SOURCE: /* dispatch_runtime_param[1]: symbol=@abi_dispatch_availability_guard, abi_role=dispatch-availability-guard, c_name=rvv_available, c_type=int, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// SOURCE: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// SOURCE: /* dispatch_fallback_link: target=@scalar_fallback_first_slice, selected_scalar_callable=@scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* dispatch_runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: /* emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// SOURCE: /* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i64_vadd_microkernel_frontend_i64_vadd_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vsetvl_e64m1
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vadd_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1
// SOURCE: void tcrv_rvv_i64_vadd_microkernel_frontend_i64_vadd_rvv_first_slice
// SOURCE: void tcrv_scalar_i64_vadd_microkernel_frontend_i64_vadd_scalar_fallback_first_slice(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n)
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i64_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i64_add
// SOURCE: tcrv_scalar_i64_add
// SOURCE: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// SOURCE: // tcrv_emitc.dispatch_guard_value=rvv_available
// SOURCE-LABEL: {{^}}void tcrv_dispatch_i64_vadd_frontend_i64_vadd
// SOURCE-SAME: (const int64_t* [[LHS:v[0-9]+]], const int64_t* [[RHS:v[0-9]+]], int64_t* [[OUT:v[0-9]+]], size_t [[LEN:v[0-9]+]], int [[GUARD:v[0-9]+]])
// SOURCE-NEXT: {{^}}  bool [[COND:v[0-9]+]] = [[GUARD]] != 0;
// SOURCE-NEXT: {{^}}  if ([[COND]]) {
// SOURCE-NEXT: {{^}}  // tcrv_emitc.source_op=tcrv.exec.case role=dispatch-case-call callee=tcrv_rvv_i64_vadd_microkernel_frontend_i64_vadd_rvv_first_slice
// SOURCE-NEXT: {{^}}  tcrv_rvv_i64_vadd_microkernel_frontend_i64_vadd_rvv_first_slice([[LHS]], [[RHS]], [[OUT]], [[LEN]]);
// SOURCE-NEXT: {{^}}  return;
// SOURCE-NEXT: {{^}}  }
// SOURCE-NEXT: {{^}}  // tcrv_emitc.source_op=tcrv.exec.fallback role=dispatch-fallback-call callee=tcrv_scalar_i64_vadd_microkernel_frontend_i64_vadd_scalar_fallback_first_slice
// SOURCE-NEXT: {{^}}  tcrv_scalar_i64_vadd_microkernel_frontend_i64_vadd_scalar_fallback_first_slice([[LHS]], [[RHS]], [[OUT]], [[LEN]]);

// STALE-RVV-DESCRIPTOR: selected RVV variant @rvv_first_slice failed plugin legality before boundary validation
// STALE-RVV-DESCRIPTOR-SAME: legacy RVV binary descriptor mirror 'i64-vsub-microkernel.v1'
// STALE-RVV-DESCRIPTOR-SAME: typed RVV authority from direct-typed-microkernel-body names family 'i64-vadd'
// STALE-RVV-DESCRIPTOR-SAME: descriptor metadata is non-authoritative mirror metadata

// STALE-SCALAR-DESCRIPTOR: selected scalar dispatch fallback component body authority failed before RVV+scalar dispatch artifact emission
// STALE-SCALAR-DESCRIPTOR-SAME: selected scalar variant @scalar_fallback_first_slice descriptor 'i64-vsub-microkernel.v1' does not match materialized tcrv_scalar.i64_vadd_microkernel
