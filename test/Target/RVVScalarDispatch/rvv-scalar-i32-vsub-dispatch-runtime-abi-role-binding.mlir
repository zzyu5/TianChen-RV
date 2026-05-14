// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c | FileCheck %s --check-prefix=ROLE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "guard", c_type = "int", ownership = "target-export-abi-owned", role = "dispatch-availability-guard"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed 's|{name = "tcrv_rvv.runtime_element_count_c_name", note = "runtime ABI/control C name resolved from tcrv.exec runtime boundary; not a compile-time vector-shape config or descriptor element_count", role = "rvv-runtime-control-name-boundary", value = "n"}|{name = "tcrv_rvv.runtime_element_count_c_name", note = "runtime ABI/control C name resolved from tcrv.exec runtime boundary; not a compile-time vector-shape config or descriptor element_count", role = "rvv-runtime-control-name-boundary", value = "len"}|' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=STALE-RVV-CONTRACT --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"/s/c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"/c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=DUPLICATE --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "long", ownership = "target-export-abi-owned", role = "runtime-element-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=WRONG-TYPE --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/s/c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/c_name = "lhs", c_type = "const int32_t *", ownership = "ir-modeled", role = "lhs-input-buffer"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=WRONG-OWNERSHIP --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "mystery-runtime-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=UNKNOWN --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed 's/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int"/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "bool"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=GUARD-TYPE --implicit-check-not="void tcrv_dispatch_i32_vsub"
// RUN: tcrv-opt %S/rvv-scalar-i32-vsub-dispatch-generic-route.mlir --tcrv-lower-linalg-i32-binary-to-exec --tcrv-execution-planning-pipeline | sed 's/tcrv.exec.dispatch {/tcrv.exec.dispatch attributes { tcrv_rvv_scalar.dispatch_runtime_abi_parameters = [{c_name = "rvv_ready", c_type = "int", ownership = "target-export-abi-owned", role = "dispatch-availability-guard"}] } {/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=DETACHED --implicit-check-not="void tcrv_dispatch_i32_vsub"

// ROLE: /* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
// ROLE: /* selected_binary_config: dtype=i32, family=i32-vsub
// ROLE-SAME: runtime_element_count_c_name=n
// ROLE-SAME: dispatch_availability_c_name=rvv_available
// ROLE-SAME: descriptor_element_count=16
// ROLE-SAME: selected_role=dispatch case */
// ROLE: /* rvv_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* rvv_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* scalar_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* scalar_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=n, c_type=size_t, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// ROLE: /* dispatch_runtime_param[1]: symbol=@abi_dispatch_availability_guard, abi_role=dispatch-availability-guard, c_name=rvv_available, c_type=int, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// ROLE: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// ROLE: /* dispatch_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vsub_frontend_dispatch_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// ROLE: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vsub_frontend_dispatch_i32_vsub
// ROLE-SAME: runtime_abi_name=rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1
// ROLE-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// ROLE-SAME: runtime_element_count_c_name=n
// ROLE-SAME: dispatch_guard_c_name=rvv_available
// ROLE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// ROLE: __riscv_vsub_vv_i32m1
// ROLE: void tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice
// ROLE: void tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice
// ROLE: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// ROLE-LABEL: {{^}}void tcrv_dispatch_i32_vsub_frontend_dispatch_i32_vsub
// ROLE-SAME: (const int32_t* [[LHS:v[0-9]+]], const int32_t* [[RHS:v[0-9]+]], int32_t* [[OUT:v[0-9]+]], size_t [[LEN:v[0-9]+]], int [[READY:v[0-9]+]])
// ROLE-NEXT: {{^}}  bool [[COND:v[0-9]+]] = [[READY]] != 0;
// ROLE-NEXT: {{^}}  if ([[COND]]) {
// ROLE-NEXT: {{^}}  // tcrv_emitc.source_op=tcrv.exec.case role=dispatch-case-call callee=tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice
// ROLE-NEXT: {{^}}  tcrv_rvv_i32_vsub_microkernel_frontend_dispatch_i32_vsub_rvv_first_slice([[LHS]], [[RHS]], [[OUT]], [[LEN]]);
// ROLE-NEXT: {{^}}  return;
// ROLE-NEXT: {{^}}  }
// ROLE-NEXT: {{^}}  // tcrv_emitc.source_op=tcrv.exec.fallback role=dispatch-fallback-call callee=tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice
// ROLE-NEXT: {{^}}  tcrv_scalar_i32_vsub_microkernel_frontend_dispatch_i32_vsub_scalar_fallback_first_slice([[LHS]], [[RHS]], [[OUT]], [[LEN]]);

// MISSING: route id 'tcrv-export-rvv-i32-vsub-microkernel-c' requires structured runtime ABI parameter role 'runtime-element-count'

// STALE-RVV-CONTRACT: target artifact candidate validation failed
// STALE-RVV-CONTRACT-SAME: selected_plan_metadata 'tcrv_rvv.runtime_element_count_c_name'
// STALE-RVV-CONTRACT-SAME: must be 'n'

// DUPLICATE: duplicate runtime ABI parameter role 'lhs-input-buffer'

// WRONG-TYPE: route id 'tcrv-export-rvv-i32-vsub-microkernel-c' runtime ABI parameter role 'runtime-element-count' must use c type 'size_t'

// WRONG-OWNERSHIP: route id 'tcrv-export-rvv-i32-vsub-microkernel-c' runtime ABI parameter role 'lhs-input-buffer' must use c type 'const int32_t *' and ownership 'target-export-abi-owned'

// UNKNOWN: unsupported runtime ABI parameter role 'mystery-runtime-count'

// GUARD-TYPE: RVV+scalar binary dispatch C export failed
// GUARD-TYPE-SAME: runtime ABI runtime_param validation failed
// GUARD-TYPE-SAME: tcrv.exec.runtime_param @abi_dispatch_availability_guard requires attribute 'c_type' = "int"

// DETACHED: RVV+scalar binary dispatch C export failed
// DETACHED-SAME: tcrv_rvv_scalar.dispatch_runtime_abi_parameters is detached dispatch ABI metadata
// DETACHED-SAME: use direct tcrv.exec.runtime_param IR
