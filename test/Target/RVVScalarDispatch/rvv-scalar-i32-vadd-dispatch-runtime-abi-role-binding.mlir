// RUN: sed 's/tcrv.exec.dispatch {/tcrv.exec.dispatch attributes { tcrv_rvv_scalar.dispatch_runtime_abi_parameters = [{c_name = "rvv_ready", c_type = "int", ownership = "target-export-abi-owned", role = "dispatch-availability-guard"}] } {/' %S/rvv-scalar-i32-vadd-dispatch-c.mlir | tcrv-opt - --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed -e 's/c_name = "lhs"/c_name = "a"/g' -e 's/c_name = "rhs"/c_name = "b"/g' -e 's/c_name = "out"/c_name = "dst"/g' -e 's/c_name = "n"/c_name = "len"/g' | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c | FileCheck %s --check-prefix=ROLE --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %S/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "guard", c_type = "int", ownership = "target-export-abi-owned", role = "dispatch-availability-guard"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %S/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"/s/c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"/c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=DUPLICATE --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %S/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "long", ownership = "target-export-abi-owned", role = "runtime-element-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=WRONG-TYPE --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %S/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/s/c_name = "lhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "lhs-input-buffer"/c_name = "lhs", c_type = "const int32_t *", ownership = "ir-modeled", role = "lhs-input-buffer"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=WRONG-OWNERSHIP --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %S/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/s/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "mystery-runtime-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=UNKNOWN --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: sed 's/tcrv.exec.dispatch {/tcrv.exec.dispatch attributes { tcrv_rvv_scalar.dispatch_runtime_abi_parameters = [{c_name = "rvv_ready", c_type = "bool", ownership = "target-export-abi-owned", role = "dispatch-availability-guard"}] } {/' %S/rvv-scalar-i32-vadd-dispatch-c.mlir | tcrv-opt - --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=GUARD-TYPE --implicit-check-not="void tcrv_dispatch_i32_vadd"

// ROLE: /* Runtime guard: explicit host-provided rvv_ready parameter; no automatic hardware probe is generated. */
// ROLE: /* rvv_runtime_abi_parameter[0]: c_name=a, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* rvv_runtime_abi_parameter[3]: c_name=len, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* scalar_runtime_abi_parameter[0]: c_name=a, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* scalar_runtime_abi_parameter[3]: c_name=len, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[0]: c_name=a, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[1]: c_name=b, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[2]: c_name=dst, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[3]: c_name=len, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_ready, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// ROLE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *a, const int32_t *b, int32_t *dst, size_t len, int rvv_ready) */
// ROLE: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice(const int32_t *a, const int32_t *b, int32_t *dst, size_t len)
// ROLE: while (offset < len)
// ROLE: __riscv_vle32_v_i32m1(&a[offset], vl)
// ROLE: __riscv_vse32_v_i32m1(&dst[offset], sum_vec, vl)
// ROLE: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)
// ROLE-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *a, const int32_t *b, int32_t *dst, size_t len, int rvv_ready)
// ROLE-NEXT: {{^}}  if (rvv_ready) {
// ROLE-NEXT: {{^}}    tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice(a, b, dst, len);
// ROLE-NEXT: {{^}}    return;
// ROLE-NEXT: {{^}}  }
// ROLE-NEXT: {{^}}  tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice(a, b, dst, len);

// MISSING: RVV+scalar i32-vadd dispatch C export failed
// MISSING-SAME: selected callable artifact route 'tcrv-export-rvv-microkernel-c' requires runtime ABI parameter role 'runtime-element-count'

// DUPLICATE: target source artifact export failed
// DUPLICATE-SAME: duplicate runtime ABI parameter role 'lhs-input-buffer'

// WRONG-TYPE: RVV+scalar i32-vadd dispatch C export failed
// WRONG-TYPE-SAME: selected callable artifact route 'tcrv-export-rvv-microkernel-c' runtime ABI parameter role 'runtime-element-count' must use c type 'size_t'

// WRONG-OWNERSHIP: RVV+scalar i32-vadd dispatch C export failed
// WRONG-OWNERSHIP-SAME: selected callable artifact route 'tcrv-export-rvv-microkernel-c' runtime ABI parameter role 'lhs-input-buffer' must use ownership 'target-export-abi-owned'

// UNKNOWN: target source artifact export failed
// UNKNOWN-SAME: unsupported runtime ABI parameter role 'mystery-runtime-count'

// GUARD-TYPE: RVV+scalar i32-vadd dispatch C export failed
// GUARD-TYPE-SAME: dispatch availability guard runtime ABI parameter must use c type 'int'
