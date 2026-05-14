// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vsub_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: rm -rf %t.vector.dynamic.vmul.bundle && mkdir %t.vector.dynamic.vmul.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vmul.bundle %s > %t.vector.dynamic.vmul.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.vector.dynamic.vmul.stdout
// RUN: test -s %t.vector.dynamic.vmul.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c
// RUN: test -s %t.vector.dynamic.vmul.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h
// RUN: test -s %t.vector.dynamic.vmul.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vadd-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vsub-dispatch-external-abi.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vadd-dispatch --implicit-check-not=tcrv-export-rvv-scalar-i32-vsub-dispatch --implicit-check-not=tcrv_frontend.source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vmul.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_vadd" --implicit-check-not="__riscv_vsub" --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vmul.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv" --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vmul.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.vector.dynamic.vmul.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.emitc_arithmetic_intrinsic\"[^}]*value = )\"__riscv_vmul_vv_i32m1\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"__riscv_vadd_vv_i32m1\"", text, count=1); assert count == 1; sys.stdout.write(text)' > %t.vector.dynamic.vmul.stale-emitc-intrinsic.planned.mlir
// RUN: rm -rf %t.vector.dynamic.vmul.stale-emitc-intrinsic.bundle && mkdir %t.vector.dynamic.vmul.stale-emitc-intrinsic.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vmul.stale-emitc-intrinsic.bundle %t.vector.dynamic.vmul.stale-emitc-intrinsic.planned.mlir 2>&1 | FileCheck %s --check-prefix=STALE-EMITC-INTRINSIC --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.vmul.stale-emitc-intrinsic.bundle/tianchenrv-target-artifact-bundle.index

module @plan_vector_dynamic_i32_vmul_bundle_input {
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

  tcrv.exec.target @vector_dynamic_vmul_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.vector.dynamic.vmul.bundle",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vector_dynamic_bundle_vmul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_bundle_i32_vmul",
        tcrv_frontend_lowering = "i32-vmul",
        tcrv_frontend_target = @vector_dynamic_vmul_bundle_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %product = arith.muli %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %product, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// IR: tcrv.exec.target @vector_dynamic_vmul_bundle_profile
// IR-SAME: capability_providers = [@no_rvv_policy, @scalar_fallback]
// IR: tcrv.exec.kernel @frontend_vector_dynamic_bundle_i32_vmul
// IR-SAME: target = @vector_dynamic_vmul_bundle_profile
// IR-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// IR-SAME: tcrv_frontend_lowering = "i32-vmul"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// IR-SAME: tcrv_frontend_runtime_extent_arg = "n"
// IR-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vmul.v1"
// IR-SAME: tcrv_frontend_source_loop_step = 16 : i64
// IR-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR: tcrv_rvv.i32_vmul_microkernel attributes
// IR-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// IR-SAME: emitc_source_op = "tcrv_rvv.i32_mul"
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_binary_dtype = "i32"
// IR-SAME: selected_binary_family = "i32-vmul"
// IR-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vmul_microkernel"
// IR-SAME: selected_binary_operator = "multiply"
// IR-SAME: selected_binary_source_kind = "frontend-lowering"
// IR-SAME: selected_vector_shape = "i32m1"
// IR: tcrv_rvv.i32_mul
// IR: tcrv_scalar.i32_vmul_microkernel
// IR-SAME: role = "dispatch fallback"
// IR: name = "tcrv_frontend.source_kind"
// IR-SAME: value = "mlir-vector-scf-runtime-i32-vmul.v1"
// IR: name = "tcrv_rvv.selected_binary_source_kind"
// IR-SAME: value = "frontend-lowering"
// IR: name = "tcrv_rvv.emitc_arithmetic_intrinsic"
// IR-SAME: value = "__riscv_vmul_vv_i32m1"

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vmul-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vmul-dispatch-external-abi.v1"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vmul-dispatch-c"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"
// INDEX: name: "tcrv_frontend.source_kind"
// INDEX-NEXT: value: "mlir-vector-scf-runtime-i32-vmul.v1"
// INDEX-NEXT: role: "source-frontdoor-runtime-avl-authority"
// INDEX: name: "tcrv_frontend.source_authority"
// INDEX-NEXT: value: "source-scf-for-runtime-upper-bound"
// INDEX: name: "tcrv_frontend.runtime_extent_arg"
// INDEX-NEXT: value: "n"
// INDEX: name: "tcrv_frontend.active_lane_authority"
// INDEX-NEXT: value: "mlir-vector-transfer-tail-active-lanes"
// INDEX: name: "tcrv_frontend.source_tail_policy"
// INDEX-NEXT: value: "runtime-n-bounded-transfer-tail-padding-and-store"
// INDEX: name: "tcrv_frontend.runtime_element_count_constraint"
// INDEX-NEXT: value: "source-runtime-extent"
// INDEX: name: "tcrv_rvv.selected_binary_source_kind"
// INDEX-NEXT: value: "frontend-lowering"
// INDEX-NEXT: role: "typed-rvv-binary-source-identity"
// INDEX: name: "tcrv_rvv.selected_binary_microkernel_op"
// INDEX-NEXT: value: "tcrv_rvv.i32_vmul_microkernel"
// INDEX-NEXT: role: "typed-rvv-binary-source-identity"
// INDEX: name: "tcrv_rvv.emitc_arithmetic_intrinsic"
// INDEX-NEXT: value: "__riscv_vmul_vv_i32m1"
// INDEX-NEXT: role: "typed-rvv-emitc-route"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_source_identity"
// INDEX-NEXT: value: "source_kind=frontend-lowering,dtype=i32,family=i32-vmul,operator=multiply,microkernel_op=tcrv_rvv.i32_vmul_microkernel,emitc_source_op=tcrv_rvv.i32_mul,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface"
// INDEX: route_claim[4]:
// INDEX-NEXT: name: "descriptor_compute_authority"
// INDEX-NEXT: value: "quarantined-by-selected-rvv-scalar-components"
// INDEX: route_claim[5]:
// INDEX-NEXT: name: "descriptor_config_authority"
// INDEX-NEXT: value: "quarantined-by-dispatch-selected-config-contract"
// INDEX: route_claim[6]:
// INDEX-NEXT: name: "descriptor_runtime_authority"
// INDEX-NEXT: value: "quarantined-runtime-avl-from-ir-backed-abi"
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vmul-dispatch-header.h"
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vmul-dispatch-object.o"

// SOURCE: /* Scope: one selected RVV i32-vmul dispatch case plus one scalar i32-vmul dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_vector_dynamic_bundle_i32_vmul */
// SOURCE: /* selected_binary_config: {{.*}}family=i32-vmul
// SOURCE-SAME: runtime_extent_arg=n
// SOURCE-SAME: source_loop_step=16
// SOURCE-SAME: source_vector_chunk_extent=16
// SOURCE-SAME: active_lane_authority=mlir-vector-transfer-tail-active-lanes
// SOURCE-SAME: source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store
// SOURCE-SAME: runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vmul.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// SOURCE: /* dispatch_selected_source_identity: source_kind=frontend-lowering,dtype=i32,family=i32-vmul,operator=multiply,microkernel_op=tcrv_rvv.i32_vmul_microkernel,emitc_source_op=tcrv_rvv.i32_mul,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface */
// SOURCE: /* dispatch_embedded_rvv_artifact_contract_consumed: selected_source_identity=rvv_microkernel_selected_source_identity, runtime_abi_invocation_contract=runtime_abi_invocation_contract, runtime_length=rvv_microkernel_runtime_length_contract, production_owner=rvv-target-export */
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_arithmetic_intrinsic, value=__riscv_vmul_vv_i32m1, role=typed-rvv-emitc-route
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vmul_microkernel_frontend_vector_dynamic_bundle_i32_vmul_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vmul_microkernel_frontend_vector_dynamic_bundle_i32_vmul_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vmul_frontend_vector_dynamic_bundle_i32_vmul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vmul_frontend_vector_dynamic_bundle_i32_vmul
// SOURCE-SAME: family=i32-vmul
// SOURCE-SAME: runtime_abi_name=rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1
// SOURCE-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// SOURCE-SAME: dispatch_guard_c_name=rvv_available
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vmul_microkernel */
// SOURCE: /* arithmetic_family: i32-vmul */
// SOURCE: /* runtime_abi_invocation_contract: source=RVVMicrokernel.cpp, callable_symbol=tcrv_rvv_i32_vmul_microkernel_frontend_vector_dynamic_bundle_i32_vmul_rvv_first_slice
// SOURCE-SAME: family=i32-vmul
// SOURCE-SAME: runtime_abi_name=rvv-i32-vmul-runtime-callable-c-function.v1
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: production_owner=rvv-target-export
// SOURCE: __riscv_vmul_vv_i32m1
// SOURCE: tcrv_scalar_i32_mul
// HEADER: void tcrv_dispatch_i32_vmul_frontend_vector_dynamic_bundle_i32_vmul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// OBJ: Format: elf64-littleriscv
// OBJ: tcrv_dispatch_i32_vmul_frontend_vector_dynamic_bundle_i32_vmul
// STALE-EMITC-INTRINSIC: selected_plan_metadata 'tcrv_rvv.emitc_arithmetic_intrinsic'
// STALE-EMITC-INTRINSIC-SAME: EmitC arithmetic intrinsic must be '__riscv_vmul_vv_i32m1'
