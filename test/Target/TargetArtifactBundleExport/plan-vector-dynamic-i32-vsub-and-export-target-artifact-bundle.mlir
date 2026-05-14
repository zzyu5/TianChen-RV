// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vsub-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: rm -rf %t.vector.dynamic.vsub.bundle && mkdir %t.vector.dynamic.vsub.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vsub.bundle %s > %t.vector.dynamic.vsub.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.vector.dynamic.vsub.stdout
// RUN: test -s %t.vector.dynamic.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: test -s %t.vector.dynamic.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: test -s %t.vector.dynamic.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vadd-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vadd-dispatch --implicit-check-not=tcrv_frontend.source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vsub.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_vadd" --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv" --implicit-check-not=i32_vadd --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.vector.dynamic.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/value = "tcrv_rvv.with_vl"/s//value = "descriptor-element-count"/' > %t.vector.dynamic.vsub.stale-vl.planned.mlir
// RUN: rm -rf %t.vector.dynamic.vsub.stale-vl.bundle && mkdir %t.vector.dynamic.vsub.stale-vl.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vsub.stale-vl.bundle %t.vector.dynamic.vsub.stale-vl.planned.mlir 2>&1 | FileCheck %s --check-prefix=STALE-BUNDLE-VL --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.vsub.stale-vl.bundle/tianchenrv-target-artifact-bundle.index
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(?m)^\s*tcrv_rvv\.(?:lowering_boundary|i32_vsub_microkernel attributes)[^\n]*"; attr=r", (?:selected_binary_(?:source_kind|dtype|family|operator|microkernel_op)|emitc_(?:source_op|lowerable_op_interface)) = \"[^\"]*\""; text,count=re.subn(pattern, lambda m: re.sub(attr, "", m.group(0)), text); assert count == 2; sys.stdout.write(text)' > %t.vector.dynamic.vsub.missing-op-source-id.planned.mlir
// RUN: rm -rf %t.vector.dynamic.vsub.missing-op-source-id.bundle && mkdir %t.vector.dynamic.vsub.missing-op-source-id.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vsub.missing-op-source-id.bundle %t.vector.dynamic.vsub.missing-op-source-id.planned.mlir 2>&1 | FileCheck %s --check-prefix=MISSING-OP-SOURCE-ID --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.vsub.missing-op-source-id.bundle/tianchenrv-target-artifact-bundle.index
// RUN: not tcrv-translate --tcrv-export-rvv-microkernel-self-check-c %t.vector.dynamic.vsub.missing-op-source-id.planned.mlir 2>&1 | FileCheck %s --check-prefix=SELF-CHECK-QUARANTINE --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.emitc_arithmetic_intrinsic\"[^}]*value = )\"__riscv_vsub_vv_i32m1\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"__riscv_vadd_vv_i32m1\"", text, count=1); assert count == 1; sys.stdout.write(text)' > %t.vector.dynamic.vsub.stale-emitc-intrinsic.planned.mlir
// RUN: rm -rf %t.vector.dynamic.vsub.stale-emitc-intrinsic.bundle && mkdir %t.vector.dynamic.vsub.stale-emitc-intrinsic.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.vsub.stale-emitc-intrinsic.bundle %t.vector.dynamic.vsub.stale-emitc-intrinsic.planned.mlir 2>&1 | FileCheck %s --check-prefix=STALE-EMITC-INTRINSIC --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.vsub.stale-emitc-intrinsic.bundle/tianchenrv-target-artifact-bundle.index

module @plan_vector_dynamic_i32_vsub_bundle_input {
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

  tcrv.exec.target @vector_dynamic_vsub_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.vector.dynamic.vsub.bundle",
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

  func.func @source_vector_dynamic_bundle_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_bundle_i32_vsub",
        tcrv_frontend_lowering = "i32-vsub",
        tcrv_frontend_target = @vector_dynamic_vsub_bundle_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %diff = arith.subi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %diff, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// IR: tcrv.exec.target @vector_dynamic_vsub_bundle_profile
// IR-SAME: capability_providers = [@no_rvv_policy, @scalar_fallback]
// IR: tcrv.exec.kernel @frontend_vector_dynamic_bundle_i32_vsub
// IR-SAME: target = @vector_dynamic_vsub_bundle_profile
// IR-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// IR-SAME: tcrv_frontend_lowering = "i32-vsub"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// IR-SAME: tcrv_frontend_runtime_extent_arg = "n"
// IR-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vsub.v1"
// IR-SAME: tcrv_frontend_source_loop_step = 16 : i64
// IR-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// IR: tcrv.exec.runtime_param @abi_runtime_element_count
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vsub.v1"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR: tcrv_rvv.i32_vsub_microkernel attributes
// IR-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// IR-SAME: emitc_source_op = "tcrv_rvv.i32_sub"
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_binary_dtype = "i32"
// IR-SAME: selected_binary_family = "i32-vsub"
// IR-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vsub_microkernel"
// IR-SAME: selected_binary_operator = "subtract"
// IR-SAME: selected_binary_source_kind = "frontend-lowering"
// IR-SAME: selected_vector_shape = "i32m1"
// IR: tcrv_rvv.i32_sub
// IR: tcrv_scalar.i32_vsub_microkernel
// IR-SAME: role = "dispatch fallback"
// IR: name = "tcrv_frontend.source_kind"
// IR-SAME: value = "mlir-vector-scf-runtime-i32-vsub.v1"
// IR: name = "tcrv_frontend.active_lane_authority"
// IR-SAME: value = "mlir-vector-transfer-tail-active-lanes"
// IR: name = "tcrv_frontend.source_tail_policy"
// IR-SAME: value = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR: name = "tcrv_rvv.selected_binary_source_kind"
// IR-SAME: value = "frontend-lowering"
// IR: name = "tcrv_rvv.emitc_route_kind"
// IR-SAME: role = "typed-rvv-emitc-route"
// IR-SAME: value = "extension-family-ops-to-emitc-call-opaque"
// IR: name = "tcrv_rvv.emitc_source_authority"
// IR-SAME: value = "mlir-emitc-cpp-emitter"
// IR: name = "tcrv_rvv.emitc_required_header"
// IR-SAME: value = "riscv_vector.h"
// IR: name = "tcrv_rvv.emitc_arithmetic_intrinsic"
// IR-SAME: value = "__riscv_vsub_vv_i32m1"

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-c"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: name: "tcrv_frontend.source_kind"
// INDEX-NEXT: value: "mlir-vector-scf-runtime-i32-vsub.v1"
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
// INDEX-NEXT: value: "tcrv_rvv.i32_vsub_microkernel"
// INDEX-NEXT: role: "typed-rvv-binary-source-identity"
// INDEX: name: "tcrv_rvv.emitc_route_kind"
// INDEX-NEXT: value: "extension-family-ops-to-emitc-call-opaque"
// INDEX-NEXT: role: "typed-rvv-emitc-route"
// INDEX: name: "tcrv_rvv.emitc_source_authority"
// INDEX-NEXT: value: "mlir-emitc-cpp-emitter"
// INDEX-NEXT: role: "typed-rvv-emitc-route"
// INDEX: name: "tcrv_rvv.emitc_required_header"
// INDEX-NEXT: value: "riscv_vector.h"
// INDEX-NEXT: role: "typed-rvv-emitc-route"
// INDEX: name: "tcrv_rvv.emitc_arithmetic_intrinsic"
// INDEX-NEXT: value: "__riscv_vsub_vv_i32m1"
// INDEX-NEXT: role: "typed-rvv-emitc-route"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_element_count_c_name"
// INDEX-NEXT: value: "n"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_vector_config"
// INDEX-NEXT: value: "shape=i32m1,sew=32,lmul=m1,tail_policy=agnostic,mask_policy=agnostic,vector_type=vint32m1_t,vector_suffix=i32m1,setvl_suffix=e32m1"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_vl_boundary"
// INDEX-NEXT: value: "runtime_element_count_c_name=n,runtime_avl_source=runtime-element-count-abi-parameter,runtime_avl_role=runtime-element-count,runtime_vl_source=tcrv_rvv.setvl,runtime_vl_scope=tcrv_rvv.with_vl"
// INDEX: name: "tcrv_rvv.dispatch_contract_descriptor_element_count"
// INDEX-NEXT: value: "16"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_source_identity"
// INDEX-NEXT: value: "source_kind=frontend-lowering,dtype=i32,family=i32-vsub,operator=subtract,microkernel_op=tcrv_rvv.i32_vsub_microkernel,emitc_source_op=tcrv_rvv.i32_sub,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface"
// INDEX: route_claim[4]:
// INDEX-NEXT: name: "descriptor_compute_authority"
// INDEX-NEXT: value: "quarantined-by-selected-rvv-scalar-components"
// INDEX: route_claim[5]:
// INDEX-NEXT: name: "descriptor_config_authority"
// INDEX-NEXT: value: "quarantined-by-dispatch-selected-config-contract"
// INDEX: route_claim[6]:
// INDEX-NEXT: name: "descriptor_runtime_authority"
// INDEX-NEXT: value: "quarantined-runtime-avl-from-ir-backed-abi"
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h"
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o"

// SOURCE: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_vector_dynamic_bundle_i32_vsub */
// SOURCE: /* selected_binary_config: {{.*}}family=i32-vsub
// SOURCE-SAME: runtime_extent_arg=n
// SOURCE-SAME: source_loop_step=16
// SOURCE-SAME: source_vector_chunk_extent=16
// SOURCE-SAME: active_lane_authority=mlir-vector-transfer-tail-active-lanes
// SOURCE-SAME: source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store
// SOURCE-SAME: runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vsub.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// SOURCE: /* dispatch_selected_source_identity: source_kind=frontend-lowering,dtype=i32,family=i32-vsub,operator=subtract,microkernel_op=tcrv_rvv.i32_vsub_microkernel,emitc_source_op=tcrv_rvv.i32_sub,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface */
// SOURCE: /* dispatch_embedded_rvv_artifact_contract_consumed: selected_source_identity=rvv_microkernel_selected_source_identity, runtime_abi_invocation_contract=runtime_abi_invocation_contract, runtime_length=rvv_microkernel_runtime_length_contract, production_owner=rvv-target-export */
// SOURCE: /* dispatch_rvv_emitc_body_mapping: route_kind=extension-family-ops-to-emitc-call-opaque, source_authority=mlir-emitc-cpp-emitter, required_header_metadata=validated-selected-plan-entry, arithmetic_intrinsic_metadata=validated-selected-plan-entry, embedded_rvv_body=selected-rvv-source-artifact */
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_route_kind, value=extension-family-ops-to-emitc-call-opaque, role=typed-rvv-emitc-route
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_source_authority, value=mlir-emitc-cpp-emitter, role=typed-rvv-emitc-route
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_required_header, value=riscv_vector.h, role=typed-rvv-emitc-route
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_arithmetic_intrinsic, value=__riscv_vsub_vv_i32m1, role=typed-rvv-emitc-route
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub
// SOURCE-SAME: runtime_abi_kind=rvv-scalar-dispatch-runtime-callable-c-abi
// SOURCE-SAME: runtime_abi_name=rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1
// SOURCE-SAME: parameter_count=5
// SOURCE-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: dispatch_guard_c_name=rvv_available
// SOURCE-SAME: production_owner=rvv-scalar-dispatch-target
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* arithmetic_family: i32-vsub */
// SOURCE: /* rvv_microkernel_selected_source_identity: source_kind=frontend-lowering,dtype=i32,family=i32-vsub,operator=subtract,microkernel_op=tcrv_rvv.i32_vsub_microkernel,emitc_source_op=tcrv_rvv.i32_sub,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface */
// SOURCE: /* emitc_body_mapping_source: selected_plan_metadata */
// SOURCE: /* emitc_body_mapping: route_kind=extension-family-ops-to-emitc-call-opaque, source_authority=mlir-emitc-cpp-emitter, required_header=riscv_vector.h, arithmetic_intrinsic=__riscv_vsub_vv_i32m1 */
// SOURCE: /* runtime_abi_invocation_contract: source=RVVMicrokernel.cpp, callable_symbol=tcrv_rvv_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_rvv_first_slice
// SOURCE-SAME: runtime_abi_name=rvv-i32-vsub-runtime-callable-c-function.v1
// SOURCE-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: production_owner=rvv-target-export
// SOURCE: __riscv_vsub_vv_i32m1
// SOURCE: tcrv_scalar_i32_sub
// SOURCE: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub

// HEADER: /* dispatch_selected_source_identity: source_kind=frontend-lowering,dtype=i32,family=i32-vsub,operator=subtract,microkernel_op=tcrv_rvv.i32_vsub_microkernel,emitc_source_op=tcrv_rvv.i32_sub,emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface */
// HEADER: /* dispatch_embedded_rvv_artifact_contract_consumed: selected_source_identity=rvv_microkernel_selected_source_identity, runtime_abi_invocation_contract=runtime_abi_invocation_contract, runtime_length=rvv_microkernel_runtime_length_contract, production_owner=rvv-target-export */
// HEADER: /* dispatch_rvv_emitc_body_mapping: route_kind=extension-family-ops-to-emitc-call-opaque, source_authority=mlir-emitc-cpp-emitter, required_header_metadata=validated-selected-plan-entry, arithmetic_intrinsic_metadata=validated-selected-plan-entry, embedded_rvv_body=selected-rvv-source-artifact */
// HEADER: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vsub.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// HEADER: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// HEADER: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub
// HEADER-SAME: runtime_abi_name=rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1
// HEADER-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// HEADER-SAME: runtime_element_count_c_name=n
// HEADER-SAME: dispatch_guard_c_name=rvv_available
// HEADER-SAME: production_owner=rvv-scalar-dispatch-target
// HEADER: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);

// OBJ: Name: tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub

// STALE-BUNDLE-VL: selected_plan_metadata 'tcrv_rvv.runtime_vl_scope'
// STALE-BUNDLE-VL-SAME: must use value 'tcrv_rvv.with_vl'
// MISSING-OP-SOURCE-ID: requires selected RVV binary source identity before target artifact export
// SELF-CHECK-QUARANTINE: requires selected RVV binary source identity before target artifact export
// STALE-EMITC-INTRINSIC: selected_plan_metadata 'tcrv_rvv.emitc_arithmetic_intrinsic' EmitC arithmetic intrinsic must be '__riscv_vsub_vv_i32m1'
