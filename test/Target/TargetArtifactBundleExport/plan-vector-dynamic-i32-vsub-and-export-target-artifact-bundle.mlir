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
// IR: tcrv_rvv.i32_vsub_microkernel
// IR-SAME: role = "dispatch case"
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
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_element_count_c_name"
// INDEX-NEXT: value: "n"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_vector_config"
// INDEX-NEXT: value: "shape=i32m1,sew=32,lmul=m1,tail_policy=agnostic,mask_policy=agnostic,vector_type=vint32m1_t,vector_suffix=i32m1,setvl_suffix=e32m1"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_vl_boundary"
// INDEX-NEXT: value: "runtime_element_count_c_name=n,runtime_avl_source=runtime-element-count-abi-parameter,runtime_avl_role=runtime-element-count,runtime_vl_source=tcrv_rvv.setvl,runtime_vl_scope=tcrv_rvv.with_vl"
// INDEX: name: "tcrv_rvv.dispatch_contract_descriptor_element_count"
// INDEX-NEXT: value: "16"
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
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* arithmetic_family: i32-vsub */
// SOURCE: __riscv_vsub_vv_i32m1
// SOURCE: tcrv_scalar_i32_sub
// SOURCE: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub

// HEADER: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vsub.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// HEADER: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// HEADER: void tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);

// OBJ: Name: tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub

// STALE-BUNDLE-VL: selected_plan_metadata 'tcrv_rvv.runtime_vl_scope'
// STALE-BUNDLE-VL-SAME: must use value 'tcrv_rvv.with_vl'
