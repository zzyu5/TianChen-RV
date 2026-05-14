// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: rm -rf %t.vector.dynamic.bundle && mkdir %t.vector.dynamic.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.bundle %s > %t.vector.dynamic.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.vector.dynamic.stdout
// RUN: test -s %t.vector.dynamic.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.vector.dynamic.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.vector.dynamic.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=tcrv_frontend.source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password < %t.vector.dynamic.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.vector.dynamic.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"
// RUN: rm -rf %t.vector.dynamic.stale_source_identity.bundle && mkdir %t.vector.dynamic.stale_source_identity.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.selected_binary_source_kind\"[^}]*value = )\"frontend-lowering\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"descriptor-only\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.stale_source_identity.bundle 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE-ID --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.stale_source_identity.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.vector.dynamic.missing_source_identity.bundle && mkdir %t.vector.dynamic.missing_source_identity.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"name = \"tcrv_rvv\.selected_binary_source_kind\""; text,count=re.subn(pattern, "name = \"tcrv_rvv.stale_selected_binary_source_kind\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.missing_source_identity.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-SOURCE-ID --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.missing_source_identity.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.vector.dynamic.stale_selected_config.bundle && mkdir %t.vector.dynamic.stale_selected_config.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.selected_vector_lmul\"[^}]*value = )\"m1\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"m2\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.stale_selected_config.bundle 2>&1 | FileCheck %s --check-prefix=STALE-SELECTED-CONFIG --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.stale_selected_config.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.vector.dynamic.missing_selected_config.bundle && mkdir %t.vector.dynamic.missing_selected_config.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"name = \"tcrv_rvv\.selected_vector_lmul\""; text,count=re.subn(pattern, "name = \"tcrv_rvv.stale_selected_vector_lmul\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.missing_selected_config.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-SELECTED-CONFIG --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.missing_selected_config.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.vector.dynamic.stale_runtime_length.bundle && mkdir %t.vector.dynamic.stale_runtime_length.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.runtime_avl_source\"[^}]*value = )\"runtime-element-count-abi-parameter\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"descriptor-element-count\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.stale_runtime_length.bundle 2>&1 | FileCheck %s --check-prefix=STALE-RUNTIME-LENGTH --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.stale_runtime_length.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.vector.dynamic.missing_runtime_length.bundle && mkdir %t.vector.dynamic.missing_runtime_length.bundle
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"name = \"tcrv_rvv\.runtime_vl_source\""; text,count=re.subn(pattern, "name = \"tcrv_rvv.stale_runtime_vl_source\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.vector.dynamic.missing_runtime_length.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-LENGTH --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: test ! -f %t.vector.dynamic.missing_runtime_length.bundle/tianchenrv-target-artifact-bundle.index

module @plan_vector_dynamic_i32_vadd_bundle_input {
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

  tcrv.exec.target @vector_dynamic_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.vector.dynamic.bundle",
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

  func.func @source_vector_dynamic_bundle_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_bundle_i32_vadd",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_dynamic_bundle_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %sum, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// IR: tcrv.exec.target @vector_dynamic_bundle_profile
// IR-SAME: capability_providers = [@no_rvv_policy, @scalar_fallback]
// IR: tcrv.exec.kernel @frontend_vector_dynamic_bundle_i32_vadd
// IR-SAME: target = @vector_dynamic_bundle_profile
// IR-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// IR-SAME: tcrv_frontend_runtime_extent_arg = "n"
// IR-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vadd.v1"
// IR-SAME: tcrv_frontend_source_loop_step = 16 : i64
// IR-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// IR: tcrv.exec.runtime_param @abi_runtime_element_count
// IR-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// IR-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// IR-SAME: tcrv_frontend_runtime_extent_arg = "n"
// IR-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// IR-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vadd.v1"
// IR-SAME: tcrv_frontend_source_loop_step = 16 : i64
// IR-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR: tcrv_rvv.i32_vadd_microkernel attributes
// IR-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// IR-SAME: emitc_source_op = "tcrv_rvv.i32_add"
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_binary_dtype = "i32"
// IR-SAME: selected_binary_family = "i32-vadd"
// IR-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel"
// IR-SAME: selected_binary_operator = "add"
// IR-SAME: selected_binary_source_kind = "frontend-lowering"
// IR-SAME: selected_vector_shape = "i32m1"
// IR: tcrv_scalar.i32_vadd_microkernel
// IR-SAME: role = "dispatch fallback"
// IR: name = "tcrv_frontend.source_kind"
// IR-SAME: value = "mlir-vector-scf-runtime-i32-vadd.v1"
// IR: name = "tcrv_frontend.source_authority"
// IR-SAME: value = "source-scf-for-runtime-upper-bound"
// IR: name = "tcrv_frontend.runtime_extent_arg"
// IR-SAME: value = "n"
// IR: name = "tcrv_frontend.source_loop_step"
// IR-SAME: value = "16"
// IR: name = "tcrv_frontend.source_vector_chunk_extent"
// IR-SAME: value = "16"
// IR: name = "tcrv_frontend.active_lane_authority"
// IR-SAME: value = "mlir-vector-transfer-tail-active-lanes"
// IR: name = "tcrv_frontend.source_tail_policy"
// IR-SAME: value = "runtime-n-bounded-transfer-tail-padding-and-store"
// IR: name = "tcrv_rvv.selected_binary_source_kind"
// IR-SAME: role = "typed-rvv-binary-source-identity"
// IR-SAME: value = "frontend-lowering"
// IR: name = "tcrv_rvv.selected_binary_microkernel_op"
// IR-SAME: role = "typed-rvv-binary-source-identity"
// IR-SAME: value = "tcrv_rvv.i32_vadd_microkernel"

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: name: "tcrv_frontend.source_kind"
// INDEX-NEXT: value: "mlir-vector-scf-runtime-i32-vadd.v1"
// INDEX-NEXT: role: "source-frontdoor-runtime-avl-authority"
// INDEX: name: "tcrv_frontend.source_authority"
// INDEX-NEXT: value: "source-scf-for-runtime-upper-bound"
// INDEX: name: "tcrv_frontend.runtime_extent_arg"
// INDEX-NEXT: value: "n"
// INDEX: name: "tcrv_frontend.source_loop_step"
// INDEX-NEXT: value: "16"
// INDEX: name: "tcrv_frontend.source_vector_chunk_extent"
// INDEX-NEXT: value: "16"
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
// INDEX-NEXT: value: "tcrv_rvv.i32_vadd_microkernel"
// INDEX-NEXT: role: "typed-rvv-binary-source-identity"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_element_count_c_name"
// INDEX-NEXT: value: "n"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_vector_config"
// INDEX-NEXT: value: "shape=i32m1,sew=32,lmul=m1,tail_policy=agnostic,mask_policy=agnostic,vector_type=vint32m1_t,vector_suffix=i32m1,setvl_suffix=e32m1"
// INDEX: name: "tcrv_rvv.dispatch_contract_runtime_vl_boundary"
// INDEX-NEXT: value: "runtime_element_count_c_name=n,runtime_avl_source=runtime-element-count-abi-parameter,runtime_avl_role=runtime-element-count,runtime_vl_source=tcrv_rvv.setvl,runtime_vl_scope=tcrv_rvv.with_vl"
// INDEX: name: "tcrv_rvv.dispatch_contract_descriptor_element_count"
// INDEX-NEXT: value: "16"
// INDEX: name: "tcrv_rvv.dispatch_contract_selected_source_identity"
// INDEX-NEXT: value: "source_kind=frontend-lowering,family=i32-vadd,microkernel_op=tcrv_rvv.i32_vadd_microkernel"
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"

// SOURCE: /* selected_kernel: @frontend_vector_dynamic_bundle_i32_vadd */
// SOURCE: /* selected_binary_config: {{.*}}descriptor_element_count=16, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vadd.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.selected_binary_source_kind, value=frontend-lowering, role=typed-rvv-binary-source-identity
// SOURCE: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.selected_binary_microkernel_op, value=tcrv_rvv.i32_vadd_microkernel, role=typed-rvv-binary-source-identity
// SOURCE: /* dispatch_runtime_callable_abi: void tcrv_dispatch_i32_vadd_frontend_vector_dynamic_bundle_i32_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available) */
// SOURCE: void tcrv_dispatch_i32_vadd_frontend_vector_dynamic_bundle_i32_vadd

// HEADER: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vadd.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// HEADER: /* dispatch_runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted before dispatch */
// HEADER: void tcrv_dispatch_i32_vadd_frontend_vector_dynamic_bundle_i32_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);

// OBJ: Name: tcrv_dispatch_i32_vadd_frontend_vector_dynamic_bundle_i32_vadd

// STALE-SOURCE-ID: selected_plan_metadata 'tcrv_rvv.selected_binary_source_kind' selected binary source kind must be 'frontend-lowering'
// MISSING-SOURCE-ID: requires selected_plan_metadata 'tcrv_rvv.selected_binary_source_kind'
// STALE-SELECTED-CONFIG: selected_plan_metadata 'tcrv_rvv.selected_vector_lmul' lmul must be 'm1'
// MISSING-SELECTED-CONFIG: requires selected_plan_metadata 'tcrv_rvv.selected_vector_lmul'
// STALE-RUNTIME-LENGTH: selected_plan_metadata 'tcrv_rvv.runtime_avl_source' must use value 'runtime-element-count-abi-parameter'
// MISSING-RUNTIME-LENGTH: requires selected_plan_metadata 'tcrv_rvv.runtime_vl_source'
