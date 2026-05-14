// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.frontend.i32m2.vadd.bundle && mkdir %t.frontend.i32m2.vadd.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.i32m2.vadd.bundle %s > %t.frontend.i32m2.vadd.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.i32m2.vadd.stdout
// RUN: test -s %t.frontend.i32m2.vadd.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.frontend.i32m2.vadd.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.frontend.i32m2.vadd.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vsub-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vmul-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1 --implicit-check-not=rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vsub-dispatch --implicit-check-not=tcrv-export-rvv-scalar-i32-vmul-dispatch < %t.frontend.i32m2.vadd.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vsub_vv_i32m2 --implicit-check-not=__riscv_vmul_vv_i32m1 --implicit-check-not=__riscv_vmul_vv_i32m2 --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="lhs[index] * rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" < %t.frontend.i32m2.vadd.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" < %t.frontend.i32m2.vadd.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.i32m2.vadd.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32m2_vadd_bundle_input {
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

  tcrv.exec.target @frontend_vadd_i32m2_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vadd.i32m2.bundle",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    conflicts = ["build.policy.no_rvv"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m2",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_frontend_bundle_i32m2_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i32m2_vadd",
        tcrv_frontend_target = @frontend_vadd_i32m2_bundle_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
    }
    return
  }
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: runtime_abi_parameter[4]:
// INDEX: c_name: "rvv_available"
// INDEX: role: "dispatch-availability-guard"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.selected_vector_shape"
// INDEX: value: "i32m2"
// INDEX: selected_plan_metadata[2]:
// INDEX: name: "tcrv_rvv.selected_vector_lmul"
// INDEX: value: "m2"
// INDEX: selected_plan_metadata[6]:
// INDEX: name: "tcrv_rvv.selected_vector_suffix"
// INDEX: value: "i32m2"
// INDEX: selected_plan_metadata[7]:
// INDEX: name: "tcrv_rvv.selected_setvl_suffix"
// INDEX: value: "e32m2"
// INDEX: selected_plan_metadata[8]:
// INDEX: name: "tcrv_rvv.selected_vector_sew_capability"
// INDEX: value: "rvv.i32_m2.sew32"
// INDEX: selected_plan_metadata[9]:
// INDEX: name: "tcrv_rvv.selected_vector_lmul_capability"
// INDEX: value: "rvv.i32_m2.lmul_m2"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"

// SOURCE: /* Scope: one selected RVV i32-vadd dispatch case plus one scalar i32-vadd dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_bundle_i32m2_vadd */
// SOURCE: /* rvv_selected_variant: @rvv_first_slice */
// SOURCE: /* rvv_selected_role: dispatch case */
// SOURCE: /* rvv_selected_plan_metadata[0]: name=tcrv_rvv.selected_vector_shape, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[2]: name=tcrv_rvv.selected_vector_lmul, value=m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[6]: name=tcrv_rvv.selected_vector_suffix, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[7]: name=tcrv_rvv.selected_setvl_suffix, value=e32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[8]: name=tcrv_rvv.selected_vector_sew_capability, value=rvv.i32_m2.sew32, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[9]: name=tcrv_rvv.selected_vector_lmul_capability, value=rvv.i32_m2.lmul_m2, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[10]: name=tcrv_rvv.selected_tail_policy_capability, value=rvv.i32_m2.tail_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[11]: name=tcrv_rvv.selected_mask_policy_capability, value=rvv.i32_m2.mask_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32m2_vadd_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vadd_microkernel_frontend_bundle_i32m2_vadd_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// SOURCE: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
// SOURCE: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: __riscv_vsetvl_e32m2
// SOURCE: __riscv_vle32_v_i32m2
// SOURCE: __riscv_vadd_vv_i32m2
// SOURCE: __riscv_vse32_v_i32m2
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i32_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_add
// SOURCE: tcrv_scalar_i32_add
// SOURCE: void tcrv_dispatch_i32_vadd_frontend_bundle_i32m2_vadd

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_FRONTEND_BUNDLE_I32M2_VADD_H
// HEADER: void tcrv_dispatch_i32_vadd_frontend_bundle_i32m2_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_FRONTEND_BUNDLE_I32M2_VADD_H */

// OBJ: Format: elf64-littleriscv
// OBJ-DAG: Name: tcrv_dispatch_i32_vadd_frontend_bundle_i32m2_vadd
// OBJ-DAG: Name: tcrv_rvv_i32_vadd_microkernel_frontend_bundle_i32m2_vadd_rvv_first_slice
// OBJ-DAG: Name: tcrv_scalar_i32_vadd_microkernel_frontend_bundle_i32m2_vadd_scalar_fallback_first_slice
