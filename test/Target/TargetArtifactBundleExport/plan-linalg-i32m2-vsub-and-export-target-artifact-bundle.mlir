// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.frontend.i32m2.vsub.bundle && mkdir %t.frontend.i32m2.vsub.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.i32m2.vsub.bundle %s > %t.frontend.i32m2.vsub.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.frontend.i32m2.vsub.stdout
// RUN: test -s %t.frontend.i32m2.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: test -s %t.frontend.i32m2.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: test -s %t.frontend.i32m2.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=rvv-scalar-i32-vadd-dispatch-external-abi.v1 --implicit-check-not=rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1 --implicit-check-not=tcrv-export-rvv-scalar-i32-vadd-dispatch < %t.frontend.i32m2.vsub.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" < %t.frontend.i32m2.vsub.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=__riscv --implicit-check-not=i32_vadd --implicit-check-not=") {" --implicit-check-not="int main" --implicit-check-not="_self_check" < %t.frontend.i32m2.vsub.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h
// RUN: llvm-readobj --file-headers --symbols %t.frontend.i32m2.vsub.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o | FileCheck %s --check-prefixes=OBJ --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32m2_vsub_bundle_input {
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

  tcrv.exec.target @frontend_vsub_i32m2_bundle_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@no_rvv_policy, @scalar_fallback],
    id = "rvv.profile.frontend.vsub.i32m2.bundle",
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

  func.func @source_frontend_bundle_i32m2_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_bundle_i32m2_vsub",
        tcrv_frontend_target = @frontend_vsub_i32m2_bundle_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vsub"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %diff = arith.subi %a, %b : i32
      linalg.yield %diff : i32
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
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: component[1]:
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[1]:
// INDEX: c_name: "rhs"
// INDEX: role: "rhs-input-buffer"
// INDEX: runtime_abi_parameter[2]:
// INDEX: c_name: "out"
// INDEX: role: "output-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: role: "runtime-element-count"
// INDEX: runtime_abi_parameter[4]:
// INDEX: c_name: "rvv_available"
// INDEX: role: "dispatch-availability-guard"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.selected_vector_shape"
// INDEX: value: "i32m2"
// INDEX: selected_plan_metadata[1]:
// INDEX: name: "tcrv_rvv.selected_vector_sew"
// INDEX: value: "32"
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
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vsub-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-header"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vsub-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vsub-dispatch-object"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"
// INDEX: evidence_role: "relocatable-object"

// SOURCE: /* Scope: one selected RVV i32-vsub dispatch case plus one scalar i32-vsub dispatch fallback. */
// SOURCE: /* selected_kernel: @frontend_bundle_i32m2_vsub */
// SOURCE: /* rvv_selected_variant: @rvv_first_slice */
// SOURCE: /* rvv_selected_role: dispatch case */
// SOURCE: /* rvv_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* rvv_runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* rvv_runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: /* rvv_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* rvv_selected_plan_metadata[0]: name=tcrv_rvv.selected_vector_shape, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[1]: name=tcrv_rvv.selected_vector_sew, value=32, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[2]: name=tcrv_rvv.selected_vector_lmul, value=m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[6]: name=tcrv_rvv.selected_vector_suffix, value=i32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[7]: name=tcrv_rvv.selected_setvl_suffix, value=e32m2, role=selected-rvv-vector-shape-config
// SOURCE: /* rvv_selected_plan_metadata[8]: name=tcrv_rvv.selected_vector_sew_capability, value=rvv.i32_m2.sew32, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[9]: name=tcrv_rvv.selected_vector_lmul_capability, value=rvv.i32_m2.lmul_m2, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[10]: name=tcrv_rvv.selected_tail_policy_capability, value=rvv.i32_m2.tail_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_selected_plan_metadata[11]: name=tcrv_rvv.selected_mask_policy_capability, value=rvv.i32_m2.mask_policy.agnostic, role=selected-rvv-vector-shape-capability
// SOURCE: /* rvv_required_capabilities: @frontend_vsub_i32m2_bundle_profile */
// SOURCE: /* scalar_selected_variant: @scalar_fallback_first_slice */
// SOURCE: /* scalar_selected_role: dispatch fallback */
// SOURCE: /* scalar_lowering_boundary: tcrv_scalar.lowering_boundary */
// SOURCE: /* scalar_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* scalar_runtime_abi_parameter[1]: c_name=rhs, c_type=const int32_t *, role=rhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* scalar_runtime_abi_parameter[2]: c_name=out, c_type=int32_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: /* scalar_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* scalar_required_capabilities: @scalar_fallback */
// SOURCE: /* dispatch_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// SOURCE: /* dispatch_mem_window[1]: symbol=@abi_rhs_input_buffer, abi_role=rhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// SOURCE: /* dispatch_mem_window[2]: symbol=@abi_output_buffer, abi_role=output-buffer, access=write, ownership=target-export-abi-owned, c_type=int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// SOURCE: /* dispatch_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=n, c_type=size_t, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// SOURCE: /* dispatch_runtime_param[1]: symbol=@abi_dispatch_availability_guard, abi_role=dispatch-availability-guard, c_name=rvv_available, c_type=int, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// SOURCE: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// SOURCE: /* dispatch_fallback_link: target=@scalar_fallback_first_slice, selected_scalar_callable=@scalar_fallback_first_slice */
// SOURCE: /* rvv_callable_symbol: tcrv_rvv_i32_vsub_microkernel_frontend_bundle_i32m2_vsub_rvv_first_slice */
// SOURCE: /* scalar_callable_symbol: tcrv_scalar_i32_vsub_microkernel_frontend_bundle_i32m2_vsub_scalar_fallback_first_slice */
// SOURCE: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// SOURCE: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
// SOURCE: /* control_plane_config: sew=32, lmul=m2, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// SOURCE: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: __riscv_vsetvl_e32m2
// SOURCE: __riscv_vle32_v_i32m2
// SOURCE: __riscv_vsub_vv_i32m2
// SOURCE: __riscv_vse32_v_i32m2
// SOURCE: int32_t difference = tcrv_scalar_i32_sub(lhs[index], rhs[index]);
// SOURCE: void tcrv_dispatch_i32_vsub_frontend_bundle_i32m2_vsub

// HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_BUNDLE_I32M2_VSUB_H
// HEADER: void tcrv_dispatch_i32_vsub_frontend_bundle_i32m2_vsub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH_FRONTEND_BUNDLE_I32M2_VSUB_H */

// OBJ: Format: elf64-littleriscv
// OBJ-DAG: Name: tcrv_dispatch_i32_vsub_frontend_bundle_i32m2_vsub
// OBJ-DAG: Name: tcrv_rvv_i32_vsub_microkernel_frontend_bundle_i32m2_vsub_rvv_first_slice
// OBJ-DAG: Name: tcrv_scalar_i32_vsub_microkernel_frontend_bundle_i32m2_vsub_scalar_fallback_first_slice
