// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=DELETED --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_manifest_input {
  tcrv.exec.kernel @rvv_microkernel_manifest {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_microkernel_manifest"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// DELETED: message = "runtime-callable RVV direct C source exporter was deleted; rebuild requires a materialized MLIR EmitC module source route"
// DELETED-SAME: reason = "emission_plan"
// DELETED-SAME: runtime_abi_kind = "unsupported-plugin-runtime-abi"
// DELETED-SAME: runtime_abi_name = "unsupported-emission-runtime-abi"
// DELETED-SAME: runtime_glue_role = "no-runtime-glue-unsupported"
// DELETED-SAME: status = "unsupported"

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK: module: "rvv_microkernel_manifest_input"
// CHECK-LABEL: kernel @rvv_microkernel_manifest
// CHECK: selected_surface: selected-marker
// CHECK: selection_kind: "static-variant"
// CHECK: path[0]:
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "direct variant"
// CHECK: origin: "rvv-plugin"
// CHECK: emission_status: "supported"
// CHECK: emission_kind: "rvv-explicit-i32-vadd-microkernel-c-source"
// CHECK: lowering_pipeline: "tcrv-export-rvv-microkernel-c"
// CHECK: lowering_boundary: "tcrv_rvv.lowering_boundary"
// CHECK: runtime_abi: "rvv-i32-vadd-runtime-callable-c-abi.v1"
// CHECK: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// CHECK: runtime_abi_parameters:
// CHECK: c_name: "lhs"
// CHECK: role: "lhs-input-buffer"
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: runtime_glue_role: "runtime-callable-i32-vadd-function"
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: required_capabilities: [@rvv]
// CHECK: explanation: "explicit RVV i32 vector-add microkernel C source export provides a library-style runtime-callable C ABI function for this selected path; any self-check main is an explicit harness export and is not the default artifact contract; this is not generic RVV lowering, runtime integration, arbitrary kernel emission, correctness, or performance evidence"
// CHECK: target_artifacts:
// CHECK: artifact[0]:
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: route: "tcrv-export-rvv-microkernel-c"
// CHECK: owner: "rvv-plugin"
// CHECK: generic_front_door_selectable: true
// CHECK: selectable_via: "tcrv-export-target-source-artifact"
// CHECK: direct_helper_route: true
// CHECK: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// CHECK: evidence_role: "compiler-artifact"
// CHECK: artifact[1]:
// CHECK: artifact_kind: "runtime-callable-c-header"
// CHECK: route: "tcrv-export-rvv-microkernel-header"
// CHECK: selectable_via: "tcrv-export-target-header-artifact"
// CHECK: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// CHECK: evidence_role: "header-declaration"
// CHECK: artifact[2]:
// CHECK: artifact_kind: "riscv-elf-relocatable-object"
// CHECK: route: "tcrv-export-rvv-microkernel-object"
// CHECK: selectable_via: "tcrv-export-target-artifact"
// CHECK: runtime_abi_kind: "rvv-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "rvv-i32-vadd-runtime-callable-c-function.v1"
// CHECK: evidence_role: "relocatable-object"
