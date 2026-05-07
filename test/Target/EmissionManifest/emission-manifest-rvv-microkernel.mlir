// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_manifest_input {
  tcrv.exec.kernel @rvv_microkernel_manifest {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_microkernel_manifest"
    }
  }
}

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
// CHECK: runtime_glue_role: "runtime-callable-i32-vadd-function"
// CHECK: artifact_kind: "standalone-c-source"
// CHECK: required_capabilities: [@rvv]
// CHECK: explanation: "explicit RVV i32 vector-add microkernel C source export provides a runtime-callable C ABI function plus a standalone self-check harness for this selected path; this is not generic RVV lowering, runtime integration, arbitrary kernel emission, correctness, or performance evidence"
