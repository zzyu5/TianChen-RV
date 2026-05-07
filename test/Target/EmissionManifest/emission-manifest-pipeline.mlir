// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s

module {
  tcrv.exec.kernel @pipeline_manifest {
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
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK-LABEL: kernel @pipeline_manifest
// CHECK: selected_surface: dispatch
// CHECK: dispatch_case[0]: @rvv_first_slice
// CHECK: dispatch_fallback: @scalar_fallback_first_slice
// CHECK: path[0]:
// CHECK: selected_variant: @rvv_first_slice
// CHECK: role: "dispatch case"
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
// CHECK: c_type: "const int32_t *"
// CHECK: role: "lhs-input-buffer"
// CHECK: ownership: "target-export-abi-owned"
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: runtime_glue_role: "runtime-callable-i32-vadd-function"
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: required_capabilities: [@rvv]
// CHECK: explanation: "explicit RVV i32 vector-add microkernel C source export provides a library-style runtime-callable C ABI function for this selected path; any self-check main is an explicit harness export and is not the default artifact contract; this is not generic RVV lowering, runtime integration, arbitrary kernel emission, correctness, or performance evidence"
// CHECK: path[1]:
// CHECK: selected_variant: @scalar_fallback_first_slice
// CHECK: role: "dispatch fallback"
// CHECK: origin: "scalar-plugin"
// CHECK: emission_status: "supported"
// CHECK: emission_kind: "scalar-explicit-i32-vadd-microkernel-c-source"
// CHECK: lowering_pipeline: "tcrv-export-scalar-microkernel-c"
// CHECK: lowering_boundary: "tcrv_scalar.lowering_boundary"
// CHECK: runtime_abi: "scalar-i32-vadd-runtime-callable-c-abi.v1"
// CHECK: runtime_abi_kind: "scalar-runtime-callable-c-abi"
// CHECK: runtime_abi_name: "scalar-i32-vadd-runtime-callable-c-function.v1"
// CHECK: runtime_abi_parameters:
// CHECK: c_name: "lhs"
// CHECK: role: "lhs-input-buffer"
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: runtime_glue_role: "runtime-callable-i32-vadd-fallback-function"
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: required_capabilities: [@scalar_fallback]
