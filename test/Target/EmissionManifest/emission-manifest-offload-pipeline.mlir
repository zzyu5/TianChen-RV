// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-emission-manifest | FileCheck %s

module @offload_manifest_inputs {
  tcrv.exec.kernel @pipeline_offload_manifest {
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK: module: "offload_manifest_inputs"
// CHECK-LABEL: kernel @pipeline_offload_manifest
// CHECK: selected_surface: dispatch
// CHECK: dispatch_case[0]: @offload_runtime_first_slice
// CHECK: dispatch_fallback: @scalar_fallback_first_slice
// CHECK: path[0]:
// CHECK: selected_variant: @offload_runtime_first_slice
// CHECK: role: "dispatch case"
// CHECK: origin: "offload-plugin"
// CHECK: emission_status: "supported"
// CHECK: emission_kind: "runtime-offload-handoff-descriptor"
// CHECK: lowering_pipeline: "tcrv-export-offload-runtime-descriptor"
// CHECK: lowering_boundary: "tcrv_offload.lowering_boundary"
// CHECK: runtime_abi: "generic-runtime-offload-c-abi-handoff.v1"
// CHECK: runtime_abi_kind: "runtime-offload-c-abi-handoff"
// CHECK: runtime_abi_name: "generic-runtime-offload-c-abi-handoff.v1"
// CHECK: runtime_glue_role: "plugin-owned-runtime-offload-glue-boundary"
// CHECK: artifact_kind: "runtime-offload-handoff-descriptor"
// CHECK: required_capabilities: [@offload_runtime]
// CHECK: explanation: "runtime-offload first slice can export a deterministic compiler handoff descriptor
// CHECK: preference:
// CHECK: available: true
// CHECK: policy: "prefer runtime-offload metadata handoff only when explicit offload.runtime capability metadata is available"
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
// CHECK: runtime_glue_role: "runtime-callable-i32-vadd-fallback-function"
// CHECK: artifact_kind: "runtime-callable-c-source"
// CHECK: required_capabilities: [@scalar_fallback]
// CHECK: fallback_role: "conservative"
