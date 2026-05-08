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
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "n",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}

// CHECK: tianchenrv.emission_manifest.version: 1
// CHECK: module: "offload_manifest_inputs"
// CHECK-LABEL: kernel @pipeline_offload_manifest
// CHECK: selected_surface: selected-marker
// CHECK: selection_kind: "static-variant"
// CHECK: path[0]:
// CHECK: selected_variant: @offload_runtime_first_slice
// CHECK: role: "direct variant"
// CHECK: origin: "offload-plugin"
// CHECK: emission_status: "supported"
// CHECK: emission_kind: "runtime-offload-handoff-descriptor"
// CHECK: lowering_pipeline: "tcrv-export-offload-runtime-descriptor"
// CHECK: lowering_boundary: "tcrv_offload.lowering_boundary"
// CHECK: runtime_abi: "generic-runtime-offload-c-abi-handoff.v1"
// CHECK: runtime_abi_kind: "runtime-offload-c-abi-handoff"
// CHECK: runtime_abi_name: "generic-runtime-offload-c-abi-handoff.v1"
// CHECK: runtime_abi_parameters:
// CHECK: parameter[0]:
// CHECK: c_name: "lhs"
// CHECK: role: "lhs-input-buffer"
// CHECK: parameter[1]:
// CHECK: c_name: "rhs"
// CHECK: role: "rhs-input-buffer"
// CHECK: parameter[2]:
// CHECK: c_name: "out"
// CHECK: role: "output-buffer"
// CHECK: parameter[3]:
// CHECK: c_name: "n"
// CHECK: role: "runtime-element-count"
// CHECK: runtime_glue_role: "plugin-owned-runtime-offload-glue-boundary"
// CHECK: artifact_kind: "runtime-offload-handoff-descriptor"
// CHECK: required_capabilities: [@offload_runtime]
// CHECK: explanation: "runtime-offload first slice can export a deterministic compiler handoff descriptor
// CHECK: preference:
// CHECK: available: true
// CHECK: policy: "prefer runtime-offload metadata handoff only when explicit offload.runtime capability metadata is available"
// CHECK-NOT: path[1]:
