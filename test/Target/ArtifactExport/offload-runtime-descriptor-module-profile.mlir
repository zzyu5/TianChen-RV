// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact | FileCheck %s --check-prefix=DESC --implicit-check-not=vendor_runtime_call --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=http

module {
  tcrv.exec.target @module_offload_profile {
    id = "profile.offload.runtime",
    kind = "profile",
    provides = ["offload.runtime", "scalar.fallback"],
    status = "available",
    runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
    handoff_kind = "runtime-offload"
  }

  tcrv.exec.kernel @profile_offload_descriptor attributes {target = @module_offload_profile} {
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

// DESC: tianchenrv.offload_runtime_handoff_descriptor.version: 1
// DESC: source_kernel: @profile_offload_descriptor
// DESC: selected_variant: @offload_runtime_first_slice
// DESC: selected_role: "direct variant"
// DESC: origin_plugin: "offload-plugin"
// DESC: route_id: "tcrv-export-offload-runtime-descriptor"
// DESC: artifact_kind: "runtime-offload-handoff-descriptor"
// DESC: runtime_abi: "generic-runtime-offload-c-abi-handoff.v1"
// DESC: handoff_kind: "runtime-offload"
// DESC: required_capabilities: [@module_offload_profile]
// DESC: abi_contract_entry_count: 4
// DESC: role: "lhs-input-buffer"
// DESC: role: "rhs-input-buffer"
// DESC: role: "output-buffer"
// DESC: role: "runtime-element-count"
// DESC: selected_plan_metadata[0]:
// DESC: name: "runtime_offload_capability_id"
// DESC: value: "offload.runtime"
// DESC: selected_plan_metadata[1]:
// DESC: name: "runtime_offload_handoff_kind"
// DESC: value: "runtime-offload"
// DESC: evidence_scope: "descriptor export only; no offload runtime execution, vendor call, DMA, object generation, hardware correctness, or performance evidence"
