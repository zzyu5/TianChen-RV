// RUN: rm -rf %t.offload.bundle && mkdir %t.offload.bundle
// RUN: tcrv-opt %S/../EmissionManifest/emission-manifest-offload-pipeline.mlir --tcrv-execution-planning-pipeline --tcrv-check-execution-plan-coherence | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.offload.bundle > %t.offload.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.offload.stdout
// RUN: test -s %t.offload.bundle/artifact-0-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=http < %t.offload.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=DESC --implicit-check-not=vendor_runtime_call --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=http < %t.offload.bundle/artifact-0-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt
// RUN: rm -rf %t.offload.plan.bundle && mkdir %t.offload.plan.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.offload.plan.bundle %S/../EmissionManifest/emission-manifest-offload-pipeline.mlir > %t.offload.plan.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.offload.plan.stdout
// RUN: test -s %t.offload.plan.bundle/artifact-0-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=http < %t.offload.plan.bundle/tianchenrv-target-artifact-bundle.index
// RUN: FileCheck %s --check-prefix=DESC --implicit-check-not=vendor_runtime_call --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=http < %t.offload.plan.bundle/artifact-0-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt

// RUN: rm -rf %t.rawlog.bundle && mkdir %t.rawlog.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.rawlog.bundle %S/../ArtifactExport/Inputs/offload-descriptor-raw-log.txt 2>&1 | FileCheck %s --check-prefix=RAWLOG
// RUN: test ! -e %t.rawlog.bundle/tianchenrv-target-artifact-bundle.index

module @unused_offload_bundle_test_anchor {
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 1
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt"
// INDEX: selected_variant: @offload_runtime_first_slice
// INDEX: role: "direct variant"
// INDEX: component[0]:
// INDEX: selected_variant: @offload_runtime_first_slice
// INDEX: role: "direct variant"
// INDEX: artifact_kind: "runtime-offload-handoff-descriptor"
// INDEX: route: "tcrv-export-offload-runtime-descriptor"
// INDEX: owner: "offload-plugin"
// INDEX: runtime_abi: "generic-runtime-offload-c-abi-handoff.v1"
// INDEX: runtime_abi_kind: "runtime-offload-c-abi-handoff"
// INDEX: runtime_abi_name: "generic-runtime-offload-c-abi-handoff.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: c_type: "const int32_t *"
// INDEX: role: "lhs-input-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[1]:
// INDEX: c_name: "rhs"
// INDEX: role: "rhs-input-buffer"
// INDEX: runtime_abi_parameter[2]:
// INDEX: c_name: "out"
// INDEX: role: "output-buffer"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: c_type: "size_t"
// INDEX: role: "runtime-element-count"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "runtime_offload_capability_id"
// INDEX: value: "offload.runtime"
// INDEX: role: "capability-requirement"
// INDEX: selected_plan_metadata[1]:
// INDEX: name: "runtime_offload_handoff_kind"
// INDEX: value: "runtime-offload"
// INDEX: role: "runtime-offload-handoff"
// INDEX: selected_plan_metadata[2]:
// INDEX: name: "runtime_offload_descriptor_scope"
// INDEX: value: "descriptor-only"
// INDEX: role: "evidence-scope"
// INDEX: handoff_kind: "runtime-offload"
// INDEX: evidence_role: "compiler-artifact"
// INDEX-NOT: artifact[1]:

// DESC: tianchenrv.offload_runtime_handoff_descriptor.version: 1
// DESC: descriptor_schema_version: 1
// DESC: descriptor_kind: "runtime-offload-handoff-descriptor"
// DESC: descriptor_status: "supported"
// DESC: adapter_contract: "external-runtime-adapter-runtime-offload-descriptor.v1"
// DESC: source_kernel: @pipeline_offload_manifest
// DESC: selected_variant: @offload_runtime_first_slice
// DESC: selected_role: "direct variant"
// DESC: origin_plugin: "offload-plugin"
// DESC: route_id: "tcrv-export-offload-runtime-descriptor"
// DESC: artifact_kind: "runtime-offload-handoff-descriptor"
// DESC: lowering_boundary_status: "metadata-only"
// DESC: runtime_abi: "generic-runtime-offload-c-abi-handoff.v1"
// DESC: handoff_kind: "runtime-offload"
// DESC: artifact_component_role: "descriptor"
// DESC: evidence_role: "compiler-artifact"
// DESC: abi_contract_owner: "compiler-target-export"
// DESC: abi_contract_source: "tcrv.exec.mem_window + tcrv.exec.runtime_param"
// DESC: abi_contract_entry_count: 4
// DESC: abi_contract_entry[0]:
// DESC: kind: "buffer"
// DESC: role: "lhs-input-buffer"
// DESC: c_name: "lhs"
// DESC: c_type: "const int32_t *"
// DESC: purpose: "runtime-abi-buffer"
// DESC: ownership: "target-export-abi-owned"
// DESC: source_symbol: @abi_lhs_input_buffer
// DESC: binding: "kernel-argument"
// DESC: memory_space: "host"
// DESC: access: "read"
// DESC: abi_contract_entry[2]:
// DESC: role: "output-buffer"
// DESC: c_name: "out"
// DESC: access: "write"
// DESC: abi_contract_entry[3]:
// DESC: kind: "scalar"
// DESC: role: "runtime-element-count"
// DESC: c_name: "n"
// DESC: c_type: "size_t"
// DESC: purpose: "runtime-abi-scalar"
// DESC: source_symbol: @abi_runtime_element_count
// DESC: selected_plan_metadata[0]:
// DESC: name: "runtime_offload_capability_id"
// DESC: value: "offload.runtime"
// DESC: role: "capability-requirement"
// DESC: selected_plan_metadata[1]:
// DESC: name: "runtime_offload_handoff_kind"
// DESC: value: "runtime-offload"
// DESC: role: "runtime-offload-handoff"
// DESC: selected_plan_metadata[2]:
// DESC: name: "runtime_offload_descriptor_scope"
// DESC: value: "descriptor-only"
// DESC: role: "evidence-scope"
// DESC: evidence_scope: "descriptor export only; no offload runtime execution, vendor call, DMA, object generation, hardware correctness, or performance evidence"
// DESC: non_claims: ["no-vendor-runtime-call", "no-dma-or-buffer-management", "no-accelerator-kernel", "no-object-generation", "no-hardware-execution", "no-correctness-proof", "no-performance-claim"]

// RAWLOG: target artifact bundle export failed
// RAWLOG: handoff_reason must not contain secret-like, URL, or raw credential text
