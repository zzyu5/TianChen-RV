// RUN: tcrv-opt %S/../../Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans --tcrv-check-execution-plan-coherence | FileCheck %s --check-prefix=RVV
// RUN: tcrv-opt %S/../../Target/ArtifactExport/Inputs/scalar-microkernel-source.txt --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans --tcrv-check-execution-plan-coherence | FileCheck %s --check-prefix=SCALAR
// RUN: tcrv-opt %S/../../Target/EmissionManifest/emission-manifest-offload-pipeline.mlir --tcrv-execution-planning-pipeline --tcrv-check-execution-plan-coherence | FileCheck %s --check-prefix=OFFLOAD
// RUN: tcrv-opt %S/../ExecutionPlanning/execution-planning-pipeline-builtin.mlir --split-input-file --tcrv-execution-planning-pipeline --tcrv-check-execution-plan-coherence | FileCheck %s --check-prefix=PIPE

// RVV-LABEL: tcrv.exec.kernel @rvv_microkernel_manifest
// RVV: tcrv_rvv.lowering_boundary
// RVV-SAME: selected_variant = @rvv_first_slice
// RVV: tcrv.exec.diagnostic
// RVV-SAME: artifact_kind = "runtime-callable-c-source"
// RVV-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// RVV-SAME: origin = "rvv-plugin"
// RVV-SAME: status = "supported"
// RVV-SAME: target = @rvv_first_slice

// SCALAR-LABEL: tcrv.exec.kernel @scalar_microkernel_export
// SCALAR: tcrv_scalar.lowering_boundary
// SCALAR-SAME: selected_variant = @scalar_fallback_first_slice
// SCALAR: tcrv.exec.diagnostic {{.*}}artifact_kind = "runtime-callable-c-source"
// SCALAR-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
// SCALAR-SAME: origin = "scalar-plugin"
// SCALAR-SAME: status = "supported"
// SCALAR-SAME: target = @scalar_fallback_first_slice

// OFFLOAD-LABEL: tcrv.exec.kernel @pipeline_offload_manifest
// OFFLOAD: tcrv_offload.lowering_boundary
// OFFLOAD-SAME: selected_variant = @offload_runtime_first_slice
// OFFLOAD: tcrv.exec.diagnostic
// OFFLOAD-SAME: artifact_kind = "runtime-offload-handoff-descriptor"
// OFFLOAD-SAME: lowering_pipeline = "tcrv-export-offload-runtime-descriptor"
// OFFLOAD-SAME: origin = "offload-plugin"
// OFFLOAD-SAME: runtime_abi_name = "generic-runtime-offload-c-abi-handoff.v1"
// OFFLOAD-SAME: status = "supported"
// OFFLOAD-SAME: target = @offload_runtime_first_slice

// PIPE-LABEL: tcrv.exec.kernel @pipeline_rvv_plus_scalar
// PIPE: tcrv.exec.dispatch
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE: tcrv_scalar.lowering_boundary
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: target = @scalar_fallback_first_slice
