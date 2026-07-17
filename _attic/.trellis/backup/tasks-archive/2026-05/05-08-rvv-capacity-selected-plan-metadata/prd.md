# RVV Capacity Metadata Propagation

## Goal

Carry structured RVV capacity facts that were already used for selected variant planning across the selected lowering/emission/artifact boundary as bounded plugin-owned diagnostic metadata. The result must make generated RVV or RVV+scalar target artifact bundles self-descriptive about the RVV capacity facts used for the selected RVV variant, while rejecting stale, malformed, missing, unpaired, or mismatched selected-plan capacity metadata before export success.

## What I Already Know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* HEAD before this round: `91e5cc9 feat: add RVV vlenb capacity capability path`.
* The prior round added RVV probe facts and plugin-local RVV capacity participation: `rvv.vlenb_bytes` / `rvv.i32_m1_lane_count` capability facts become `tcrv_rvv.vlenb_bytes` / `tcrv_rvv.i32_m1_lanes` selected variant metadata.
* Capacity facts are hardware/target capability facts and plugin-owned selected-plan metadata. They are not runtime SSA/control values, not descriptor-local `element_count`, and not performance evidence.
* The current selected planning path materializes `tcrv_rvv.lowering_boundary`, emission-plan diagnostics, and target artifact bundles through C++/MLIR passes and target exporter code.
* User explicitly requires one serial full-access worker and no subagents, spawned agents, parallel agents, background queues, or multi-agent workflow.

## Requirements

* Keep RVV-specific interpretation plugin-local, preferably in RVV capability/profile/plugin lowering-boundary and emission-plan code.
* Preserve the existing plan-and-export front door and C++/MLIR pass/translation plumbing.
* Propagate structured selected RVV capacity metadata from selected RVV variant metadata into selected lowering boundary and/or emission plan, then into manifest or bundle self-description.
* Add coherence validation so capacity-aware selected RVV plans cannot silently disagree with the selected variant and `TargetCapabilitySet` facts.
* Reject stale, missing, malformed, non-integer, unpaired, ratio-invalid, or mismatched capacity metadata when selected RVV capacity metadata is present.
* Keep emitted capacity metadata diagnostic/self-descriptive only. It must not become runtime input, shape model, VL/AVL value, or performance claim.
* Do not add Python compiler decisions or broad test matrices.

## Acceptance Criteria

* A focused positive test proves structured RVV capacity facts flow through the relevant C++ path into selected-plan metadata and target artifact bundle self-description.
* A focused negative test proves malformed or stale selected RVV capacity metadata fails before target bundle export success.
* Existing execution planning and plan-and-export path remain coherent with RVV capacity facts present.
* Local checks run:
  * `git diff --check`
  * `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Out Of Scope

* No generic RVV lowering, arbitrary RVV executable emission, runtime correctness claim, or performance claim.
* No IME, AME, Sophgo, vendor, or offload expansion.
* No reinterpretation of `tcrv_rvv.element_count` as shape, AVL, VL, runtime `n`, or hardware capacity.
* No conversion of vlenb or i32 lane count into runtime SSA/control values.

## Technical Notes

* Specs inspected: architecture, capability-model, core-dialect, extension-plugins/RVV, lowering-runtime, testing, variant-pipeline.
* Relevant code inspected: RVV capability profile/plugin, plugin interfaces, RVV/Exec ODS, lowering boundary/emission/coherence passes, emission manifest, target artifact export, RVV microkernel, RVV+scalar dispatch, tools, probe scripts, and focused existing tests.
* `lib/Target/Builtin/RVVMicrokernel.cpp` and `test/Transforms/LoweringBoundary/materialize-rvv-lowering-boundary.mlir` are not present; actual RVV microkernel source is `lib/Target/RVV/RVVMicrokernel.cpp`, and existing lowering-boundary tests live under `test/Transforms/LoweringBoundary/`.
