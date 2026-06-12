# compiler owner continuation

## Goal

Advance the real compiler path by letting the existing
`tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` front door
consume the current bounded marked `linalg.generic` i32 vector-add frontend
input, lower it to the `tcrv.exec` boundary, run the existing execution
planning pipeline, and export the selected target artifact bundle.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD before this task is `15fbee0 feat: lower bounded linalg vadd to exec boundary`.
* Latest Hermes audit shows the previous completed round added
  `--tcrv-lower-linalg-i32-vadd-to-exec`, positive/negative lit tests, and
  docs/specs for the bounded linalg frontend slice.
* `tcrv-opt` can already run:
  `--tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline`.
* `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` currently
  runs execution planning and bundle export for input that already contains
  `tcrv.exec.kernel` anchors.
* Feeding the existing linalg frontend test input directly to
  `--tcrv-plan-and-export-target-artifact-bundle` currently fails with:
  `TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel`.

## Requirements

* Keep implementation in C++/MLIR/CMake/lit/FileCheck.
* Do not add Python pseudo-IR or Python compiler internals.
* Keep `tcrv.exec` compute-free; the frontend lowering may create only the
  bounded exec kernel plus existing `mem_window` / `runtime_param` ABI
  boundary.
* Do not add RVV/scalar/offload/IME semantic branches to core orchestration.
* Reuse the existing bounded linalg lowering pass, execution planning pipeline,
  target artifact exporter registry, and bundle exporter.
* The translate front door must still work for existing hand-written
  TianChen-RV exec input.
* Invalid marked linalg input must fail during frontend lowering or planning
  before a complete bundle is reported.
* Generated bundle output remains compiler artifact handoff metadata only; it
  must not claim runtime success, correctness, performance, or `ssh rvv`
  evidence.

## Acceptance Criteria

* [ ] `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` accepts a
  bounded marked linalg i32-vadd module and exports a deterministic bundle
  through the existing target-owned RVV/scalar routes.
* [ ] Existing exec-input plan-and-export bundle tests continue to pass.
* [ ] A focused lit/FileCheck test covers linalg input through translate
  planning/export and checks bundle index/source/header/object presence when
  local object support is available.
* [ ] README/spec mention that the plan-and-export bundle front door may run
  the bounded frontend lowering before planning, without turning it into generic
  linalg lowering or runtime evidence.
* [ ] Build and relevant tests pass; if possible, run full `check-tianchenrv`.

## Out Of Scope

* Generic linalg lowering.
* New `tcrv` compute ops.
* New RVV/scalar/offload plugin semantics.
* New runtime ABI design beyond existing `mem_window` / `runtime_param`
  surfaces.
* Hardware runtime correctness or performance claims.

## Technical Notes

* Relevant code:
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `lib/Transforms/LowerLinalgI32VAddToExec.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`.
* Relevant specs:
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Compiler path made more real:
  marked high-level MLIR linalg i32-vadd input -> `tcrv.exec` ABI boundary ->
  plugin-proposed variants -> selection/dispatch -> plugin-owned
  lowering-boundary/emission-plan -> target-owned artifact bundle.
