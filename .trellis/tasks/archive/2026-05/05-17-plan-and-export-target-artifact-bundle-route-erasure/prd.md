# Plan-and-export target artifact bundle route erasure

## Goal

Delete the legacy `tcrv-translate
--tcrv-plan-and-export-target-artifact-bundle` production wrapper route. Target
artifact bundle export should consume already planned/materialized MLIR through
the existing `--tcrv-export-target-artifact-bundle` front door, while supported
source-level workflows should use the plugin source artifact bundle front door.
This round removes the wrapper implementation, route registration, tests, and
durable spec wording that still describe plan-and-export as a current workflow.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD `e454607 test: retire manual
  execution planning artifact pipe`; worktree clean; no `.trellis/.current-task`
  existed.
- The previous archived task retired positive manual
  `tcrv-opt --tcrv-execution-planning-pipeline | tcrv-translate ...` target
  artifact tests, but it intentionally did not remove production translate
  code.
- `tools/tcrv-translate/tcrv-translate.cpp` still registers
  `tcrv-plan-and-export-target-artifact-bundle` and implements a helper that
  runs `buildExecutionPlanningPipeline` inside `tcrv-translate` before calling
  bundle export.
- `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
  still invokes the deleted wrapper as named negative coverage.
- `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`
  reuses that plan-and-export-named fixture for source front-door negative
  coverage.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` still contain wording that
  treats plan-and-export target artifact bundle as supported or required
  coverage.

## Boundaries

- This is deletion/refactor-only. Deletion before rebuild is the campaign rule.
- Preserve the current low-level bundle exporter for already
  planned/materialized MLIR:
  `tcrv-translate --tcrv-export-target-artifact-bundle`.
- Preserve the sanctioned plugin source workflow:
  `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- Do not add a replacement route, compatibility alias, fallback wrapper, source
  front door, target route, descriptor adapter, direct C semantic exporter, or
  Python compiler-core behavior.
- Do not use prompt edits, reports, helper-only changes, broad smoke tests, or
  task metadata as the main result.
- Do not require new `ssh rvv` evidence because this task removes a wrapper and
  does not change generated RVV artifact semantics.

## Requirements

- Remove the `tcrv-plan-and-export-target-artifact-bundle` translate
  registration from `tcrv-translate`.
- Remove the route-specific implementation helper that runs the execution
  planning pipeline in-process before bundle export.
- Ensure `tcrv-translate --help` no longer advertises the deleted option.
- Ensure invoking the deleted option fails closed as an unknown/deleted command
  line option or equivalent no-route diagnostic.
- Remove or rewrite the no-viable plan-and-export test so it no longer protects
  the deleted route.
- Keep source artifact bundle front-door negative coverage, but make it use a
  neutral source-front-door fixture rather than a plan-and-export-named file.
- Update durable spec wording so it no longer says the deleted translate route
  may invoke the planning pipeline in-process or that tests should cover that
  front door as current behavior.
- Preserve current positive source bundle tests for RVV and Toy.
- Preserve materialized/planned Template and Toy low-level bundle exporter
  tests.

## Acceptance Criteria

- [x] `tcrv-translate --help` does not include
  `--tcrv-plan-and-export-target-artifact-bundle`.
- [x] Invoking `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`
  fails closed before artifact export and emits no bundle completion output.
- [x] `tools/tcrv-translate/tcrv-translate.cpp` contains no registration,
  implementation helper, or route diagnostic for
  `tcrv-plan-and-export-target-artifact-bundle`.
- [x] The old plan-and-export no-viable lit test is deleted or rewritten so it
  no longer invokes or names the deleted route.
- [x] Source artifact bundle front-door tests still cover the sanctioned
  one-command source workflow and negative behavior without depending on the
  deleted wrapper.
- [x] Specs no longer describe plan-and-export target artifact bundle as a
  current workflow, in-process planning wrapper, or required test surface.
- [x] Focused scans over `tools/tcrv-translate/tcrv-translate.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `test/Target/TargetArtifactBundleExport`, `test/Target/Template`,
  `test/Target/Toy`, and directly touched specs show no positive
  plan-and-export route protection.
- [x] Focused build/test coverage for `tcrv-translate` and affected target
  artifact tests passes, or failures are classified as expected deletion gaps
  without restoring the old path.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is recorded.

## Out of Scope

- New general RVV lowering, new source front doors, new plan-and-export
  wrappers, compatibility aliases, descriptor adapters, direct C semantic
  exporters, source-export routes, Python compiler-core behavior, core
  extension-specific branches, artifact ledger/state machine changes, or new
  `ssh rvv` evidence.
- Rebuilding future high-level frontend lowering.
- Broad test matrices unrelated to the removed translate route.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-17-legacy-execution-planning-artifact-pipe-test-cleanup/prd.md`.
- Relevant code/tests read:
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  source artifact bundle front-door tests, and Template/Toy target tests.
- Initial route scan:
  `rg -n "plan-and-export|tcrv-plan-and-export-target-artifact-bundle|export-target-artifact-bundle|source-artifact-bundle-front-door|execution-planning-pipeline" tools/tcrv-translate/tcrv-translate.cpp lib/Transforms/ExecutionPlanningPipeline.cpp test/Target/TargetArtifactBundleExport test/Target/Template test/Target/Toy .trellis/spec/lowering-runtime .trellis/spec/variant-pipeline`.

## Completion Evidence

- Removed `planAndExportTargetArtifactBundle` from
  `tools/tcrv-translate/tcrv-translate.cpp`.
- Removed the `TranslateFromMLIRRegistration` for
  `tcrv-plan-and-export-target-artifact-bundle`.
- Deleted
  `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`.
- Rewired
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`
  to use its own neutral no-artifact input fixture rather than a
  plan-and-export-named fixture.
- Updated
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` so the route is documented
  as deleted/absent instead of a current in-process planning/export workflow.

## Checks

- [x] `cmake --build build --target tcrv-translate -j2`
- [x] `build/bin/tcrv-translate --help 2>&1 | rg -n
  "tcrv-plan-and-export-target-artifact-bundle|plan-and-export"; test $? -eq 1`
- [x] `build/bin/tcrv-translate
  --tcrv-plan-and-export-target-artifact-bundle - </dev/null` -> exit status
  1, unknown command-line argument, no
  `tianchenrv.target_artifact_bundle_export: complete` marker.
- [x] Focused lit from `build/test` with filter
  `TargetArtifactBundleExport|Template|Toy` -> 26/26 passed.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2` -> 123/123 passed.
- [x] Route scan:
  `rg -n "tcrv-plan-and-export-target-artifact-bundle|plan-and-export target artifact bundle failed|planAndExportTargetArtifactBundle" tools lib include test .trellis/spec; test $? -eq 1`
  -> no matches.
- [x] Manual-pipe/spec-residue scan:
  `rg -n "may invoke this same planning pipeline|plan-and-export target artifact bundle front-door coverage|run TianChen-RV execution planning and export selected target artifacts|tcrv-opt .*--tcrv-execution-planning-pipeline.*\\|.*tcrv-translate|--tcrv-execution-planning-pipeline \\| tcrv-translate|tcrv-execution-planning-pipeline \\| tcrv-translate|tcrv_opt|--tcrv-opt" tools lib include test .trellis/spec; test $? -eq 1`
  -> no matches.
- [x] Remaining `plan-and-export` string scan over code/test/spec reports only
  negative spec wording that the deleted wrapper must stay absent.
