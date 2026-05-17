# Legacy execution-planning artifact pipe test cleanup

## Goal

Delete or rewrite the remaining positive Template/Toy/Target tests that still
accept manually piping
`tcrv-opt --tcrv-execution-planning-pipeline` into low-level target artifact,
header artifact, bundle artifact, or plugin EmitC-to-C++ translate routes.

Low-level translate/export routes should be tested with explicit
materialized/planned IR fixtures. Source-level positive workflows should use
the existing one-command source artifact bundle front door, not a manual
planning-pipeline compatibility chain.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD `8430bed test: retire manual source
  artifact pipe`; worktree clean; no `.trellis/.current-task` existed.
- The previous archived cleanup task retired the old manual
  `source-artifact-front-door-pipeline | tcrv-translate
  --tcrv-export-target-artifact{,-header-artifact,-bundle}` source-artifact
  acceptance surface.
- Current bounded scan still found positive manual execution-planning pipes in:
  `test/Target/Template/template-target-artifact-object.mlir`,
  `test/Target/Template/template-target-artifact-header.mlir`,
  `test/Target/Template/template-emitc-to-cpp.mlir`,
  `test/Target/Template/template-emitc-to-cpp-compile.test`, and
  `test/Target/Toy/toy-target-artifact-header.mlir`.
- Existing one-command source bundle positive coverage already lives under
  `test/Target/TargetArtifactBundleExport/`.
- Existing Toy lower-level materialized exporter coverage already exists in
  `test/Target/Toy/toy-materialized-target-artifact-object.mlir`.

## Boundaries

- This is deletion/cleanup only.
- Do not add a new source front door, target route, compiler pass, plugin
  behavior, RVV/Toy/Template/TensorExtLite semantics, compatibility wrapper,
  descriptor adapter, direct C semantic exporter, or Python compiler-core
  behavior.
- Do not treat prompt edits, task metadata, helper-only changes, or broad smoke
  tests as the main result.
- Do not require new `ssh rvv` evidence unless production artifact bytes or ABI
  generation code changes. This task should not change production code.
- Retained low-level exporter tests must consume explicit materialized/planned
  IR and must keep negative checks against descriptor, source-export, direct-C,
  extension-specific core-branch, or metadata-diagnostic residue where
  applicable.
- Fail-closed tests for stale input, disabled plugins, missing materialized
  EmitC, or missing output directories may remain if they do not protect the
  old manual pipe as success behavior.

## Requirements

- Rewrite Template target object/header/bundle positive tests so their
  low-level export commands consume a materialized/planned fixture directly.
- Rewrite Template plugin EmitC-to-C++ positive and compile tests so the
  translate route consumes materialized/planned IR directly.
- Delete or rewrite Toy header positive coverage that still accepts the manual
  execution-planning pipe into `--tcrv-export-target-header-artifact`.
- Preserve existing one-command source artifact bundle front-door tests.
- Preserve retained fail-closed coverage only where it does not establish the
  manual pipe as a successful workflow.
- Add task notes or completion evidence identifying exact old success paths
  removed and retained materialized/fail-closed coverage.

## Acceptance Criteria

- [x] No positive lit/script RUN line under `test/Target/Template`,
  `test/Target/Toy`, or `test/Target/TargetArtifactBundleExport` matches a
  source MLIR input manually piped through
  `tcrv-opt --tcrv-execution-planning-pipeline` into
  `tcrv-translate --tcrv-export-target-artifact`,
  `--tcrv-export-target-header-artifact`,
  `--tcrv-export-target-artifact-bundle`, or plugin EmitC-to-C++ translate
  routes.
- [x] Retained Template/Toy low-level exporter tests use explicit
  materialized/planned fixtures.
- [x] Retained source-level positive coverage uses the existing one-command
  front door when applicable.
- [x] Retained fail-closed coverage does not claim manual planning-pipeline
  compatibility as success behavior.
- [x] Focused scans over `test/Target/Template`, `test/Target/Toy`,
  `test/Target/TargetArtifactBundleExport`, and directly related scripts show
  no forbidden positive manual-pipe pattern.
- [x] Focused lit for touched Template/Toy/Target tests passes.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes if practical; if not run, the reason is
  recorded.

## Out of Scope

- New architecture rebuild work after deletion.
- New source-level target export front doors.
- New route registration, lowering, emission, runtime ABI, or artifact byte
  generation behavior.
- New RVV hardware proof.
- Broad evidence matrices unrelated to this test cleanup.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-17-legacy-manual-source-artifact-pipe-cleanup/prd.md`.
- Relevant code read:
  `lib/Transforms/ExecutionPlanningPipeline.cpp` and
  `tools/tcrv-translate/tcrv-translate.cpp`.
- Initial forbidden-surface scan:
  `rg -n "execution-planning-pipeline|tcrv-execution-planning-pipeline|tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle|emitc-to-cpp" lib/Transforms/ExecutionPlanningPipeline.cpp tools/tcrv-translate/tcrv-translate.cpp test/Target/Template test/Target/Toy test/Target/TargetArtifactBundleExport`.

## Completion Evidence

- Rewrote `test/Target/Template/template-target-artifact-object.mlir` as
  lower-level materialized/planned Template exporter coverage. Its object,
  header, and bundle RUN lines now invoke `tcrv-translate` directly on the
  explicit fixture and no longer accept a manual execution-planning pipe.
- Rewrote `test/Target/Template/template-emitc-to-cpp.mlir` and
  `test/Target/Template/template-emitc-to-cpp-compile.test` so the Template
  EmitC-to-C++ route consumes the materialized/planned Template fixture
  directly.
- Rewrote the stale-route fail-closed path in
  `test/Target/Template/template-target-artifact-fail-closed.mlir` so the
  stale candidate is derived from the materialized/planned fixture, not a
  manual pipe.
- Deleted old positive manual-pipe header success coverage in
  `test/Target/Template/template-target-artifact-header.mlir` and
  `test/Target/Toy/toy-target-artifact-header.mlir`.
- Retained Toy lower-level object/header/bundle coverage in
  `test/Target/Toy/toy-materialized-target-artifact-object.mlir`; retained Toy
  stale-provenance fail-closed coverage in
  `test/Target/Toy/toy-target-artifact-stale-provenance.mlir`.
- Retained one-command source bundle front-door coverage under
  `test/Target/TargetArtifactBundleExport/`.

## Checks

- [x] Manual Template probes:
  `build/bin/tcrv-translate --tcrv-export-target-artifact
  test/Target/Template/template-target-artifact-object.mlir`,
  `--tcrv-export-target-header-artifact`,
  `--tcrv-template-emitc-to-cpp`, and
  `--tcrv-export-target-artifact-bundle` all succeeded on the materialized
  fixture.
- [x] Focused lit from `build/test` with filter
  `template-target-artifact-object|template-emitc-to-cpp|template-target-artifact-fail-closed|toy-materialized-target-artifact-object|toy-target-artifact-stale-provenance|source-artifact-bundle-front-door-(toy|rvv|fail-closed)`
  -> 9/9 passed.
- [x] `git diff --check`
- [x] Forbidden manual-pipe scan:
  `rg -n "tcrv-opt .*--tcrv-execution-planning-pipeline.*\\|.*tcrv-translate|--tcrv-execution-planning-pipeline \\| tcrv-translate|tcrv-execution-planning-pipeline \\| tcrv-translate|tcrv_opt|--tcrv-opt" test/Target/Template test/Target/Toy test/Target/TargetArtifactBundleExport scripts tools/tcrv-translate/tcrv-translate.cpp lib/Transforms/ExecutionPlanningPipeline.cpp`
  -> no matches.
- [x] `cmake --build build --target check-tianchenrv -j2` -> 124/124 passed.
- [x] No production code changed; no new `ssh rvv` proof was required.
