# Legacy manual source-artifact pipe cleanup

## Goal

Migrate or delete positive source-to-artifact acceptance surfaces that still
advertise the manual
`tcrv-opt --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact{,-header-artifact,-bundle}`
pipe as the default workflow. The accepted source-input artifact path for this
round is the one-command
`tcrv-translate --tcrv-source-artifact-bundle-front-door` front door.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`; HEAD before this task
  is `d56a502 test(rvv): prove source bundle ABI execution`; the worktree was
  clean before task creation.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits and started as current task.
- Commit `d56a502` already proved RVV add/sub/mul generated bundle ABI
  execution on real `ssh rvv` through the one-command source bundle front door.
  This round does not need a new RVV runtime proof unless production artifact
  bytes or ABI generation change.
- Current positive one-command source bundle lit coverage exists for RVV and
  Toy under `test/Target/TargetArtifactBundleExport/`.
- Positive manual-pipe source-to-artifact surfaces still exist in
  `test/Target/RVV/`, `test/Target/Toy/`, and `test/Target/TensorExtLite/`.

## Boundaries

- This is a cleanup/migration task, not a new compiler feature.
- Do not add a new RVV family, dtype, LMUL, source lowering, plugin template,
  evidence matrix, wrapper CLI, compatibility mode, descriptor route, direct C
  semantic exporter, source-export route, or Python compiler-core behavior.
- Keep `tcrv-export-target-artifact`,
  `tcrv-export-target-header-artifact`, and
  `tcrv-export-target-artifact-bundle` only as lower-level exporters for
  already planned/materialized IR or direct exporter coverage. Retained tests
  must not claim source-front-door workflow authority.
- Preserve fail-closed coverage for disabled/unregistered source front doors,
  stale source metadata, no selected artifact route, missing output directory,
  stale route/provenance, descriptor/direct-C/source-export residue, and
  mismatched materialized handoff identity.

## Requirements

- Rewrite source-to-bundle RVV/Toy/TensorExtLite positive lit coverage so it
  uses `--tcrv-source-artifact-bundle-front-door` or an equivalent current
  one-command front door.
- Delete or rewrite positive source-to-object and source-to-header tests that
  depend on the manual source-artifact pipe.
- Retain lower-level object/header/bundle exporter tests only when the input is
  already planned/materialized IR and the test name/comment scopes it as a
  lower-level exporter surface.
- Keep `scripts/rvv_generated_bundle_abi_e2e.py` aligned with the production
  one-command source bundle front door.
- Add or preserve focused scans proving no positive source-to-artifact
  acceptance test still advertises the old manual pipe as the workflow.

## Acceptance Criteria

- [x] RVV, Toy, and TensorExtLite source-to-bundle positive lit coverage uses
  `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- [x] No positive test under the touched target surfaces uses
  `tcrv-opt --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact{,-header-artifact,-bundle}`
  as source-to-artifact acceptance.
- [x] Retained lower-level exporter tests start from planned/materialized IR and
  explicitly scope themselves as lower-level exporter coverage.
- [x] Fail-closed source bundle front-door coverage still rejects disabled
  plugins, stale source-front-door metadata, no supported artifact route, and
  invalid/missing bundle output directories.
- [x] Focused lit passes for touched RVV/Toy/TensorExtLite/TargetArtifactBundle
  surfaces.
- [x] Focused scans over touched tests/tooling show no new descriptor authority,
  direct-C/source-export semantic path, or manual source pipe positive workflow.
- [x] `git diff --check` passes.

## Out of Scope

- New compiler lowering behavior, new target exporter behavior, new runtime ABI
  behavior, new source frontend recognition, broad evidence matrices, new
  `ssh rvv` execution proof, or broad test-suite reruns not justified by the
  touched surfaces.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-source-bundle-runtime-abi-proof/prd.md`.
- Initial scan found old manual-pipe positive surfaces in:
  `test/Target/RVV/vector-source-target-artifact-object.mlir`,
  `test/Target/RVV/vector-source-target-artifact-header.mlir`,
  `test/Target/Toy/toy-target-artifact-object.mlir`,
  `test/Target/Toy/toy-source-front-door-target-artifact-header.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-target-artifact-object.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-target-artifact-header.mlir`,
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-target-artifact-bundle.mlir`,
  and
  `test/Target/TensorExtLite/tensorext-lite-source-front-door-emitc-to-cpp.mlir`.

## Completion Evidence

- Deleted old RVV source-to-object/header positive tests:
  `test/Target/RVV/vector-source-target-artifact-object.mlir` and
  `test/Target/RVV/vector-source-target-artifact-header.mlir`.
- Added scoped RVV lower-level materialized exporter coverage:
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.
- Deleted old Toy source-front-door object/header positive tests and replaced
  the object/header/bundle route with scoped materialized-module coverage:
  `test/Target/Toy/toy-materialized-target-artifact-object.mlir`.
- Deleted old TensorExtLite source-front-door object/header positive tests and
  added scoped materialized-module object/bundle coverage:
  `test/Target/TensorExtLite/tensorext-lite-materialized-target-artifact-object.mlir`
  and
  `test/Target/TensorExtLite/tensorext-lite-materialized-target-artifact-bundle.mlir`.
- Rewrote TensorExtLite source-to-bundle lit coverage to use
  `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- Extended RVV source bundle front-door lit coverage to add/sub/mul through the
  one-command source bundle front door.
- Updated `scripts/tensorextlite_runtime_abi_e2e.py` so the source-input bundle
  is produced by the one-command front door and lower-level EmitC/header/object
  checks use an explicit materialized IR fixture.
- Retained old-pipe references are limited to the pipeline registration in
  `lib/Transforms/ExecutionPlanningPipeline.cpp` and a stale-input filename
  used by the one-command source bundle fail-closed test.

## Checks

- [x] `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- [x] Focused lit from `build/test` with filter:
  `source-artifact-bundle-front-door-rvv|source-artifact-bundle-front-door-toy|source-artifact-bundle-front-door-fail-closed|tensorext-lite-source-front-door-target-artifact-bundle|tensorext-lite-source-front-door-emitc-to-cpp|tensorext-lite-runtime-abi-harness|tensorext-lite-materialized-target-artifact-object|tensorext-lite-materialized-target-artifact-bundle|vector-materialized-target-artifact-exporters|toy-materialized-target-artifact-object`
  -> 10/10 passed.
- [x] `cmake --build build --target check-tianchenrv -j2` -> 126/126 passed.
- [x] `git diff --check`
- [x] Manual-pipe positive workflow scan:
  `rg -n "tcrv-opt .*source-artifact-front-door-pipeline.*tcrv-translate|source-artifact-front-door-pipeline \\| tcrv-translate|--tcrv-opt|tcrv_opt" test/Target scripts/rvv_generated_bundle_abi_e2e.py scripts/tensorextlite_runtime_abi_e2e.py tools/tcrv-translate/tcrv-translate.cpp lib/Transforms/ExecutionPlanningPipeline.cpp`
  -> no matches.
