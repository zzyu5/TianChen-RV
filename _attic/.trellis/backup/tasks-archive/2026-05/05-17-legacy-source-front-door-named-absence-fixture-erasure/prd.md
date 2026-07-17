# Legacy source-front-door named-absence fixture erasure

## Goal

Delete the remaining active transform fixture whose primary purpose is to keep
a historical source-seed / selected-boundary seed option spelling alive as a
public named-absence surface. Preserve current plugin source-front-door and
generic source-artifact front-door behavior tests.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `32af2cc test: erase core rvv source absence fixtures`; worktree clean; no
  `.trellis/.current-task` existed.
- The previous archived task removed RVV and SourceFrontDoor old-option
  negative fixtures without adding aliases, replacement routes, descriptor
  adapters, source frontends, EmitC routes, artifact routes, or Python
  compiler-core behavior.
- The directly related remaining active stale fixture is
  `test/Transforms/Toy/toy-template-selected-boundary-seed-deleted.mlir`, which
  invokes `--tcrv-toy-materialize-template-selected-boundary-seed` only to
  assert `Unknown command line argument`.
- Current valid source-front-door coverage must remain centered on:
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door`,
  `--tcrv-toy-materialize-template-source-front-door`,
  `--tcrv-tensorext-lite-materialize-fragment-mma-source-front-door`, and
  `--tcrv-source-artifact-front-door-pipeline`.
- Direct scans over `test/Transforms`, `lib/Transforms`,
  `include/TianChenRV/Transforms`, and directly related specs found no real
  registration for `--tcrv-toy-materialize-template-selected-boundary-seed`;
  the active residue is the Toy named-absence fixture itself.

## Boundaries

- Deletion only. This round may delete stale tests, comments, and directly
  related spec/test references whose main purpose is preserving deleted
  source-seed or selected-boundary seed option names as public CLI behavior.
- Keep current active source-front-door and pipeline tests even when they use
  `Unknown command line argument` to prove built-in-plugin disabling for current
  options.
- Keep negative tests for stale `*.lowering_seed` metadata when they exercise
  current plugin source-front-door fail-closed diagnostics rather than old
  option-name compatibility.
- Do not add replacement routes, compatibility aliases, wrappers, legacy modes,
  new source frontends, new EmitC routes, new artifact routes, descriptor
  adapters, direct C semantic exporters, broader RVV/Toy/TensorExt coverage,
  `ssh rvv` evidence, or Python compiler-core behavior.
- Do not convert this into a broad repo audit; scans stay bounded to transform
  source-front-door surfaces and directly related specs/tests.

## Requirements

- Remove active lit coverage whose purpose is unknown-option coverage for
  `--tcrv-toy-materialize-template-selected-boundary-seed`.
- Remove the stale Toy selected-boundary seed fixture rather than replacing it
  with an alias, wrapper, or new compatibility test.
- Preserve current Toy source-front-door positive coverage and disabled
  built-in plugin coverage in `test/Transforms/Toy/toy-template-source-front-door.mlir`.
- Preserve current RVV, TensorExtLite, and generic source-artifact front-door
  coverage.
- If a real old option registration or source reference is discovered, delete
  it and report any resulting build/test gap instead of adding aliases.

## Acceptance Criteria

- [x] No active lit fixture remains whose purpose is `Unknown command line
  argument` coverage for historical source-seed / selected-boundary seed route
  options.
- [x] Focused scans over `test/Transforms`, `lib/Transforms`,
  `include/TianChenRV/Transforms`, and directly related specs show no active
  `--tcrv-*-selected-boundary-seed` or source-seed route option residue except
  valid current fail-closed metadata diagnostics.
- [x] Current plugin source-front-door tests for RVV, Toy, and TensorExtLite
  still cover registered active options.
- [x] Current `--tcrv-source-artifact-front-door-pipeline` tests still cover
  registry-controlled behavior and disabled-plugin failure.
- [x] Focused lit for touched transform tests passes, or any failure is
  classified as an expected deletion gap without restoring old paths.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is recorded.

## Out of Scope

- Rebuilding source-front-door architecture.
- Restoring selected-boundary seed, source-seed, descriptor-driven
  computation, direct C semantic export, legacy wrappers, or unknown-option
  smoke matrices.
- New runtime correctness, performance, or `ssh rvv` claims.
- Broad unrelated test matrices.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
- Directly related source-front-door spec context also checked:
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-core-rvv-source-to-exec-named-absence-fixture-erasure/prd.md`.
- Relevant files read:
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `include/TianChenRV/Transforms/Passes.h`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`,
  `test/Transforms/Toy/toy-template-selected-boundary-seed-deleted.mlir`,
  `test/Transforms/Toy/toy-template-source-front-door.mlir`, and
  `test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir`.
- Initial exact old-option scan:
  `rg -n "tcrv-toy-materialize-template-selected-boundary-seed|toy-materialize-template-selected-boundary-seed|template-selected-boundary-seed|selected-boundary-seed" . --glob '!build/**' --glob '!artifacts/**'`.

## Completion Evidence

- Deleted `test/Transforms/Toy/toy-template-selected-boundary-seed-deleted.mlir`.
- Removed the only active lit invocation of
  `--tcrv-toy-materialize-template-selected-boundary-seed` and its
  `Unknown command line argument` FileCheck expectation.
- Preserved current active Toy source-front-door coverage in
  `test/Transforms/Toy/toy-template-source-front-door.mlir`, including the
  active disabled-builtins registration failure for
  `--tcrv-toy-materialize-template-source-front-door`.
- Preserved current RVV, TensorExtLite, and source-artifact front-door tests;
  no production compiler code changed.
- No compatibility alias, wrapper, legacy mode, descriptor adapter, source
  frontend, EmitC route, artifact route, direct C exporter, broader coverage,
  `ssh rvv` evidence, or Python compiler-core behavior was added.
- No `.trellis/spec/` update was needed: the existing lowering-runtime,
  extension-plugin, and plugin-protocol specs already encode the relevant
  source-front-door/deleted-route constraints; this task only removed a stale
  test fixture that was preserving old option-name public surface area.

## Checks

- [x] Exact selected-boundary/source-seed option scan over `test/Transforms`,
  `lib/Transforms`, `include/TianChenRV/Transforms`, and directly related
  specs:
  `rg -n "tcrv-[a-z0-9-]*selected-boundary-seed|selected-boundary-seed|source-seed-artifact-front-door-pipeline|tcrv-[a-z0-9-]*source-seed" test/Transforms lib/Transforms include/TianChenRV/Transforms .trellis/spec/lowering-runtime .trellis/spec/extension-plugins .trellis/spec/plugin-protocol`
  -> no matches.
- [x] Active unknown-option scan:
  `rg -n "Unknown command line argument" test/Transforms`
  -> only current disabled-builtins coverage remains for RVV, Toy, and
  TensorExtLite source-front-door options.
- [x] Active option scan over current RVV/Toy/TensorExtLite/SourceFrontDoor
  tests confirms current front-door and source-artifact pipeline coverage is
  retained.
- [x] Initial focused lit invocation from repo root failed because
  `build/test/lit.site.cfg.py` resolves `../../test/lit.cfg.py` relative to
  the working directory; reran from `build/test`.
- [x] Focused lit from `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter 'Transforms/(RVV/rvv-i32m1-vector-source-front-door\.mlir|Toy/toy-template-source-front-door\.mlir|TensorExtLite/tensorext-lite-fragment-mma-source-front-door\.mlir|SourceFrontDoor/source-artifact-front-door-pipeline-disabled\.mlir)$' .`
  -> 4/4 passed.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`
  -> 122/122 passed.
