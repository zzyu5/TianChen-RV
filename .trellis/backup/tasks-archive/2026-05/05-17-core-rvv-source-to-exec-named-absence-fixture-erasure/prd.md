# Core RVV source-to-exec named-absence fixture erasure

## Goal

Delete active named-absence fixtures and stale test/spec wording that still
protect historical core RVV/source-seed public option spellings as durable
workflow surfaces. The current valid coverage must stay centered on the
plugin-owned RVV vector source front door and the source artifact front-door
pipeline, not on deleted option-name compatibility.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD `d6c21b9 translate: delete
  plan-and-export bundle wrapper`; worktree clean; no `.trellis/.current-task`
  existed.
- The previous archived task removed the translate-side
  `--tcrv-plan-and-export-target-artifact-bundle` wrapper and updated specs so
  durable named absence probes should not remain as production workflow
  coverage.
- Active tests still invoke deleted public option spellings only to prove they
  are unknown:
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed` in
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`, and
  `--tcrv-source-seed-artifact-front-door-pipeline` in
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`.
- Current source-front-door coverage still needs to preserve:
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door` and
  `--tcrv-source-artifact-front-door-pipeline`.
- The relevant specs already classify deleted source/linalg/vector option
  names as old route residue and say tests should delete old named absence
  fixtures rather than preserve them as production contracts.

## Boundaries

- Deletion/refactor-only. Deletion before rebuild is the campaign rule.
- Work only in tests/spec/comments directly protecting deleted source-to-exec,
  source-seed, linalg/vector RVV, compatibility-option, selected-boundary seed,
  or vector i32 arithmetic public option spellings as named absence fixtures.
- If a touched file also has current valid coverage, keep the current
  plugin-owned source-front-door assertions and remove only old-option RUN
  lines/check prefixes.
- Do not change production compiler behavior except to remove dead references
  if a directly related stale test/comment requires it.
- Do not add a replacement source-to-exec route, selected-boundary seed route,
  source-seed pipeline alias, linalg/vector compatibility option, descriptor
  adapter, wrapper, quarantine mode, direct C exporter, new RVV family, new
  source frontend, new EmitC route, new artifact route, Python compiler-core
  behavior, or replacement unknown-option smoke matrix.
- Do not require new `ssh rvv` evidence because this round removes named
  absence fixtures and does not change runtime correctness or performance.

## Requirements

- Remove active lit invocations of
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed` as a named absence
  fixture.
- Remove active lit invocations of
  `--tcrv-source-seed-artifact-front-door-pipeline` as a named absence fixture.
- Remove stale `OLD-*-REMOVED` FileCheck prefixes tied to those deleted option
  probes.
- Preserve truthful positive and negative coverage for
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door`.
- Preserve truthful negative coverage for
  `--tcrv-source-artifact-front-door-pipeline` when builtin plugins are
  disabled or no `tcrv.exec.kernel` is present.
- Update directly related spec/comment wording if it still encourages durable
  named absence fixtures for historical source/linalg/vector/source-seed option
  spellings.

## Acceptance Criteria

- [x] Active lit tests under `test/Transforms/RVV` and
  `test/Transforms/SourceFrontDoor` no longer invoke historical source,
  source-seed, linalg RVV, linalg i32 compatibility, selected-boundary seed, or
  vector i32 arithmetic deleted options as named absence fixtures.
- [x] Current plugin-owned
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door` coverage remains
  parseable and truthful.
- [x] Current `--tcrv-source-artifact-front-door-pipeline` coverage remains
  parseable and truthful.
- [x] Specs/comments no longer encourage durable named absence fixtures for the
  deleted old option families; any remaining old-option-family mention is only
  high-level deletion-rule prose, not an executable workflow contract.
- [x] Focused scans over touched transform test directories and changed specs
  prove the exact old option spellings and `OLD-*-REMOVED` style check prefixes
  are gone or reduced to non-executable deletion-rule prose.
- [x] Focused lit for touched RVV and SourceFrontDoor transform tests passes,
  or failures are classified as expected deletion gaps without restoring old
  options.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` is run if practical; otherwise the reason is recorded.

## Out of Scope

- Rebuilding high-level frontend lowering.
- Restoring selected-boundary seed, source-seed, linalg/vector compatibility,
  descriptor-driven computation, direct C semantic export, source-to-exec
  wrappers, or unknown-option smoke matrices.
- Any RVV runtime, correctness, or performance claim.
- Broad unrelated test matrices.

## Technical Notes

- Specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-17-plan-and-export-target-artifact-bundle-route-erasure/prd.md`.
- Relevant files read:
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`,
  `test/Transforms/RVV/`,
  and `test/Transforms/SourceFrontDoor/`.
- Initial residue scan:
  `rg -n "tcrv-rvv-materialize-i32m1-selected-boundary-seed|tcrv-source-seed-artifact-front-door-pipeline|OLD-[A-Z0-9_-]*REMOVED|selected-boundary-seed|source-seed-artifact|source-to-exec|linalg RVV|vector i32" test/Transforms/RVV test/Transforms/SourceFrontDoor .trellis/spec/testing .trellis/spec/variant-pipeline .trellis/spec/lowering-runtime`.

## Completion Evidence

- Removed the deleted selected-boundary seed named absence fixture from
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`:
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed` and the
  `OLD-SEED-REMOVED` FileCheck prefix are gone.
- Removed the deleted source-seed pipeline named absence fixture from
  `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir`:
  `--tcrv-source-seed-artifact-front-door-pipeline` and the
  `OLD-PIPELINE-REMOVED` FileCheck prefix are gone.
- Preserved current RVV source materializer coverage using
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door`.
- Preserved current source artifact front-door pipeline coverage using
  `--tcrv-source-artifact-front-door-pipeline`.
- No production compiler code, replacement route, compatibility alias,
  descriptor adapter, wrapper, direct C exporter, new RVV family, source
  frontend, EmitC route, artifact route, or Python compiler-core behavior was
  added.
- No `.trellis/spec/` edit was needed: the relevant testing,
  variant-pipeline, and lowering-runtime specs already state the no durable
  named absence fixture rule; remaining old-family mentions are high-level
  deletion-rule prose rather than executable workflow contracts.

## Checks

- [x] Initial direct `llvm-lit` command failed because `llvm-lit` is not on
  PATH.
- [x] Direct source-path lit invocation with
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py` failed because
  source `test/lit.cfg.py` lacks generated site config attributes outside
  `build/test`; reran through `build/test/lit.site.cfg.py`.
- [x] Focused lit through build site config:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter 'Transforms/(RVV/rvv-i32m1-vector-source-front-door\.mlir|SourceFrontDoor/source-artifact-front-door-pipeline-disabled\.mlir)$' /home/kingdom/phdworks/TianchenRV/build/test`
  -> 2/2 passed.
- [x] Exact old-option/prefix scan:
  `rg -n "tcrv-rvv-materialize-i32m1-selected-boundary-seed|tcrv-source-seed-artifact-front-door-pipeline|OLD-[A-Z0-9_-]*REMOVED" test/Transforms/RVV test/Transforms/SourceFrontDoor .trellis/spec/testing .trellis/spec/variant-pipeline .trellis/spec/lowering-runtime; test $? -eq 1`
  -> no matches.
- [x] Focused active option scan over `test/Transforms/RVV` and
  `test/Transforms/SourceFrontDoor` reports only current
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door` and
  `--tcrv-source-artifact-front-door-pipeline` invocations.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2` -> 123/123 passed.
