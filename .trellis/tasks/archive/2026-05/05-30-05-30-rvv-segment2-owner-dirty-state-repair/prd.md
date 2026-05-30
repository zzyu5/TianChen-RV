# Repair dirty RVV segment2 owner state

## Goal

Restore repository coherence around the in-flight RVV EmitC route-family owner
work left dirty after the previous run. This is a repair task: classify every
dirty or untracked file, retain only verified production changes that are
already consumed by the compiler path, remove or revert stale artifacts, archive
task state truthfully, run focused checks, and commit the retained state.

## Direction Source

- Direction title: `Redirect: repair dirty RVV EmitC route-family owner state
  before new coverage`.
- Module owner: repository-coherent completion of the in-flight RVV EmitC
  segment2 route-family owner extraction, plus explicit cleanup or validated
  incorporation of unrelated untracked contraction-owner artifacts.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial HEAD: `b1ac374f rvv: move macc operand bindings to owner`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  direction brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## Dirty-State Inventory

Initial `git status --short` showed:

- Modified Trellis workspace notes:
  `.trellis/workspace/codex/index.md`,
  `.trellis/workspace/codex/journal-18.md`.
- Modified segment2 owner/planning/test files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/CMakeLists.txt`,
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- Untracked completed Trellis archive:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-segment2-route-family-owner-completion/`.
- Untracked completed Trellis archive:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-widening-contraction-operand-binding-owner-completion/`.
- Untracked contraction-owner source files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.
- New repair task directory:
  `.trellis/tasks/05-30-05-30-rvv-segment2-owner-dirty-state-repair/`.

## Resolution Classification

- Retained: segment2 owner/planning/test modifications. They move segment2
  operand-binding and provider-plan ownership into
  `RVVEmitCSegment2RouteFamilyPlanOwners` and leave central route planning as
  owner dispatch/shared validation.
- Retained: untracked contraction-owner header/cpp. They are already consumed
  by `CMakeLists.txt`, `RVVEmitCRoutePlanning.cpp`, and
  `RVVExtensionPluginTest.cpp`, so this repair incorporates them as verified
  production state instead of leaving active consumers pointing at untracked
  files.
- Retained: both pre-existing completed archive directories. They are truthful
  historical task records for the no-commit segment2 and widening-contraction
  rounds and must not remain untracked.
- Retained: Trellis workspace journal/index updates, with this repair session
  recorded separately from the previous no-commit segment2 session.
- Removed/reverted: none. Repository evidence showed the dirty production files
  are active and verified rather than stale or unreachable.

## Checks Performed

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-rvv-segment2-owner-dirty-state-repair`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Bounded added-line scans found no newly introduced source-front-door,
  descriptor, route-id/metadata, common-EmitC, exact-intrinsic, or legacy-i32
  authority in the touched owner/planning/test additions.
- Segment2 owner scans showed plan-ID constants and role mapping now live in
  `RVVEmitCSegment2RouteFamilyPlanOwners.cpp`; central route planning has no
  new segment2 semantic dispatch additions.
- Contraction scans showed active consumers through CMake, route-planning owner
  dispatch, and C++ tests.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 464/464
  tests.

## What I Already Know

- The latest run message reports the segment2 task as finished and archived but
  explicitly says no git commit was created and contraction-owner files remained
  untracked.
- The archived segment2 task PRD says its goal was to move segment2 operand
  binding and provider-plan ownership into
  `RVVEmitCSegment2RouteFamilyPlanOwners`, keeping central route planning
  neutral.
- The archived widening-contraction task PRD says its goal was to move
  widening-contraction operand-binding authority into a plugin-local owner
  module.
- The untracked contraction-owner source files are not passive leftovers:
  `CMakeLists.txt`, `RVVEmitCRoutePlanning.cpp`, and
  `RVVExtensionPluginTest.cpp` already include or call them. They must either
  be retained with focused verification or actively unwired and removed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires segment2
  route-family planning to go through one exact owner-built provider plan before
  segment2 statement-plan construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  stay neutral and consume provider-built routes; it must not choose RVV
  semantics, dtype, schedule, ABI role semantics, or intrinsic spellings.
- `.trellis/spec/testing/mlir-testing-contract.md` requires C++ tests for
  plugin/provider APIs and real `ssh rvv` evidence only when runtime,
  correctness, or performance claims are made.

## Requirements

1. Classify every dirty and untracked file as retained, removed, reverted, or
   task/archive state.
2. Retain the segment2 owner extraction only if the compiler path is genuinely:
   selected RVV route facts -> segment2 owner provider plan -> neutral central
   dispatch/registration -> provider route facts -> `TCRVEmitCLowerableRoute`.
3. Ensure central `RVVEmitCRoutePlanning.cpp` keeps only neutral owner dispatch
   or shared generic validation for segment2 owner facts, not segment2 semantic
   authority after the owner boundary.
4. Decide the untracked contraction-owner artifacts explicitly. If retained,
   prove they have active consumers and focused checks; if removed, unwind all
   CMake/includes/calls/tests that depend on them.
5. Do not start new RVV coverage, new dtype/LMUL families, direct-route
   demotion, source-front-door positives, dashboards, reports, or unrelated
   refactors.
6. Keep Trellis archive state truthful: completed archived tasks may be retained
   as historical records, but no task/archive directory may remain untracked at
   closeout.
7. Run focused build/test checks and self-repair failures before finalizing.
8. End with a clean `git status --short` and one coherent commit if production
   changes are retained.

## Acceptance Criteria

- [x] Dirty inventory is recorded and every file is classified.
- [x] Segment2 owner files own segment2 operand-binding plan IDs, role lookup,
      binding-plan derivation, provider-plan verification, and provider-plan
      construction for the active segment2 families.
- [x] Central route planning no longer keeps segment2-specific plan-ID
      constants, role mapping bodies, or local segment2 binding derivation
      cases except neutral owner dispatch.
- [x] Segment2 statement-plan/provider-plan paths fail closed when required
      segment2 owner facts, route-family plans, materialization facts,
      operand-binding facts, runtime-control facts, operation/memory-form facts,
      or ABI order facts are stale or missing.
- [x] Contraction-owner artifacts are either removed with their consumers
      unwired, or retained as active consumed production repair with CMake,
      route-planning, and C++ test coverage.
- [x] The two pre-existing untracked archive directories are either committed
      as truthful archives or removed as stale; they are not left untracked.
- [x] Focused RVV plugin build passes:
      `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
      or stronger.
- [x] Focused RVV plugin binary passes:
      `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [x] Focused scans prove no new central ad hoc, metadata-derived,
      descriptor-derived, route-id-derived, common-EmitC-derived,
      source-front-door-derived, direct-route-entry-only,
      pre-realized-fixture-only, exact-intrinsic-derived, or legacy-i32-derived
      authority was introduced in the touched owner/planning/test files.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes if feasible, or the exact blocker is recorded.
- [x] The repair task is finished and archived truthfully.
- [x] Final `git status --short` is clean.
- [x] A commit exists for retained production changes, or all attempted changes
      are removed/reverted and no commit is needed.

## Validation Plan

1. Validate this task context with
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-rvv-segment2-owner-dirty-state-repair`.
2. Build the focused plugin test target:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run the focused plugin binary:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run bounded owner/authority scans over touched source and test files.
5. Run `git diff --check`.
6. Run `cmake --build build --target check-tianchenrv -j2` if feasible; record
   the exact blocker if not.
7. Archive the repair task, commit the retained repair state, and confirm
   `git status --short` is clean.

## Out of Scope

- New RVV coverage or feature expansion.
- New direct-route demotion work.
- New dtype, LMUL, or source-shape family batches.
- Broad audits, dashboards, readiness reports, or report-only progress.
- Moving common EmitC or central planning into RVV semantic ownership.
- Runtime, correctness, or performance claims requiring `ssh rvv` unless this
  repair changes emitted runtime behavior.

## Technical Notes

Read before implementation/check:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/workspace/codex/journal-18.md`
- `artifacts/tmp/hermes_codex_supervisor/runs/20260529T021235Z-r0047-20260530T020302Z/last_message.md`
- `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-segment2-route-family-owner-completion/prd.md`
- `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-widening-contraction-operand-binding-owner-completion/prd.md`

## Definition Of Done

The repository has no untracked or dirty repair residue, retained owner changes
are verified and committed, stale artifacts are removed or explicitly retained,
Trellis archive state is truthful, and final status is clean.
