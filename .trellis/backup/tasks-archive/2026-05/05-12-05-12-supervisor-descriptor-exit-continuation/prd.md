# Scalar selected-boundary descriptor-only quarantine

## Goal

Close the remaining scalar fallback selected-boundary gap in the descriptor-exit
mainline. A selected scalar fallback variant that carries only legacy
`tcrv_scalar.lowering_descriptor` / `tcrv_scalar.element_count` metadata must
fail closed during selected lowering-boundary materialization instead of
producing a plugin-local `tcrv_scalar.lowering_boundary`. Descriptor mirrors may
remain only after a typed scalar microkernel body is already the selected-path
authority.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; task-start worktree was clean.
- Task-start HEAD is
  `cb887be feat(rvv): quarantine selected-boundary descriptor mirror`.
- `.trellis/.current-task` was absent, so this task was created as
  `.trellis/tasks/05-12-05-12-supervisor-descriptor-exit-continuation/`.
- The latest supervisor audit/review input for run
  `20260511T-resume-descriptor-exit-2-r0032-20260512T060534Z` reports a clean
  worktree and no active task after the RVV selected-boundary descriptor mirror
  quarantine commit.
- RVV selected-boundary already rejects descriptor-only direct RVV binary
  materialization. Scalar selected emission/readiness also no longer derives
  supported source plans from scalar descriptors.
- Current scalar selected-boundary behavior still accepts a valid descriptor-only
  scalar fallback variant and materializes a metadata-only
  `tcrv_scalar.lowering_boundary`, while deliberately not creating a typed
  scalar microkernel.
- The scalar fallback spec and emission-runtime spec still contain stale text
  describing descriptor-carried scalar proposal/boundary materialization.

## Requirements

- Rewire `ScalarExtensionPlugin::materializeSelectedLoweringBoundary` so
  descriptor-only scalar fallback selected paths fail closed before
  `tcrv_scalar.lowering_boundary` creation.
- Keep the descriptorless typed scalar fallback default path working for the
  bounded finite i32/i64 add/sub/mul families: boundary plus exactly one typed
  `tcrv_scalar.*_microkernel` must still materialize and feed supported
  emission plans.
- Keep explicit hand-authored typed scalar microkernel paths working.
- Keep `tcrv_scalar.lowering_descriptor` and `tcrv_scalar.element_count` only as
  optional legacy mirror metadata validated after typed scalar microkernel body
  authority is identified; stale mirrors must continue to fail.
- Update tests that currently expect descriptor-only scalar boundary
  materialization so they instead expect a selected-boundary failure and no
  descriptor-derived scalar microkernel.
- Add focused lit coverage for a valid descriptor-only scalar fallback fixture
  failing during selected lowering-boundary materialization.
- Correct durable scalar fallback and emission-runtime spec text so future work
  does not reintroduce descriptor-driven scalar boundary materialization.
- Keep the change plugin-local; do not add scalar semantic branches to core
  orchestration.

## Acceptance Criteria

- [x] Descriptor-only scalar fallback selected lowering-boundary materialization
      fails with a bounded diagnostic naming legacy descriptor-only metadata.
- [x] Descriptorless typed scalar fallback selected paths for existing finite
      i32/i64 add/sub/mul families still materialize typed scalar microkernel
      ops and supported emission plans.
- [x] Explicit typed scalar microkernel selected paths still produce supported
      scalar emission plans.
- [x] Stale legacy scalar descriptor mirrors beside typed scalar bodies continue
      to fail as mirror mismatches, not as family authority.
- [x] Scalar plugin C++ tests are updated and pass.
- [x] Focused lowering-boundary / scalar artifact lit tests are updated and
      pass.
- [x] `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` and relevant
      emission-runtime text describe descriptorless typed scalar authority and
      descriptor-only quarantine accurately.
- [x] `git diff --check`, Trellis task validation, and focused build/test
      checks pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
      -j2` passes if focused checks pass and the existing build tree is usable.

## Non-Goals

- No new scalar/RVV dtype, arithmetic family, runtime scheduler, benchmark,
  performance claim, or broad evidence matrix.
- No `ssh rvv` claim; this task changes compiler selected-boundary behavior
  only.
- No descriptor-to-C exporter and no descriptor-driven compute semantics.
- No Python compiler implementation.
- No compute semantics in `tcrv.exec`.
- No IME, offload, Template, Toy, RVV source-authority, or dispatch redesign
  beyond tests/spec text required by this scalar selected-boundary change.
- No helper-only, report-only, probe-only, or docs-only closeout.

## Technical Approach

Add a scalar plugin-local check after scalar plugin legality but before boundary
creation: if the selected variant has legacy scalar descriptor or
descriptor-local element-count metadata and no matching typed scalar
microkernel body was found for the selected path, reject it as descriptor-only
legacy metadata. The existing descriptorless default typed materialization
continues to create the typed scalar microkernel for normal planned paths; the
existing mirror validation remains in `findMatchingExplicitMicrokernelFamily`
and scalar target/export validation after a typed body is present.

## Minimal Validation

- Build and run `tianchenrv-scalar-extension-plugin-test`.
- Focused lit for `test/Transforms/LoweringBoundary/scalar-*`,
  `test/Transforms/EmissionReadiness/materialize-emission-plans-scalar-microkernel.mlir`,
  and touched scalar artifact/export fixtures.
- Run `git diff --check`.
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-05-12-supervisor-descriptor-exit-continuation`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` after focused checks pass if the build tree remains usable.

## Definition Of Done

This task is complete only when scalar fallback descriptor-only selected-boundary
materialization fails closed, typed scalar fallback selected-boundary and
emission-plan paths still work, durable specs match the new contract, focused
checks pass, Trellis state is truthful, and one coherent commit records the
module. If unfinished, leave the task open with the exact remaining file,
function, fixture, or failing command.

## Completion Notes

- `ScalarExtensionPlugin::materializeSelectedLoweringBoundary` now rejects a
  selected scalar fallback variant that carries only legacy
  `tcrv_scalar.lowering_descriptor` and/or `tcrv_scalar.element_count`
  metadata before creating `tcrv_scalar.lowering_boundary`.
- Descriptorless typed scalar default paths and explicit typed scalar
  microkernel paths remain supported and continue to feed scalar source export
  plans.
- Optional legacy scalar descriptor mirrors are still validated only after
  typed scalar microkernel authority exists; stale mirrors remain fatal mirror
  mismatches.
- Focused C++/lit checks, `git diff --check`, Trellis validation, and full
  `check-tianchenrv` passed on the existing build tree.
