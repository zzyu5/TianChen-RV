# Scalar finite fallback header/object artifact routes

## Goal

Complete the finite scalar fallback target artifact route surface so the
existing i32/i64 add/sub/mul scalar callable source candidates can also feed
matching header and relocatable-object artifact helpers through the generic
target artifact front doors. This keeps scalar fallback artifacts bounded to
the current plugin-owned finite family and prevents sub/mul/i64 scalar
callables from being source-only while RVV already has source/header/object
parity.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `fea02ff feat(rvv): register selected exporters via plugin bundle`.
- No `.trellis/.current-task` existed at session start; the previous archived
  task `05-11-rvv-plugin-owned-target-exporter-bundle` finished with no
  continuation point.
- The latest supervisor `repo_audit.md` / `review_input.md` show the previous
  RVV plugin-owned exporter task was archived and committed.
- Current scalar source artifact routes already cover finite i32/i64
  add/sub/mul through `registerScalarMicrokernelTargetExporters`.
- Current scalar header/object artifact registration is still bounded to the
  legacy add route ids `tcrv-export-scalar-microkernel-header` and
  `tcrv-export-scalar-microkernel-object`.
- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` explicitly
  states that scalar header/object helper routes remain bounded to the add
  callable until a later task adds matching subtract helper routes and tests.

## Requirements

- Add family-derived scalar header and object route ids for the finite
  i32/i64 add/sub/mul scalar fallback families.
- Preserve the legacy i32-vadd header/object route ids for compatibility:
  `tcrv-export-scalar-microkernel-header` and
  `tcrv-export-scalar-microkernel-object`.
- Register scalar header and object composite artifact exporters for every
  existing finite scalar family, using the same target-owned C++ validation and
  exporter implementation.
- Ensure header/object composite matching validates that the selected source
  candidate belongs to the same scalar family as the requested helper route,
  so stale add/sub/mul or i32/i64 route mismatches fail before output.
- Keep generic target artifact export, execution-plan coherence, plugin
  registry, and `tcrv.exec` free of scalar-family semantic branches.
- Preserve existing RVV, Toy, offload, scalar source, and RVV+scalar dispatch
  behavior unless a narrow compile/test update is required by the scalar route
  expansion.
- Update durable scalar/lowering-runtime specs only for the actual route
  boundary change.

## Non-Goals

- No generic scalar backend, arbitrary scalar lowering, linker/runtime
  integration, runtime execution, correctness claim, or performance claim.
- No new arithmetic families beyond the existing finite i32/i64 add/sub/mul
  scalar fallback descriptors.
- No RVV runtime/correctness/performance claim and no required `ssh rvv`
  evidence.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or target artifact export.
- No compute semantics in `tcrv.exec`.
- No smoke/probe/guardrail/test-harness-only closeout as the main deliverable.

## Acceptance Criteria

- [x] The scalar family descriptor derives distinct header/object route ids for
      i32/i64 sub/mul and i64 add, while preserving the legacy i32-vadd ids.
- [x] `registerScalarMicrokernelTargetExporters` registers source, header, and
      object target artifact routes for all finite scalar fallback families.
- [x] Header/object route matching is family-specific and rejects stale source
      candidate families before header/object output.
- [x] Built-in target artifact registry counts and route coverage tests are
      updated to reflect the expanded scalar header/object route set.
- [x] Focused C++ tests prove representative scalar sub/mul or i64
      header/object registration and stale-family rejection.
- [x] Focused lit/FileCheck coverage proves at least one non-add scalar
      header/object generic target artifact route reaches `tcrv-translate`
      through the selected-plan/front-door path.
- [x] Specs record the new finite scalar header/object route boundary without
      turning scalar fallback into a generic backend or runtime claim.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass; `check-tianchenrv -j2` is run if feasible.

## Minimal Validation

- `git diff --check`
- Build `tcrv-opt`, `tcrv-translate`, and
  `tianchenrv-target-artifact-export-test`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit tests that exercise non-add scalar header/object artifact
  front doors.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  after focused checks pass if feasible.
- Validate this Trellis task path before finish/archive.

## Definition of Done

- Source changes are implemented in C++/MLIR/CMake/lit/FileCheck as
  appropriate.
- PRD acceptance criteria and minimal validation are satisfied, or the task
  remains open with a precise continuation point.
- Trellis task context and workspace journal truthfully record the outcome.
- The task is finished/archived only after focused validation and one coherent
  commit, unless a blocker is documented precisely.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`,
  and `.trellis/spec/testing/mlir-testing-contract.md`.
- Key code surfaces:
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  and focused scalar target artifact lit tests.
