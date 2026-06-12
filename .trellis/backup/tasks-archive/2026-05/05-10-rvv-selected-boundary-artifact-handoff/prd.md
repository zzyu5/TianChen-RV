# Manifest-backed RVV selected-boundary artifact handoff

## Goal

Complete one bounded compiler handoff where a selected finite RVV or RVV+scalar
dispatch path carries a target-manifest-derived artifact route reference from
the selected lowering-boundary / emission-plan boundary into target artifact
export validation. This follows the completed manifest ownership cleanup in
`rvv-binary-artifact-manifests`: the next module behavior is not another route
count cleanup, but proof that the selected compiler path and target exporter
consume the same target-owned manifest route identity.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial state for this round is clean at
  `fc314c1 refactor(rvv): derive binary artifact manifests from families`.
* `.trellis/.current-task` was missing, so this task was created from the
  Hermes brief.
* The previous task is finished and archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-artifact-manifests`; it must
  not be reopened.
* The previous task made direct RVV microkernel and RVV+scalar dispatch route
  manifests target-owned and family-derived.
* The finite route family set remains exactly `i32-vadd`, `i32-vsub`,
  `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
* This round should prefer the already evidenced `i64-vmul` RVV+scalar
  dispatch path if current repository fixtures and route support make it the
  smallest coherent executable handoff; otherwise it should choose the smallest
  existing finite RVV binary path with focused fixtures.

## Requirements

* Inventory the current compiler path from marked finite RVV binary frontend IR
  through execution planning, selected lowering-boundary materialization,
  emission-plan materialization, and target artifact export.
* Pick one representative finite selected path for executable handoff coverage,
  with preference for `i64-vmul` RVV+scalar dispatch when repository evidence
  supports it.
* Ensure selected-boundary or emission-plan metadata records route identity
  using target-owned manifest APIs, not duplicated route strings reconstructed
  in tests, scripts, or generic code.
* Ensure target artifact export validates that the route referenced by the
  selected boundary/emission plan exists in the appropriate direct RVV or
  RVV+scalar dispatch manifest before exporting.
* Add or tighten focused C++ and/or lit coverage proving the selected compiler
  metadata and target exporter agree on the manifest route for the chosen finite
  path.
* Move ownership into C++ target/export code only where scripts or tests are
  reconstructing compiler-owned route facts; Python may remain orchestration or
  artifact parsing only.
* Preserve the distinct direct RVV microkernel and RVV+scalar dispatch route
  families and the exact finite family set.
* If the complete pipeline-to-object handoff is too large, finish one coherent
  submodule: selected boundary/emission plan to source/header export validation
  for one finite path, keep the task open, and record the next continuation
  point.

## Acceptance Criteria

* [x] Current selected-path to artifact-export flow is documented in this task
  and reflected in code/test choices.
* [x] One representative finite family/path has manifest-derived route identity
  in selected-boundary or emission-plan metadata.
* [x] The target artifact exporter validates the referenced route through the
  relevant target-owned manifest API before artifact output.
* [x] Focused coverage proves route identity agreement between selected
  compiler metadata and target exporter for the chosen path.
* [x] No generic core pass gains RVV/scalar dtype, family, route, or intrinsic
  semantics; generic code remains route-id/artifact-kind/plugin-interface
  driven.
* [x] No new RVV runtime, correctness, or performance claim is made without
  fresh real `ssh rvv` evidence.
* [x] Focused build/tests pass, or any unfinished edge is left open with the
  smallest next continuation point recorded.

## Definition Of Done

* `git diff --check` passes.
* Affected tools/tests build, at minimum `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`, and the focused RVV
  planning/registry test target touched by the change.
* Focused C++ test binaries covering target artifact export and RVV binary
  planning/manifest behavior pass.
* Focused lit from `artifacts/tmp/tianchenrv-build/test` covers the selected
  boundary or execution-planning path, target artifact export, and the relevant
  RVV microkernel or RVV scalar dispatch bundle fixture.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run if public tool, pass pipeline, target exporter, or route registration
  code changes.
* Active task validates before finish; archived task validates after finish if
  the task is completed.
* One coherent commit is created only when the acceptance criteria are
  truthfully satisfied.

## Out Of Scope

* New RVV family, dtype, vector shape, mask/tail policy, backend claim, runtime
  integration, correctness claim, or performance claim.
* Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission, artifact selection, or route semantics.
* Compute semantics in `tcrv.exec`.
* Extension-specific semantic branches in generic core passes.
* Broad smoke matrices, dashboards, report-only work, route-manifest-only
  cleanup, or task-metadata-only closeout.
* Weakening existing bundle evidence tests to fit the new handoff.

## Technical Approach

Use the previous manifest APIs as the source of truth at the selected
emission-plan/exporter boundary. The concrete route should be looked up from
the selected family descriptor and route kind through the target-owned direct
RVV or RVV+scalar dispatch manifest, then serialized or validated as bounded
compiler metadata. The target artifact front door should fail closed when a
selected emission plan names a route that is absent from the manifest or does
not match the selected family/path.

The first implementation choice is to inspect whether the existing
`i64-vmul` RVV+scalar dispatch fixtures already flow through frontend lowering,
execution planning, supported emission plans, and artifact export. If that path
is not currently complete enough for a focused handoff, use the smallest
existing finite direct RVV or RVV+scalar path with established fixtures and
keep the scope to route identity handoff and validation.

## Completed Handoff

The representative path for this round is the existing finite `i64-vmul`
RVV+scalar dispatch fixture. The current flow is:

1. `tcrv_frontend_lowering = "i64-vmul"` reaches the execution-planning
   pipeline and materializes the selected RVV dispatch case and scalar fallback
   boundaries.
2. RVV and scalar plugin emission plans preserve the selected callable artifact
   routes in `lowering_pipeline` metadata:
   `tcrv-export-rvv-i64-vmul-microkernel-c` and
   `tcrv-export-scalar-i64-vmul-microkernel-c`.
3. The RVV target preflight now resolves selected RVV callable routes through
   `lookupRVVMicrokernelDirectRoute(...)` before accepting the route as a
   direct source artifact route.
4. The RVV+scalar dispatch target preflight now resolves dispatch source,
   header, object, and self-check routes through
   `lookupRVVScalarDispatchRoute(...)` before export. Generated source/header
   artifacts print the dispatch manifest route id and artifact kind used by the
   target exporter.
5. Generic artifact export remains target-neutral: no generic core pass gained
   RVV family, dtype, intrinsic, or route branches.

The full source/header handoff for the selected `i64-vmul` dispatch path is
complete in this round. No new runtime, correctness, or performance claim was
made, so no fresh `ssh rvv` run was required.

## Validation Evidence

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
* Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-i64-vmul-dispatch-generic-route|rvv-scalar-dispatch-bundle-e2e|rvv-microkernel-bundle-e2e'`
* Broader focused artifact/planning lit from
  `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TargetArtifact|target-artifact|artifact-export|execution-planning|ExecutionPlanning|rvv-binary-planning'`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed all `193/193` lit tests.

## Decision (ADR-lite)

**Context**: Route manifests are now target-owned and family-derived, but the
selected compiler path still needs to prove it consumes those route identities
instead of reconstructing them outside the manifest boundary.

**Decision**: Add manifest-backed route identity at the selected
boundary/emission-plan to artifact-export handoff for one finite executable
path, and validate that identity through target-owned C++ manifest APIs before
export.

**Consequences**: The round stays bounded to compiler/export metadata
coherence. It may leave broader object/runtime evidence for a later task unless
the selected path already has the needed fixtures and local tool support.

## Technical Notes

* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Prior task context read:
  * `.trellis/tasks/archive/2026-05/05-10-rvv-binary-artifact-manifests/prd.md`
* Workspace journal context read:
  * `.trellis/workspace/codex/journal-2.md`
