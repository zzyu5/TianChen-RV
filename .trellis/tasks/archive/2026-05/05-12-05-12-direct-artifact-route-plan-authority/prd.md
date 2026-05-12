# Direct artifact route planning authority

## Goal

Make marked artifact-backed direct `tcrv-translate` routes authoritative through
the common execution-planning, coherence, and exact-route target artifact export
path. A production direct RVV or RVV+scalar source/header/object command must no
longer fall through to descriptor-only or selected-boundary-only target-local C
emission when the route is marked with `targetArtifactRouteID`.

## What I Already Know

* Current clean HEAD is `9913662 feat(translate): route direct artifacts through
  coherence`.
* The predecessor task
  `.trellis/tasks/archive/2026-05/05-12-05-12-direct-target-translate-route-artifact-coherence/`
  marked direct artifact-backed routes and routed emission-plan-backed inputs
  through `checkExecutionPlanCoherence` and the generic exact-route target
  artifact exporter.
* That predecessor intentionally preserved selected-boundary-only direct
  target-local callbacks when no emission-plan metadata exists. This task closes
  that remaining production fallback.
* Artifact-backed direct route entries are identified by
  `TargetTranslateRoute::targetArtifactRouteID`.
* Explicit self-check, smoke, and probe helper routes can remain target-local
  only when they are unmarked and not artifact-backed production front doors.

## Requirements

* For any direct `TargetTranslateRoute` with `targetArtifactRouteID`,
  source/header/object export must not invoke target-local descriptor or
  selected-boundary-only callbacks as the production path.
* If the input module already has emission-plan diagnostics, preserve the
  current behavior: populate built-in plugin/exporter registries, run
  `checkExecutionPlanCoherence`, and export through the generic exact-route
  target artifact exporter.
* If the input module lacks emission-plan diagnostics, run the bounded planning
  path appropriate for the public translate front door so selected lowering
  boundaries and emission-plan diagnostics are materialized before export.
* If planning cannot produce a coherent exact selected candidate group for the
  marked route, fail closed with a diagnostic before any descriptor-driven or
  selected-boundary direct C/header/object output.
* Keep generic `tcrv-translate` code family-neutral. It may inspect route
  metadata and call common planning/coherence/export APIs, but it must not
  branch on RVV/scalar family names, arithmetic op names, dtypes, vector
  shapes, intrinsic names, or dispatch semantics.
* Preserve explicit self-check and smoke/probe helpers as target-local harness
  routes only when unmarked; their route names and diagnostics should not imply
  that they are production artifact front doors.
* Update or delete tests that require marked artifact-backed direct routes to
  emit from descriptor-only or selected-boundary-only state. Retain only
  explicitly quarantined helper tests for unmarked harness behavior.

## Acceptance Criteria

* [x] Marked direct RVV source/header/object routes with no pre-existing
      emission-plan metadata self-plan into the generic exact-route exporter or
      fail closed before target-local descriptor/selected-boundary output.
* [x] Marked direct RVV+scalar source/header/object routes with no pre-existing
      emission-plan metadata self-plan into the generic exact-route exporter or
      fail closed before target-local descriptor/selected-boundary output.
* [x] Marked direct routes that already contain emission-plan diagnostics still
      run built-in registry population, execution-plan coherence, and generic
      exact-route export.
* [x] Stale route metadata is rejected by planning/coherence/exact-route logic,
      not by descriptor-only or target-local direct callback emission.
* [x] Unmarked self-check, smoke, and probe helper routes remain target-local
      helpers and are not routed through artifact-backed production authority.
* [x] Focused FileCheck coverage proves at least one marked direct route without
      pre-existing emission-plan metadata reaches plan-and-export or fails
      closed before descriptor/selected-boundary direct C output.
* [x] Focused negative coverage proves stale route metadata rejection happens
      through planning/coherence/exact-route logic.
* [x] Focused C++/lit checks, `git diff --check`, `git diff --cached --check`,
      and Trellis validation pass.

## Completion Notes

* `tcrv-translate` now routes every marked `TargetTranslateRoute` with a
  `targetArtifactRouteID` through common planning/coherence/exact target
  artifact export before invoking any route-local callback.
* No-plan marked direct routes choose a bounded route by available IR state:
  existing selected lowering boundaries get emission-plan materialization plus
  coherence; existing selected variants get lowering-boundary materialization,
  emission-plan materialization, and coherence; raw capability-only modules run
  the execution-planning pipeline before exact-route export.
* Route-local descriptor/selected-boundary callbacks remain reachable only for
  unmarked helper/harness routes such as explicit self-check exports.
* Focused lit now covers raw no-plan RVV direct source planning, selected
  variant-to-boundary RVV direct planning, RVV+scalar no-plan source/header
  planning, RVV+scalar no-plan object fail-closed planning diagnostics, stale
  exact-route rejection, and unmarked self-check behavior.

## Non-Goals

* No new arithmetic families, RVV kernels, broad smoke matrix, or standalone
  `ssh rvv` evidence.
* No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.
* No descriptor-driven computation or direct descriptor-to-C production
  authority.
* No compute semantics in `tcrv.exec`.
* No MLIR-vector, LLVM-RVV, inline assembly, or backend detour.
* No Hermes prompt, runner policy, or supervisor workflow changes.
* No runtime correctness or performance claims without fresh exact-slice
  `ssh rvv` evidence.

## Minimal Validation

* Build touched targets, expected:
  `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* From `build/test`, run focused lit with filter
  `RVVScalarDispatch|RVVMicrokernel|target-source-artifact-routes|target-artifact-export-registry|plan-and-export`.
* Run `git diff --check` and `git diff --cached --check`.
* Validate the active Trellis task before finish and the archived task if
  completed.
* Run full `check-tianchenrv` only if focused checks leave uncertainty around
  shared route registration or planning behavior.

## Technical Notes

* Specs read or required:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/unified-riscv-mlir.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Primary code surfaces:
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `include/TianChenRV/Target/TargetTranslateRegistration.h`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`.
