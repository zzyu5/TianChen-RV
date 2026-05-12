# Direct target translate route artifact coherence

## Goal

Rewire the direct `tcrv-translate --tcrv-export-rvv-*` and
`--tcrv-export-rvv-scalar-*` source/header/object route-family commands so the
production emission-plan path crosses the same generic target artifact exporter
and execution-plan coherence surface as `--tcrv-export-target-*`.

This task is a compiler route migration. It must change the active
`tcrv-translate` behavior for emission-plan-backed source/header/object
exports, not only add helper APIs, tests, comments, or evidence.

## What I already know

* Current clean HEAD is `e37a86e chore: harden Hermes review turn budget`.
* Recent committed work added the common lower-to-EmitC source boundary and
  rewired scalar, RVV, and RVV+scalar dispatch source bodies through it.
* `tcrv-translate --tcrv-export-target-source-artifact`,
  `--tcrv-export-target-header-artifact`, `--tcrv-export-target-artifact`, and
  bundle export already populate built-in plugin/exporter registries and run
  `checkExecutionPlanCoherence` before target-owned artifact emission.
* Direct route-family commands are registered through
  `TargetTranslateRouteRegistry` and currently call target-owned callbacks
  directly. That keeps direct RVV/RVV+scalar source/header/object helpers
  separate from the generic route metadata and coherence surface.
* The direct route registry also contains explicit self-check helper routes for
  RVV+scalar dispatch; those are evidence helpers and are not generic target
  artifact front doors.
* Boundary-only direct helper tests still exist and should remain a bounded
  target-local path when no emission-plan metadata is present.

## Requirements

* Add a generic exact-route target artifact export API that selects one
  registered standalone or composite target artifact route by route id,
  validates the matching candidate(s), and invokes the registered exporter.
* Mark direct source/header/object `TargetTranslateRoute` entries with their
  corresponding target artifact route id.
* In `tcrv-translate`, when a marked direct route sees emission-plan metadata,
  populate the built-in plugin/exporter registries, run
  `checkExecutionPlanCoherence`, and export through the generic exact-route API
  instead of directly invoking the target-local callback.
* Preserve direct target-local callback behavior for explicit self-check helper
  routes and for bounded selected-boundary-only inputs that do not yet contain
  emission-plan diagnostics.
* Keep RVV, scalar, and dispatch semantics out of generic code. The generic
  exact-route layer may match route ids and artifact route registry entries,
  but it must not branch on RVV/scalar family names, dtypes, vector shapes,
  intrinsic names, descriptor mirrors, or dispatch semantics.
* Do not add descriptor-driven computation, descriptor-to-C emission, Python
  compiler internals, new frontend families, new runtime claims, or automatic
  hardware probing.

## Acceptance Criteria

* [x] Direct RVV source/header/object translate route entries expose their
      corresponding target artifact route id.
* [x] Direct RVV+scalar dispatch source/header/object translate route entries
      expose their corresponding target artifact route id, while self-check
      routes stay target-local helpers.
* [x] Emission-plan-backed direct source/header/object translation goes through
      `checkExecutionPlanCoherence` and the generic exact-route exporter.
* [x] Boundary-only direct helper inputs without emission-plan metadata keep
      their previous target-local diagnostics and behavior.
* [x] Focused lit coverage proves a direct route rejects stale emission-plan
      metadata through the generic route/coherence surface before source output.
* [x] Focused C++ coverage proves target translate route registry metadata
      distinguishes artifact-backed routes from self-check helper routes.
* [x] Focused build/test targets for `tcrv-translate`, target artifact export,
      and touched RVV/RVV+scalar route tests pass.
* [x] `git diff --check` and Trellis task validation pass.

## Completion Notes

* Added exact-route target artifact export for standalone and composite route
  ids.
* Marked direct RVV and RVV+scalar source/header/object translation routes with
  their generic target artifact route ids; self-check helper routes remain
  unmarked.
* `tcrv-translate` now detects emission-plan diagnostics for marked direct
  routes, runs execution-plan coherence, and dispatches through the generic
  exact-route exporter.
* Updated focused C++ and lit coverage for the new route metadata and generic
  coherence diagnostics.

## Out of Scope

* No runtime, correctness, or performance claim.
* No `ssh rvv` evidence requirement; this is a local compiler route migration.
* No removal of selected-boundary-only target helper coverage in this round.
* No change to generated RVV/scalar/dispatch C ABI signatures, route ids, file
  names, or artifact kinds.
* No changes to Hermes prompts, supervisor settings, or workflow docs.

## Technical Notes

* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/architecture/unified-riscv-mlir.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
* Recent task context read:
  `.trellis/tasks/archive/2026-05/05-12-common-extension-lower-to-emitc-pass-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-12-rvv-common-emitc-source-boundary-production-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-12-05-12-rvv-scalar-dispatch-common-emitc-source-boundary/prd.md`.
* Primary implementation surfaces:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetTranslateRegistration.h`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  focused C++ and lit tests.
