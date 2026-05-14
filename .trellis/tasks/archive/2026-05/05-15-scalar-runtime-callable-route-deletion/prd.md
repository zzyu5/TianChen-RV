# Delete scalar runtime-callable direct C route authority

## Goal

Delete the scalar direct runtime-callable C route authority from the current
target/artifact surface. This round is part of the Wrong Logic Deletion
Campaign: scalar fallback must not be selected, validated, registered, or
tested through finite `tcrv-export-scalar-*` source/header/object route strings
or `scalar-*-runtime-callable-c-abi.v1` identities.

Future scalar executable output must come from materialized scalar extension
family IR through a real MLIR EmitC module route. This task does not rebuild
that route.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state was clean at HEAD `35864fe`.
- `.trellis/.current-task` did not exist at session start.
- Previous archived task
  `.trellis/tasks/archive/2026-05/05-15-direct-c-source-frontdoor-deletion`
  deleted the RVV smoke-probe source frontdoor and recorded scalar direct route
  strings as the next deletion owner.
- Current scalar target registration is already fail-closed in one important
  way: `registerScalarMicrokernelTargetExporters` registers no exporters, and
  scalar source/object export functions return the deleted-route error.
- Residual scalar direct route authority still exists in:
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`, where
    `ScalarBinaryMicrokernelRecord` owns scalar `routeID`, `headerRouteID`,
    `objectRouteID`, runtime ABI, runtime ABI kind/name, runtime glue role, and
    compatibility string builders.
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`, where source-candidate
    validation still maps `TargetArtifactCandidate` values back to scalar
    route IDs and runtime-callable C ABI metadata.
  - target artifact tests and fixtures that still preserve
    `tcrv-export-scalar-*`, `scalar-explicit-*`, and
    `scalar-runtime-callable-c-abi` as legal positive route/ABI identities.
- Relevant specs define descriptor/direct-C source export as deletion debt and
  the current architecture route as extension family ops -> EmitC -> C/C++
  emitter -> intrinsic/vendor/runtime C/C++ -> native compiler.

## Requirements

- Remove scalar source/header/object route ID fields and compatibility route
  string builders from the scalar family registration surface.
- Remove scalar runtime-callable C ABI string authority from the scalar family
  registration surface, including `scalar-*-runtime-callable-c-abi.v1`,
  `scalar-*-runtime-callable-c-function.v1`, and scalar runtime glue role
  strings.
- Delete or fail-close scalar candidate validation logic that treats those
  route/ABI strings as supported executable artifact authority.
- Delete obsolete scalar direct C/header/object helper declarations or leave
  them fail-closed only when they are still needed by a public negative surface.
- Rewrite tests and fixtures that protect scalar descriptor/direct-C
  source/header/object routes as legal input. Negative fixtures may mention old
  strings only to prove deleted-route/unknown-route behavior.
- Keep typed scalar extension op materialization and scalar plugin
  metadata-only planning intact when they do not publish executable C artifact
  authority.
- Keep generic target artifact infrastructure and generic artifact-kind strings
  such as `runtime-callable-c-source` only as route-agnostic infrastructure.
- If a build or focused test now fails because the new EmitC architecture is
  missing, record it as an expected deletion gap rather than restoring a
  compatibility path.

## Acceptance Criteria

- [x] No scalar direct source route can be selected as supported executable
      artifact evidence.
- [x] `tcrv-export-scalar-*` route IDs are removed from production route
      registration, family metadata, and positive tests unless the remaining
      occurrence is explicitly deleted-route/unknown-route negative evidence.
- [x] `scalar-*-runtime-callable-c-abi.v1` and
      `scalar-runtime-callable-c-abi` no longer act as scalar route-selection
      or candidate-validation authority.
- [x] Scalar direct C/header/object helper functions are deleted or fail closed
      without compatibility wrappers.
- [x] Tests no longer protect scalar descriptor/direct-C fixtures as legal
      executable artifact input.
- [x] No new scalar fallback implementation, Common Lower-To-EmitC rebuild,
      new EmitC route, replacement route registry, descriptor wrapper, direct
      header/source/object shim, compatibility mode, RVV finite-family
      extension, Python compiler semantics, or ssh RVV evidence campaign is
      added.
- [x] Bounded before/after scans classify remaining hits for scalar direct
      route terms as deleted-route negative fixtures, route-agnostic generic
      infrastructure, non-scalar runtime ABI infrastructure, or residual
      rebuild debt.
- [x] Focused affected builds/tests run; full `check-tianchenrv` runs if the
      deletion slice is coherent.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean `git status --short`, and
      one coherent commit are produced if complete.

## Non-goals

- No new scalar executable fallback implementation.
- No Common Lower-To-EmitC rebuild.
- No new EmitC route or replacement route registry.
- No descriptor wrapper, compatibility mode, direct C/header/object shim, or
  legacy alias.
- No RVV finite-family extension or new hardware path.
- No ssh RVV correctness/performance campaign.
- No Python compiler-core semantics.
- No broad repo audit beyond bounded scalar/artifact ref-scans.
- No restoring old scalar direct C paths to satisfy stale tests.

## Minimal Evidence

- Focused before/after scans over touched scalar/artifact files for:
  `tcrv-export-scalar`, `ScalarBinaryMicrokernelRecord`,
  `RVVScalarBinaryFamilyRecord`, `runtime-callable-c-source`,
  `runtime-callable-c-header`, `scalar-runtime-callable-c-abi`,
  `runtime-callable-c-abi.v1`, `routeID`, `headerRouteID`, `objectRouteID`,
  `descriptor`, `scalar-explicit`, and `target artifact route`.
- Focused build/test targets for scalar target artifact export,
  target-artifact bundle/export C++ tests, RVV scalar dispatch tests affected
  by fixture deletion, and translate registration if touched.
- Full `cmake --build build --target check-tianchenrv -j2` when coherent.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive.

## Technical Notes

- `.trellis/spec/architecture/design-boundaries.md` rejects a
  descriptor-driven microkernel/exporter framework.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the future accepted
  route as extension family ops -> EmitC -> C/C++ emitter.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires direct
  RVV/scalar/dispatch C compute-body exporters to be deleted or fail-closed
  until a materialized MLIR EmitC module route exists.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleting or
  rewriting tests whose only assertion is successful old direct source/header/
  object/bundle output.
- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` already states
  scalar direct runtime-callable C source/header/object routes are deleted
  production routes; this task aligns code/tests with that statement.

## Completion Evidence

- `ScalarBinaryMicrokernelRecord` no longer carries route IDs, direct artifact
  kinds, runtime ABI names, runtime ABI kind/name fields, or runtime glue role
  authority.
- Scalar target exporter implementation is fail-closed only; scalar source,
  header, and object export helpers return the deleted-route diagnostic, while
  scalar target exporter registration contributes no route authority.
- Runtime ABI contract no longer carries scalar standalone callable identities;
  remaining `rvv-scalar-*` strings are dispatch ABI identities, not scalar
  standalone source/header/object route authority.
- Deleted scalar direct artifact fixtures:
  `missing-scalar-boundary.txt`, `missing-scalar-microkernel.txt`,
  `stale-scalar-selected-variant.txt`,
  `artifact-export-coherence-preflight.txt`, `rvv-spoof-scalar-route.txt`,
  `offload-spoof-scalar-source-route.txt`, and
  `multiple-standalone-bundle-frontdoors.txt`.
- Remaining `tcrv-export-scalar-*` hits are negative absence/deleted-route
  checks in specs and C++ registry tests; no production registration, family
  metadata, or positive fixture keeps them legal.
- Focused tests run:
  `cmake --build build --target TianChenRVScalarTarget TianChenRVSupport
  TianChenRVRVVPlugin tianchenrv-target-artifact-export-test
  tianchenrv-runtime-abi-callable-plan-test -j2`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `cmake --build build --target tianchenrv-emitc-lowerable-interface-test -j2`,
  and `./build/bin/tianchenrv-emitc-lowerable-interface-test`.
- Full validation run:
  `cmake --build build --target check-tianchenrv -j2` passed 114/114 tests,
  `git diff --check` passed, and Trellis validation passed before finish.
