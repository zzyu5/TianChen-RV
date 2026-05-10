# Journal - codex (Part 2)

> Continuation from `journal-1.md` (archived at ~2000 lines)
> Started: 2026-05-10

---


## Session 26: Plugin-local RVV binary proposal legality planning

**Date**: 2026-05-10
**Task**: Plugin-local RVV binary proposal legality planning
**Branch**: `main`

### Summary

Created and completed the Trellis task from the Hermes brief after confirming
there was no active current task. Extracted finite RVV binary proposal planning
and required capability metadata from `RVVExtensionPlugin.cpp` into
`RVVBinaryPlanning`.

### Main Changes

- Added `RVVBinaryProposalPlan` and reusable binary capability/property view
  helpers to `RVVBinaryPlanning`.
- Moved frontend family lookup, selected vector-shape requirement id
  construction, descriptor-local element count derivation, and capacity fact
  proposal metadata decisions out of `RVVExtensionPlugin.cpp`.
- Kept route ids, artifact kinds, runtime ABI identity, and dispatch facts
  derived from target-owned finite family descriptors/manifests.
- Extended `tianchenrv-rvv-binary-planning-test` with focused structured
  proposal-plan coverage for `i32-vmul`, `i64-vmul`, capacity layering, finite
  family rejection, and the `i64-vmul` dispatch representative's selected RVV
  route/ABI facts.
- Preserved public diagnostic wording after `check-tianchenrv` exposed five
  lit expectation failures caused by changed helper error text.

### Evidence

- No RVV runtime/correctness/performance claim was made; no fresh `ssh rvv`
  run was required.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-binary-planning|rvv-extension-plugin|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  first run failed 5 lit diagnostics after helper extraction; after restoring
  public diagnostic wording, rerun passed `194/194`.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-proposal-legality-planning`

### Status

[OK] **Completed and archived**


## Session 29: RVV+scalar dispatch plugin-owned exporter bundle

**Date**: 2026-05-11
**Task**: RVV+scalar dispatch artifact routes through plugin-owned exporter registration
**Branch**: `main`

### Summary

Created the Trellis task and PRD from the Hermes brief, then moved the
RVV+scalar dispatch source/header/object target artifact route group out of
central non-plugin built-in exporter registration. The routes now publish
through a plugin-owned target exporter bundle owned by `scalar-plugin` and
requiring an enabled `rvv-plugin`, so missing or disabled relevant plugins fail
closed by not exposing the dispatch route group.

### Main Changes

- Extended `PluginTargetArtifactExporterBundle` with generic required-plugin
  dependency metadata.
- Registered RVV+scalar dispatch source/header/object composite exporters
  through the plugin-owned bundle boundary while keeping all dispatch route
  matching, runtime ABI preflight, source/header/object emission, and
  component-group semantics in RVVScalarDispatch target code.
- Removed direct RVV+scalar dispatch route publication from central
  non-plugin built-in target exporter composition.
- Added C++ coverage for enabled contribution, duplicate route rejection,
  disabled/missing scalar owner behavior, and disabled/missing RVV dependency
  behavior.
- Updated the lowering/runtime spec to record plugin-owned composite dispatch
  registration with required enabled plugin dependencies.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate -j2`
- Focused lit:
  `cd artifacts/tmp/tianchenrv-build/test && /usr/bin/python3.10
  /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv
  /home/kingdom/phdworks/TianchenRV/artifacts/tmp/tianchenrv-build/test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir`
  passed 1/1.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- No RVV runtime/correctness/performance claim was made; no `ssh rvv` evidence
  was needed or collected.

### Status

[OK] **Completed and archived; commit pending in this session**


## Session 29: Scalar finite fallback header/object artifact routes

**Date**: 2026-05-11
**Task**: Scalar finite fallback header/object artifact routes
**Branch**: `main`

### Summary

Created the Trellis task from the malformed-review fallback brief after
confirming there was no active current task. Completed the finite scalar
fallback artifact surface so existing i32/i64 add/sub/mul scalar callable
source candidates can also select matching scalar header and relocatable-object
artifact routes through the generic target artifact front doors.

### Main Changes

- Added family-derived scalar header/object route ids for non-legacy finite
  scalar families while preserving the legacy i32-vadd route ids.
- Reworked scalar target exporter registration to register source, header, and
  object routes for all finite scalar fallback families.
- Made scalar header/object composite matching family-specific so stale add
  candidates cannot satisfy vmul or i64 helper routes.
- Updated target artifact and i32 family registry C++ coverage for the expanded
  scalar route set and stale-family rejection.
- Added focused lit coverage for non-add scalar vmul header/object/bundle
  front-door artifact export.
- Updated scalar fallback and lowering/runtime specs to record the finite
  scalar header/object route boundary without making runtime claims.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-target-artifact-export-test
  tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `Target/ArtifactExport/scalar-target-header-object-artifact-routes.test`
  and `Target/ArtifactExport/scalar-target-vmul-source-artifact-routes.test`,
  2/2 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- No RVV runtime/correctness/performance claim was made; no `ssh rvv` evidence
  was needed or collected.

### Status

[OK] **Completed and archived**


## Session 29: RVV selected artifact exporters through plugin-owned registration

**Date**: 2026-05-11
**Task**: RVV selected artifact routes through plugin-owned exporter registration
**Branch**: `main`

### Summary

Created the Trellis task and PRD from the Hermes brief, then moved the
existing finite RVV selected binary microkernel source/header/object target
artifact exporter group behind the generic plugin-owned target exporter bundle
boundary. The default built-in front door still exposes the selected RVV
artifact routes because builtin extension plugins enable `rvv-plugin`, while a
registry without enabled `rvv-plugin` no longer publishes those selected RVV
microkernel routes centrally.

### Main Changes

- Added a `rvv-plugin` target-exporter bundle entry in `Target/RVV` that
  delegates to the existing RVV microkernel target exporter registration.
- Removed direct RVV microkernel route registration from the central
  non-plugin built-in target exporter list and registered it through the
  plugin-owned bundle registry next to Toy.
- Kept RVV route identity, source/header/object exporters, selected binary
  family checks, runtime ABI role requirements, and candidate preflight in
  RVV target-owned C++ code.
- Extended target artifact export C++ coverage for RVV plugin-owned bundle
  registration, duplicate bundle/route rejection, disabled/missing
  `rvv-plugin` fail-closed behavior, and default built-in route visibility.
- Updated the lowering/runtime spec to record that plugin-owned target exporter
  bundles now cover real RVV selected binary microkernel routes as well as Toy
  metadata routes.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `target-source-artifact-routes|rvv-microkernel-object`: 3/3 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-plugin-owned-target-exporter-bundle`: passed.
- No RVV runtime/correctness/performance claim was made; no fresh `ssh rvv`
  evidence was needed or collected.

### Status

[OK] **Completed and archived**


## Session 16: Target-owned artifact route translation registration

**Date**: 2026-05-10
**Task**: Target-owned artifact route translation registration
**Branch**: `main`

### Summary

Moved tcrv-translate RVV direct and RVV+scalar route-family helper command registration behind a target-owned translate route registry, added built-in route contribution tests, updated the durable route-registration spec, archived the Trellis task, and passed check-tianchenrv 193/193.

### Main Changes

- Added `TargetTranslateRoute` / `TargetTranslateRouteRegistry` plus `registerBuiltinTargetTranslateRoutes`.
- Moved RVV direct binary microkernel source/header/object translate route-family contribution into RVV target support.
- Moved RVV+scalar dispatch source/header/object/self-check route-family contribution into target support.
- Simplified `tcrv-translate` to one generic built-in target translate route registration loop while leaving RVV smoke probe and standalone RVV self-check C as direct legacy helpers.
- Added C++ registry coverage for malformed routes, duplicate route ids, representative direct RVV routes, representative RVV+scalar dispatch routes, and legacy helper exclusion.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with the built-in target translate route registration boundary.

### Testing

- [OK] `git diff --check`
- [OK] focused build: `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] focused lit filter: `rvv-microkernel-i64-vadd|rvv-microkernel-header|rvv-scalar-i32-vadd-dispatch-generic-route|rvv-scalar-i64-vsub-dispatch-generic-route` passed 6/6
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 193/193
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-10-target-owned-artifact-route-translation-registration`

### Status

[OK] **Completed and archived**


## Session 25: Plugin-local RVV binary planning extraction

**Date**: 2026-05-10
**Task**: Plugin-local RVV binary planning extraction
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes brief after confirming there was no
active `.trellis/.current-task`. Completed a plugin-local C++ extraction for
selected RVV binary emission identity planning without reopening the archived
runtime evidence task.

### Main Changes

- Added `RVVBinaryEmissionIdentity` to `RVVBinaryPlanning` so selected direct
  RVV binary readiness path, route id, emission kind, source artifact kind,
  runtime ABI identity, runtime glue role, and bounded support explanation are
  derived from the target-owned finite RVV binary family descriptor.
- Removed the `RVVI32MicrokernelFamilySpec` table from
  `RVVExtensionPlugin.cpp`; the plugin now consumes the planner identity for
  direct i32 and i64 RVV binary emission readiness and emission plans.
- Extended RVV binary planning C++ coverage across i32 and i64 families,
  including the i64-vmul dispatch representative's selected RVV component
  facts and target-owned dispatch object route/success marker.

### Evidence

- No RVV runtime/correctness/performance claim was made in this round; no fresh
  `ssh rvv` run was required.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-binary-planning|rvv-extension-plugin|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed

### Status

[OK] **Completed and archived**


## Session 19: Fixture-free RVV frontend dispatch runtime evidence

**Date**: 2026-05-10
**Task**: Fixture-free RVV dispatch runtime evidence from frontend pipeline
**Branch**: `main`

### Summary

Completed the bounded fixture-free `i64-vmul` RVV+scalar dispatch runtime
evidence path. The existing C++/MLIR pipeline already carried marked linalg
frontend input through `--tcrv-lower-linalg-rvv-binary-to-exec` and
`--tcrv-execution-planning-pipeline` to selected manifest-backed dispatch
object/self-check object routes, so this round tightened runner evidence and
validated the path live on `ssh rvv`.

### Main Changes

- Added direct evidence fields for `frontend_pipeline_command` and
  `frontend_pipeline_args`, proving the fixture-free path uses the frontend
  lowering pass before execution planning.
- Added live remote evidence summary fields preserving remote link flags and
  bounded stdout/stderr tails for compile/link/run steps.
- Extended runner self-test for link flags and stdout/stderr summary
  preservation.
- Updated focused script lit coverage to assert the fixture-free `i64-vmul`
  direct path records the frontend pipeline command.
- Preserved selected-fixture evidence and C++/MLIR/target-owned route
  semantics; no generic core pass or compiler route ownership moved into
  Python.

### Evidence

- Runtime/correctness claim is bounded to fixture-free `i64-vmul`
  RVV+scalar dispatch self-check execution on real `ssh rvv`.
- Live evidence:
  `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-fixture-free-i64-vmul-runtime-live/evidence.json`
  records `fixture_free_frontend_pipeline: true`,
  `ssh_evidence_verified: true`, route ids
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-object` and
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-object`, link flags
  including `-no-pie`, stdout/stderr summaries, and marker
  `tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok`.
- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Fixture-free dry run for `i64-vmul` with `--lower-linalg-frontend`
- Fixture-free live run for `i64-vmul` with `--lower-linalg-frontend` and
  `--ssh-target rvv`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit filter:
  `rvv-scalar-i64-vmul-dispatch-object|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e`
  passed after reordering sorted-JSON FileCheck assertions.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed

### Status

[OK] **Completed; ready to archive and commit**
## Session 21: RVV binary artifact manifests as target-owned family source of truth

**Date**: 2026-05-10
**Task**: RVV binary artifact manifests as target-owned family source of truth
**Branch**: `main`

### Summary

Continued the Round 50 route-manifest cleanup by removing the remaining direct
RVV microkernel handwritten finite route list. Direct RVV microkernel routes
and RVV+scalar dispatch routes now expose target-owned route-kind/count APIs and
derive manifest records from the finite binary family registries.

### Main Changes

- Added direct RVV microkernel route-kind/count APIs in
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h` and implemented
  registry-derived direct source/header/object route manifest construction in
  `lib/Target/RVV/RVVMicrokernel.cpp`.
- Added RVV+scalar dispatch route-kind/count APIs in
  `include/TianChenRV/Target/RVVScalarDispatch.h` and made
  `lib/Target/Builtin/RVVScalarDispatch.cpp` build dispatch route records via
  route-kind x family-registry derivation.
- Updated `test/Target/TargetArtifactExportTest.cpp` and
  `test/Target/I32BinaryFamilyRegistryTest.cpp` so expected route counts and
  all-family invariants are manifest-owned, with representative direct and
  dispatch route-name checks.

### Evidence

- No RVV runtime/correctness/performance claim was made; this was a local
  C++ target-support manifest refactor.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --show-pass . --filter 'target-artifact-export|i32-binary-family-registry|rvv-microkernel-bundle|rvv-scalar-dispatch-bundle'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: `193/193` passed

### Status

[OK] **Completed and archived**

## Session 17: Bounded linalg RVV binary frontend pass naming

**Date**: 2026-05-10
**Task**: Bounded linalg RVV binary frontend pass naming
**Branch**: `main`

### Summary

Exposed the bounded linalg frontend lowering as an RVV binary pass, kept i32 aliases, updated specs/scripts/tests, and passed focused plus full checks.

### Main Changes

- Added the public tcrv-lower-linalg-rvv-binary-to-exec pass and C++ factory while keeping the old i32 binary and i32-vadd pass options as compatibility aliases.
- Updated tcrv-translate plan-and-export frontend lowering, RVV/scalar scripts, planned_pipeline metadata, Trellis specs, and focused lit coverage to use the RVV binary public surface.
- Validated with git diff --check, focused build, focused lit, script self-tests, full check-tianchenrv, and Trellis context validation.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## Session 20: First-class RVV selected config and runtime VL boundary model

**Date**: 2026-05-10
**Task**: First-class RVV selected config and runtime VL boundary model
**Branch**: `main`

### Summary

Created the bounded Trellis task from the Hermes brief, recorded the current
RVV parameter-layering inventory, tightened the plugin-local selected config
surface for finite RVV binary selected plans, and made runtime AVL/VL boundary
metadata visible in selected emission planning for the current C-intrinsics
artifact path.

### Main Changes

- Added `RVVBinarySelectedConfig` around the selected `RVVVectorShapeConfig`
  and routed selected-plan shape consumers through that structured surface.
- Generalized selected vector-shape metadata helpers from misleading i32-only
  names to `RVVVectorShapeSelectedPlanMetadataDescriptor` and
  `appendRVVVectorShapeSelectedPlanMetadata`.
- Added selected-emission metadata entries for runtime AVL source, AVL role,
  VL source, and VL scope so the selected path exposes `runtime n -> AVL ->
  tcrv_rvv.setvl -> !tcrv_rvv.vl -> tcrv_rvv.with_vl`.
- Added C++ and lit coverage for the runtime AVL/VL boundary metadata on
  `i64-vsub` and `i32-vsub`/`i32m2` without changing generated artifact
  semantics.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVPlugin
  TianChenRVRVVTarget TianChenRVTransforms tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-dialect-test tianchenrv-emission-readiness-test
  tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-dialect-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `linalg-i64-vsub-to-rvv-artifact`,
  `linalg-i32-vsub-to-rvv-artifact`,
  `rvv-microkernel-auto-materialization`, and `rvv-microkernel-i64-vsub`:
  4/4 passed after repairing FileCheck order to match exporter output.
- `python3 scripts/rvv_microkernel_e2e.py --dry-run
  --arithmetic-family=i64-vsub --lower-linalg-frontend --input
  test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir
  --run-id codex-selected-config-vl-boundary-i64-vsub-dry --overwrite`:
  status `success`, `ssh_evidence=false`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  194/194 lit tests passed.
- No new RVV runtime/correctness/performance claim was made; no fresh
  `ssh rvv` run was required.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete

## Session 25: RVV i64-vsub selected path artifact evidence

**Date**: 2026-05-10
**Task**: RVV i64-vsub selected path from plugin planning to artifact evidence
**Branch**: `main`

### Summary

Created the Trellis task for the bounded `i64-vsub` selected-path composition
round and proved current HEAD carries the marked linalg frontend route through
RVV plugin proposal planning, variant legality, selected lowering-boundary
materialization, selected emission planning, target artifact export, dry-run
artifact evidence, and fresh real `ssh rvv` runtime correctness evidence.

### Main Changes

- Added direct `i64-vsub` coverage to the extracted
  `RVVBinaryVariantLegality` module test.
- Parameterized the extracted `RVVBinarySelectedLoweringBoundary` module test
  so it now covers both `i64-vsub` and `i64-vmul`.
- Strengthened the `linalg-i64-vsub-to-rvv-artifact.mlir` frontend lit check
  to assert descriptor-owned runtime ABI parameters and selected-plan metadata
  on the supported emission-plan diagnostic.

### Evidence

- Dry-run evidence:
  `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vsub-selected-path-dry/evidence.json`
  records `status = "success"`, `mode = "dry-run"`,
  `manifest_record.kernel = "frontend_i64_vsub"`, and route
  `tcrv-export-rvv-i64-vsub-microkernel-c`.
- Live evidence:
  `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vsub-selected-path-live/evidence.json`
  records `status = "success"`, `mode = "ssh"`,
  `ssh_evidence.success = true`, remote compile/link/run success, and marker
  `tcrv_rvv_i64_vsub_microkernel_external_abi_ok`.
- Claim scope is bounded finite-family generated RVV `i64-vsub`
  C-intrinsics direct helper artifact correctness on real `ssh rvv`; this is
  not a generic RVV backend, not an MLIR vector lowering route, and not a
  performance claim.

### Testing

- `git diff --check`
- Focused build for `tcrv-opt`, `tcrv-translate`, RVV planning, RVV variant
  legality, RVV selected lowering-boundary, RVV extension plugin, target
  artifact export, and family registry tests.
- Focused C++ tests:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-binary-variant-legality-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-i32-binary-family-registry-test`.
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `linalg-i64-vsub-to-rvv-artifact|rvv-microkernel-e2e` passed 2/2 selected
  tests.
- Dry-run and live `scripts/rvv_microkernel_e2e.py` commands for
  `--arithmetic-family=i64-vsub --lower-linalg-frontend`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  194/194 lit tests passed.

### Status

[OK] **Completed; ready to archive and commit**


## Session 20: Plugin-local RVV selected lowering-boundary materialization

**Date**: 2026-05-10
**Task**: Plugin-local RVV selected lowering-boundary materialization
**Branch**: `main`

### Summary

Extracted finite RVV binary selected lowering-boundary materialization and
validation into a plugin-local C++ component. `RVVExtensionPlugin` now
delegates selected boundary materialization/validation to that module while
keeping generic core passes free of RVV semantics and preserving smoke-probe
and unsupported diagnostic behavior.

### Main Changes

- Added `RVVBinarySelectedLoweringBoundary` as the plugin-local owner for
  selected `tcrv_rvv.lowering_boundary` materialization and validation.
- Moved selected boundary capability-summary construction, boundary op
  construction, selected vector metadata copying, capacity metadata
  copying/validation, duplicate-boundary rejection, runtime ABI
  mem_window/runtime_param ensuring, and i32/i64 callable microkernel
  orchestration out of `RVVExtensionPlugin.cpp`.
- Added focused module C++ coverage for i32 selected materialization,
  `i64-vmul` selected materialization, ABI boundary ensuring, duplicate
  boundary rejection, and ratio-valid boundary/variant capacity mismatch
  validation.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-extension-plugin`, `rvv-scalar-dispatch-e2e`, and
  `rvv-scalar-dispatch-bundle-e2e`: 3/3 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  194/194 lit tests passed.
- No generated runtime artifact semantics changed and no new RVV
  runtime/correctness/performance claim was made; no fresh `ssh rvv` run was
  required.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 24: Plugin-local RVV binary variant legality validation

**Date**: 2026-05-10
**Task**: Plugin-local RVV binary variant legality validation
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes malformed-review fallback after
confirming there was no active `.trellis/.current-task`. Extracted finite RVV
binary variant legality validation from `RVVExtensionPlugin.cpp` into the
plugin-local `RVVBinaryVariantLegality` C++ module. The public plugin now
delegates legality to that module, and smoke-probe readiness/plan construction
reuse the same bounded smoke metadata helper instead of keeping a private
legality-helper island in the extension plugin file.

### Main Changes

- Added `RVVBinaryVariantLegality` as the plugin-local owner for finite RVV
  binary variant legality validation.
- Moved origin, available `rvv` capability, `requires`, finite vector-shape,
  descriptor dtype/shape, typed policy, selected vector-shape metadata,
  required march, smoke-probe descriptor, capacity metadata, and selected
  i32/i64 microkernel selected-plan legality checks out of
  `RVVExtensionPlugin.cpp`.
- Updated `RVVExtensionPlugin::verifyVariantLegality` to delegate to the new
  module.
- Updated smoke-probe readiness and emission-plan paths to use
  `verifyRVVBinarySmokeProbeVariantMetadata`.
- Added focused `tianchenrv-rvv-binary-variant-legality-test` direct module
  coverage for i32, `i64-vmul`, smoke-probe, origin mismatch, and selected
  vector-shape metadata mismatch paths.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-extension-plugin`, `rvv-scalar-dispatch-e2e`, and
  `rvv-scalar-dispatch-bundle-e2e`: 3/3 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  194/194 lit tests passed.
- No generated runtime artifact semantics changed and no new RVV
  runtime/correctness/performance claim was made; no fresh `ssh rvv` run was
  required.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 18: Front-door RVV binary frontend bundle with ssh rvv execution evidence

**Date**: 2026-05-10
**Task**: Front-door RVV binary frontend bundle with ssh rvv execution evidence
**Branch**: `main`

### Summary

Completed a direct i64-vmul marked-Linalg front-door evidence path through
`tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`, target-owned
source/header/object bundle emission, generated external caller construction,
and real `ssh rvv` compile/link/run validation.

### Main Changes

- Updated `scripts/rvv_microkernel_e2e.py` so direct RVV bundle records accept
  the compiler-emitted selected variant symbol while still requiring all
  source/header/object records to agree.
- Added structured live `ssh_evidence` with explicit remote compile, link, run,
  and output-marker validation status.
- Added focused bundle lit coverage for i64-vmul marked-Linalg input through
  the plan-and-export bundle front door.

### Evidence

- Live command:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir --expect-selected-kernel frontend_i64_vmul --ssh-target rvv --run-id codex-frontdoor-i64-vmul-live --overwrite`
- Live artifact:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-frontdoor-i64-vmul-live/evidence.json`
- Live result:
  `ssh_evidence.success = true`, remote compile/link/run/output validation all
  true; claim scope is bounded i64-vmul target-artifact bundle external caller
  correctness only.

### Testing

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] focused lit:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv /home/kingdom/phdworks/TianchenRV/artifacts/tmp/tianchenrv-build/test --filter rvv-microkernel-bundle-e2e`
- [OK] live `ssh rvv` evidence command above
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-front-door-rvv-binary-frontend-bundle-ssh-evidence`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 193/193

### Status

[OK] **Completed; ready to archive and commit**


## Session 19: Front-door RVV+scalar dispatch bundle ssh evidence

**Date**: 2026-05-10
**Task**: Front-door RVV+scalar dispatch bundle ssh evidence
**Branch**: `main`

### Summary

Completed a bounded `i64-vmul` RVV+scalar dispatch bundle evidence path from
`tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` through
source/header/object bundle records, generated external caller construction,
and real `ssh rvv` compile/link/run validation for both dispatch branches.

### Main Changes

- Updated `scripts/rvv_scalar_dispatch_e2e.py` to persist first-class dispatch
  evidence fields for runtime guard linkage, scalar fallback linkage, runtime
  params, branch coverage, output validation contract, and structured ssh
  compile/link/run/output-validation summaries.
- Added self-test coverage for dispatch linkage parsing, branch coverage
  evidence, and bundle ssh summary aggregation.
- Updated focused bundle lit coverage to assert the live-capable `i64-vmul`
  front-door evidence schema without contacting `ssh rvv`.

### Evidence

- Dry-run command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --run-id codex-frontdoor-dispatch-i64-vmul-dry --overwrite`
- Dry-run artifact:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-frontdoor-dispatch-i64-vmul-dry/evidence.json`
- Live command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --ssh-target rvv --run-id codex-frontdoor-dispatch-i64-vmul-live --overwrite`
- Live artifact:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-frontdoor-dispatch-i64-vmul-live/evidence.json`
- Live result:
  `ssh_evidence.success = true`, remote compile/link/run/output validation all
  true; recorded host facts include `architecture = riscv64`,
  `clang_path = /usr/bin/clang`, and
  `clang_version = Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Scope:
  bounded RVV+scalar `i64-vmul` target-artifact bundle external caller
  correctness only; no performance, generic lowering, broad correctness, or
  generic RVV backend claim.

### Testing

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv . --filter rvv-scalar-dispatch-bundle-e2e`
- [OK] live `ssh rvv` evidence command above
- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 193/193
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-10-front-door-rvv-scalar-dispatch-bundle-ssh-evidence`

### Status

[OK] **Completed and archived**

## Session 20: Manifest-derived RVV+scalar dispatch route registration

**Date**: 2026-05-10
**Task**: Manifest-derived RVV+scalar dispatch route registration
**Branch**: `main`

### Summary

Repaired the malformed Hermes review fallback into a concrete target-route
registration task. The RVV+scalar dispatch route manifest now derives its
source/header/object/self-check source/self-check object records by iterating
the finite `getRVVScalarBinaryFamilyDescriptors()` registry, so artifact and
translate route registration no longer carries a second six-family append list.

### Main Changes

- Updated `lib/Target/Builtin/RVVScalarDispatch.cpp` so
  `getRVVScalarDispatchRouteManifest()` consumes the finite bridge family
  registry directly.
- Updated `test/Target/TargetArtifactExportTest.cpp` so the built-in target
  translate route count is derived from the RVV microkernel route manifest plus
  the RVV+scalar dispatch route manifest, not a hard-coded total.

### Evidence

- No RVV runtime/correctness claim was made in this round; the behavior change
  is target-owned C++ route registration.
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'target-artifact-export|i32-binary-family-registry|rvv-scalar-dispatch'`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: `193/193` passed

### Status

[OK] **Completed and archived**


## Session 22: Manifest-backed RVV selected-boundary artifact handoff

**Date**: 2026-05-10
**Task**: Manifest-backed RVV selected-boundary artifact handoff
**Branch**: `main`

### Summary

Created the new Trellis task from the Hermes brief after confirming there was
no active `.trellis/.current-task`. Completed the bounded `i64-vmul`
RVV+scalar dispatch handoff from selected emission-plan route metadata into
manifest-backed target artifact export validation.

### Main Changes

- Added public C++ manifest lookup APIs for direct RVV microkernel routes and
  RVV+scalar dispatch routes.
- Changed RVV callable source preflight so selected emission-plan route ids are
  accepted only after resolving a source route in the direct RVV manifest.
- Changed RVV+scalar dispatch export/preflight so source/header/object and
  self-check routes are resolved through the dispatch manifest before export.
- Tightened `i64-vmul` dispatch lit coverage to prove the selected emission
  plans carry the RVV/scalar callable routes and the generated source/header
  artifacts expose the dispatch manifest route consumed by the exporter.
- Added C++ coverage for representative direct RVV and RVV+scalar dispatch
  manifest lookup, including unknown-route rejection.

### Evidence

- No RVV runtime/correctness/performance claim was made in this round; no fresh
  `ssh rvv` run was required.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-i64-vmul-dispatch-generic-route|rvv-scalar-dispatch-bundle-e2e|rvv-microkernel-bundle-e2e'`
- Broader artifact/planning lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TargetArtifact|target-artifact|artifact-export|execution-planning|ExecutionPlanning|rvv-binary-planning'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `193/193` passed

### Status

[OK] **Completed and archived**


## Session 23: Manifest-backed RVV selected dispatch object evidence

**Date**: 2026-05-10
**Task**: Manifest-backed RVV selected dispatch object evidence
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes brief after confirming there was no
active `.trellis/.current-task`. Completed the bounded selected `i64-vmul`
RVV+scalar dispatch object/self-check object evidence handoff from selected
emission-plan route metadata through target-owned manifest routes to real
`ssh rvv` runtime evidence.

### Main Changes

- Added selected `i64-vmul` lit coverage for dispatch object and self-check
  object export through the emission-plan fixture, including ELF relocatable
  checks and symbol separation.
- Extended target artifact export C++ coverage for `i64-vmul`
  self-check object manifest lookup metadata.
- Updated the generated self-check harness to declare `puts` directly instead
  of including host sysroot headers, allowing local RISC-V self-check object
  export without a RISC-V sysroot.
- Extended `scripts/rvv_scalar_dispatch_e2e.py` to export dispatch object and
  self-check object artifacts through target-owned routes, record
  manifest route/kind/path/hash evidence, and link/run the exported
  self-check object on the real RVV host.
- Updated script lit expectations for the new manifest-backed object and
  self-check object dry-run evidence.

### Evidence

- Runtime/correctness claim is bounded to selected `i64-vmul`
  RVV+scalar dispatch self-check source-built and exported self-check object
  execution on real `ssh rvv`.
- Live evidence:
  `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-selected-i64-vmul-object-live/evidence.json`
  records `ssh_evidence_verified: true` and marker
  `tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok`.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i64-vmul --run-id codex-selected-i64-vmul-object-dry --overwrite`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vmul --ssh-target rvv --run-id codex-selected-i64-vmul-object-live --overwrite`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering selected
  `i64-vmul` object/self-check object, route, bundle, target artifact export,
  and script e2e filters.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed

### Status

[OK] **Completed and archived**


## Session 18: Fixture-free RVV frontend dispatch route evidence

**Date**: 2026-05-10
**Task**: Fixture-free RVV frontend dispatch route evidence
**Branch**: `main`

### Summary

Proved fixture-free i64-vmul frontend pipeline route metadata reaches manifest-backed RVV scalar dispatch object evidence without a new runtime claim.

Created and completed the Trellis task for the fixture-free `i64-vmul` frontend pipeline dispatch route evidence module. Inventory showed the existing C++/MLIR frontend lowering and execution planning pipeline already produce selected RVV+scalar route metadata, so the round added focused evidence and lit coverage instead of duplicating pipeline code.

### Main Changes

- Extended `scripts/rvv_scalar_dispatch_e2e.py` evidence with `fixture_free_frontend_pipeline`, route provenance, selected route id, manifest route id, artifact kind, path, and hash for direct dispatch/self-check object export and bundle source/header/object export.
- Added fixture-free frontend direct object/self-check object lit coverage for marked `i64-vmul` linalg input lowered through `--tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline`.
- Added script lit expectations proving the fixture-free direct path and bundle front-door path record manifest-backed artifact route identity.
- Preserved RVV semantics in plugin/target-local C++/MLIR surfaces; no generic core pass gained RVV-specific branches.

### Evidence

- No new RVV runtime/correctness claim was made; no fresh `ssh rvv` run was performed.
- First focused lit run exposed a stale `pass_fail_result` expectation in the new direct dry-run evidence; the expectation was removed because dry-run evidence has no runtime pass/fail result, then the focused lit was rerun successfully.
- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Fixture-free direct dry run for `i64-vmul` with `--lower-linalg-frontend`
- Fixture-free bundle dry run for `i64-vmul` with `--use-plan-and-export-bundle-front-door`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit filters for RVV scalar dispatch object, script e2e, bundle e2e, target artifact bundle, and plan/export bundle paths all passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: `194/194` passed

### Status

[OK] **Completed and archived**


## Session 19: Plugin-local RVV binary selected emission planning

**Date**: 2026-05-10
**Task**: Plugin-local RVV binary selected emission planning
**Branch**: `main`

### Summary

Extracted finite RVV binary selected-emission planning into a plugin-local C++ component, delegated RVV extension readiness/plan construction to it, preserved target-owned route/runtime ABI metadata, and completed focused C++ plus lit validation without a new RVV runtime claim.

### Main Changes

- Added `RVVBinarySelectedEmissionPlanning` as a plugin-local C++ selected
  emission planner for finite RVV binary selected paths.
- Moved i32/i64 explicit callable microkernel matching, descriptor-backed
  selected plan construction, runtime ABI parameter collection, required
  capability symbol extraction, selected vector metadata, and capacity
  metadata propagation out of `RVVExtensionPlugin.cpp`.
- Updated `RVVExtensionPlugin::checkVariantEmissionReadiness` and
  `RVVExtensionPlugin::buildVariantEmissionPlan` to delegate finite binary
  selected direct/callable paths to the new planner while keeping smoke-probe
  and unsupported diagnostics separate.
- Added focused plugin C++ coverage for the new planner API on an i32 selected
  path and the i64 finite family path, including `i64-vmul`.

### Git Commits

this commit

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test
  tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-binary-planning`, `rvv-extension-plugin`, `rvv-scalar-dispatch-e2e`,
  and `rvv-scalar-dispatch-bundle-e2e`: 4/4 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  194/194 lit tests passed.
- No new RVV runtime/correctness/performance claim was made; no fresh
  `ssh rvv` run was required.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 26: Toy extension plugin template through registry pipeline

**Date**: 2026-05-11
**Task**: Toy extension plugin template through the real registry pipeline
**Branch**: `main`

### Summary

Created the Trellis task and PRD from the Hermes brief, then implemented a
minimal Toy extension template that enters TianChen-RV through the existing
C++/MLIR plugin registry, variant materialization, generic selection,
selected lowering-boundary materialization, and emission-plan metadata path.

### Main Changes

- Added the plugin-local `tcrv_toy` dialect with a metadata-only
  `tcrv_toy.lowering_boundary` op and verifier.
- Added `ToyExtensionPlugin` with a bounded `toy.template` capability,
  capability-gated proposal, plugin legality, cost, lowering-boundary,
  emission readiness, and metadata-only emission plan hooks.
- Registered Toy through builtin extension plugin aggregation and CMake without
  adding Toy-specific semantic branches to generic core passes.
- Added focused Toy C++ coverage and lit coverage for positive pipeline
  participation plus malformed-capability fail-closed behavior.
- Updated the existing RVV builtin-registry smoke test to account for Toy as
  the fourth builtin plugin.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tianchenrv-toy-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter `toy`:
  3/3 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-extension-plugin-test tianchenrv-toy-extension-plugin-test
  -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused lit with filter `rvv-extension-plugin|toy`: 4/4 passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-toy-extension-plugin-template`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  197/197 lit tests passed.
- No RVV runtime/correctness/performance claim was made; no `ssh rvv` evidence
  was needed or collected.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 27: Toy metadata artifact route through target exporter

**Date**: 2026-05-11
**Task**: Toy plugin metadata artifact route through the target exporter front door
**Branch**: `main`

### Summary

Created the Trellis task and PRD from the Hermes brief, then extended the Toy
plugin template to the existing C++ target artifact exporter front door. The
Toy route now emits deterministic non-executable metadata evidence through
`tcrv-translate --tcrv-export-target-artifact` from real selected
`TargetArtifactCandidate` records.

### Main Changes

- Added a Toy-owned target exporter under `Target/Toy` with route-local
  candidate validation and bounded text artifact formatting.
- Registered the Toy metadata route through builtin target artifact exporter
  aggregation; the central change is registration only.
- Adjusted the Toy emission plan to be a supported metadata artifact route
  while keeping readiness and the `tcrv_toy.lowering_boundary` explicitly
  metadata-only.
- Added positive lit coverage for `tcrv-opt --tcrv-execution-planning-pipeline`
  piped into `tcrv-translate --tcrv-export-target-artifact`.
- Added fail-closed lit coverage for mismatched Toy runtime ABI metadata.
- Updated target artifact registry C++ coverage for the new Toy route and
  candidate preflight callback.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-toy-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `toy|ToyMetadataArtifact`: 5/5 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- No RVV runtime/correctness/performance claim was made; no `ssh rvv` evidence
  was needed or collected.

### Status

[OK] **Completed and archived**


## Session 28: Plugin-owned target artifact exporter registration

**Date**: 2026-05-11
**Task**: Plugin-owned target artifact exporter registration through the extension pipeline
**Branch**: `main`

### Summary

Created the Trellis task and PRD from the Hermes brief, then added a narrow
target-layer plugin-owned exporter bundle registry. Built-in target artifact
registration now takes the active `ExtensionPluginRegistry` and uses that same
enabled-plugin view to populate plugin-owned target artifact exporters. Toy
metadata artifact registration moved behind this generic plugin-owned boundary
while the Toy exporter body and candidate validation remain in `Target/Toy`.

### Main Changes

- Added `PluginTargetArtifactExporterBundle` and
  `PluginTargetArtifactExporterRegistry` to the C++ target artifact export
  surface.
- Added enabled-plugin population and explicit single-plugin registration
  helpers with fail-closed diagnostics for unknown, disabled, duplicate-bundle,
  and duplicate-route cases.
- Reworked built-in target artifact exporter registration so RVV/scalar/offload
  routes stay unchanged and Toy is contributed through the plugin-owned bundle
  path.
- Updated `tcrv-opt` and `tcrv-translate` to pass the active extension plugin
  registry into target exporter registration.
- Added C++ coverage proving Toy plugin-owned exporter registration,
  duplicate-route rejection, and disabled/missing plugin fail-closed behavior.
- Recorded the plugin-owned exporter bundle boundary in the lowering/runtime
  spec.

### Testing

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-toy-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `toy|ToyMetadataArtifact`: 5/5 passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-plugin-owned-target-exporter-registration`: passed.
- No RVV runtime/correctness/performance claim was made; no `ssh rvv` evidence
  was needed or collected.

### Status

[OK] **Completed and archived**
