# Manifest-backed RVV selected dispatch object evidence

## Goal

Complete one bounded compiler/export/runtime evidence module for the selected
`i64-vmul` RVV+scalar dispatch path. The selected emission-plan route must be
consumed by target-owned manifest lookup for runtime-callable object and
self-check object export, and the generated artifacts must be validated with
real `ssh rvv` compile/link/run evidence.

This task continues the archived selected-boundary artifact handoff task at
`.trellis/tasks/archive/2026-05/05-10-rvv-selected-boundary-artifact-handoff`.
That task proved source/header/export preflight route handoff and made no
runtime claim. This task must not reopen it.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round is clean at
  `8f16acf feat(rvv): validate selected artifact routes via manifests`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes brief.
* The previous task is finished and archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-boundary-artifact-handoff`.
* The previous task added target-owned manifest lookup APIs for direct RVV
  microkernel routes and RVV+scalar dispatch routes.
* Current RVV+scalar dispatch route kinds include source, header, object,
  self-check source, and self-check object.
* The finite family scope remains exactly `i32-vadd`, `i32-vsub`,
  `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.

## Requirements

* Inventory the current `i64-vmul` selected emission-plan path and the
  RVV+scalar dispatch object/self-check object export path.
* Preserve selected route ownership in C++ target/export manifests. Python may
  orchestrate tool invocation, artifact copying, remote clang/link/run, and
  evidence parsing only.
* Ensure object and self-check object export/preflight validate selected route
  identity through `lookupRVVScalarDispatchRoute(...)` and the
  target-owned dispatch manifest.
* If exporter validation already has the correct manifest-backed behavior, add
  focused coverage through the selected `i64-vmul` emission-plan fixture rather
  than only direct unit-style lookup.
* Generate bounded evidence containing selected emission-plan route id,
  manifest route id, artifact kind, generated source/header/object or
  self-check object artifact paths, remote compile/link command status, remote
  execution status, and the expected self-check success marker.
* Use real `ssh rvv` evidence for any runtime/correctness claim.
* Preserve direct RVV microkernel and RVV+scalar dispatch route families as
  distinct target-owned surfaces.
* Preserve the finite family scope exactly.

## Acceptance Criteria

* [x] The current selected `i64-vmul` dispatch path is documented in this task
  and reflected in focused code/test choices.
* [x] Direct or generic object export for the selected `i64-vmul` fixture is
  covered through the selected emission-plan path.
* [x] Self-check object export for the selected `i64-vmul` fixture is covered
  through the selected emission-plan path.
* [x] C++ target/export route ownership remains manifest-backed and
  target-local; no Python or generic core pass becomes a route/ABI/family
  source of truth.
* [x] Runner evidence records the selected emission-plan route, manifest route,
  artifact kind, generated artifact paths, remote compile/link/run status, and
  success marker for the bounded `i64-vmul` path.
* [x] Any runtime/correctness claim is backed by a passing real `ssh rvv`
  evidence run.
* [x] No new RVV family, dtype, vector shape, mask/tail policy, backend claim,
  benchmark, or performance claim is introduced.
* [x] Focused validation passes, or the task remains open with the exact
  blocker and smallest next continuation point recorded.

## Definition Of Done

* `git diff --check` passes.
* Affected tools/tests build, at minimum `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`, and focused RVV planning/registry
  test targets touched by the change.
* Focused C++ test binaries covering target artifact export and RVV manifest /
  planning behavior pass.
* Focused lit from `artifacts/tmp/tianchenrv-build/test` covers the selected
  `i64-vmul` dispatch fixture, RVV+scalar bundle evidence, target artifact
  export, and any new selected object/self-check evidence fixture.
* The bounded `ssh rvv` evidence path passes before any runtime/correctness
  claim is made.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run if public tool, pass pipeline, target exporter, route registration,
  script, or lit surfaces change.
* Active task validates before finish; archived task validates after finish if
  completed.
* One coherent commit is created only when acceptance criteria are truthfully
  satisfied.

## Out Of Scope

* New RVV family, dtype, vector shape, mask/tail policy, generic RVV backend
  claim, runtime auto-probing, benchmark, or performance claim.
* Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission, route selection, runtime ABI, family
  facts, or dispatch semantics.
* Compute semantics in `tcrv.exec`.
* RVV-specific branches in generic core passes.
* Broad smoke matrices, dashboards, report-only changes, or helper-only
  changes as the main result.
* Weakening existing selected-boundary, manifest lookup, bundle, FileCheck, or
  secret-scrubbing coverage.
* Storing credentials, tokens, passwords, or connection strings in source,
  tasks, artifacts, or reports.

## Technical Approach

The initial code inventory shows that
`lib/Target/Builtin/RVVScalarDispatch.cpp` already derives route manifest
entries from the finite family registry and validates direct object and
self-check object helpers with `lookupRVVScalarDispatchRoute(family, kind)`
before export. The implementation should therefore prefer focused proof and
evidence wiring over moving ownership.

The intended bounded change is:

1. Add positive lit coverage for the selected `i64-vmul` fixture proving direct
   object and self-check object export produce RISC-V ELF artifacts from the
   selected emission-plan route, including symbol checks and no-main vs main
   separation.
2. Tighten C++ route lookup coverage to include `i64-vmul` self-check object
   manifest lookup where useful.
3. Extend `scripts/rvv_scalar_dispatch_e2e.py` so the non-bundle evidence path
   also exports target-owned dispatch object and self-check object artifacts,
   records their route ids, artifact kinds, paths, and hashes, copies the
   self-check object to `ssh rvv`, links it, runs it, and checks the
   manifest-owned success marker. The script must remain an orchestrator only.

## Completed Evidence

The selected `i64-vmul` inventory for this round is:

* selected RVV callable route:
  `tcrv-export-rvv-i64-vmul-microkernel-c`
* selected scalar callable route:
  `tcrv-export-scalar-i64-vmul-microkernel-c`
* dispatch object manifest route:
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-object`
* self-check source manifest route:
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-c`
* self-check object manifest route:
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-object`

Code/test behavior completed:

* `lib/Target/Builtin/RVVScalarDispatch.cpp` self-check harness now declares
  `puts` directly instead of including host sysroot headers. This preserves the
  generated runtime marker behavior while allowing local RISC-V relocatable
  self-check object export without a RISC-V sysroot.
* `test/Target/RVVScalarDispatch/rvv-scalar-i64-vmul-dispatch-object.test`
  covers the selected `i64-vmul` fixture through direct and generic object
  export plus direct self-check object export. It verifies relocatable ELF
  artifacts, dispatch symbols, `main` separation, and the self-check object's
  unresolved `puts` runtime dependency.
* `test/Target/TargetArtifactExportTest.cpp` now checks the `i64-vmul`
  self-check object route manifest lookup, artifact kind, binary stdout
  behavior, and success marker metadata.
* `scripts/rvv_scalar_dispatch_e2e.py` exports dispatch object and self-check
  object artifacts through target-owned routes, validates the ELF objects,
  records manifest route/kind/path/hash evidence, copies the self-check object
  to `ssh rvv`, links it with explicit `-no-pie`, runs it, and records the
  observed success marker. Python remains orchestration/evidence tooling only.

The passing live evidence artifact is:

* `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-selected-i64-vmul-object-live/evidence.json`

That evidence records `ssh_evidence_verified: true`,
`tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok`, source-built remote
self-check success, and exported self-check object remote success. The runtime
claim is bounded to the selected `i64-vmul` RVV+scalar dispatch path.

## Validation Plan

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
* `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
* Focused lit from `artifacts/tmp/tianchenrv-build/test` for
  `rvv-scalar-i64-vmul-dispatch-generic-route`,
  `rvv-scalar-dispatch-bundle-e2e`, target artifact export, and the new
  selected object/self-check fixture.
* Real `ssh rvv` bounded evidence command for selected `i64-vmul`.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if script/lit/public tool surfaces change.

## Validation Evidence

Passed in this round:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
* `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i64-vmul --run-id codex-selected-i64-vmul-object-dry --overwrite`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vmul --ssh-target rvv --run-id codex-selected-i64-vmul-object-live --overwrite`
* Focused lit from `artifacts/tmp/tianchenrv-build/test` with filters covering
  selected `i64-vmul` object/self-check object, selected route, bundle e2e,
  target artifact export, and script e2e.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-selected-dispatch-object-evidence`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-10-rvv-selected-dispatch-object-evidence`

Self-repair performed:

* The first live `ssh rvv` run failed when linking the locally exported
  self-check object because the remote linker defaulted to PIE and rejected
  non-PIC RISC-V relocations. Manual remote validation with `-no-pie` passed,
  so the runner now uses explicit `-no-pie` for remote executable links and
  records link flags in evidence.
* The first full `check-tianchenrv` run failed only on stale FileCheck text in
  `test/Scripts/rvv-scalar-dispatch-e2e.test`; the lit expectations were
  updated to match the new manifest-backed object/self-check object evidence,
  and the full check then passed.

## Technical Notes

Relevant specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-10-rvv-selected-boundary-artifact-handoff/prd.md`

Workspace journal context read:

* `.trellis/workspace/codex/journal-2.md`
