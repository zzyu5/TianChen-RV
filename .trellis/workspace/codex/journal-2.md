# Journal - codex (Part 2)

> Continuation from `journal-1.md` (archived at ~2000 lines)
> Started: 2026-05-10

---



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
