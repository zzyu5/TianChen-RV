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
