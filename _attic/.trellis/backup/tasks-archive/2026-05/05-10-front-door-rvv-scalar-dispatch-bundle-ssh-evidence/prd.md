# Front-door RVV+scalar dispatch bundle ssh evidence

## Goal

Complete one bounded RVV+scalar dispatch artifact path end to end from the
public `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` front
door to generated source/header/object bundle records, an external C caller,
and live `ssh rvv` compile/link/run evidence. The representative path for this
round is `i64-vmul` unless repository evidence proves another already-planned
dispatch fixture is the smallest truthful representative.

## Owner

The module owner is the public plan-and-export target-artifact bundle path for
RVV+scalar host dispatch. That includes IR-backed runtime element count,
IR-linked dispatch availability guard, scalar fallback target linkage, bundle
source/header/object records, external caller construction, and sanitized live
RVV evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial state for this round is clean on `main` at
  `d19cf7c feat(rvv): add front-door bundle ssh evidence`.
- `.trellis/.current-task` was absent at session start. This task was created:
  `.trellis/tasks/05-10-front-door-rvv-scalar-dispatch-bundle-ssh-evidence`.
- The previous archived task at
  `.trellis/tasks/archive/2026-05/05-10-front-door-rvv-binary-frontend-bundle-ssh-evidence`
  proved a direct RVV `i64-vmul` bundle external caller path. It is context
  only; this task must prove RVV+scalar dispatch.
- `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
  is already a planned front-door RVV+scalar dispatch fixture with IR-backed
  mem windows, runtime `n`, typed `runtime_guard_required = true`, a
  `runtime_guard` link to `@abi_dispatch_availability_guard`, and a scalar
  fallback target.
- `scripts/rvv_scalar_dispatch_e2e.py` already has target-artifact bundle mode,
  plan-and-export front-door mode, i32/i64 add/sub/mul family selection,
  bundle index parsing, generated external caller construction, and optional
  `ssh rvv` execution.
- Existing dry-run lit coverage exercises bundle export and front-door mode
  without contacting `ssh rvv`. It must remain local evidence only.

## Requirements

1. Use the public
   `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` front door,
   not a deprecated pass alias and not hand-authored selected-path or
   emission-plan metadata, for the representative live evidence command.
2. Prefer the bounded `i64-vmul` fixture
   `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
   because it is a true dispatch bundle fixture: selected RVV case, scalar
   fallback, runtime guard, source/header/object bundle records, and i64
   arithmetic are all planned from the frontend input.
3. The generated bundle must contain one source record, one header record, and
   one RISC-V relocatable object record for the same RVV+scalar dispatch
   external ABI. The runner must reject missing or inconsistent route,
   component group, component role, external ABI name, runtime ABI metadata,
   or ordered runtime ABI parameter signature.
4. The external caller must include the generated header, call the generated
   dispatcher, exercise `rvv_available = 0` and `rvv_available = 1`, use at
   least runtime counts `7` and `16`, check arithmetic output, and check that
   elements beyond `n` are not overwritten.
5. Live evidence must copy only generated source/header/object artifacts and
   the generated caller to `ssh rvv`, compile the generated caller, compile
   the generated dispatch source into a source-built object, link/run the
   caller against both the source-built object and generated bundle object,
   and validate the family-specific success marker.
6. Evidence JSON must remain sanitized and structured. It must record remote
   architecture, clang path/version, compile/link/run status, output
   validation status, branch coverage, selected family, component group,
   source/header/object routes, runtime ABI parameters, runtime guard link,
   fallback target/linkage, and artifact paths relative to the repo where
   possible.
7. Python remains runner/evidence tooling only. Compiler internals, dialects,
   passes, plugin registry, capability model, lowering, and emission must stay
   in C++/MLIR/LLVM/TableGen/CMake unless an exporter/ABI bug is found.
8. The final claim scope must be narrow: bounded RVV+scalar `i64-vmul`
   target-artifact bundle external ABI executability/correctness on the real
   `ssh rvv` target. No performance, broad correctness, generic RVV backend,
   or generic Linalg lowering claims.

## Acceptance Criteria

- [x] Representative dry-run command succeeds:
      `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --run-id codex-frontdoor-dispatch-i64-vmul-dry --overwrite`.
- [x] Representative live command succeeds on `ssh rvv`:
      `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --ssh-target rvv --run-id codex-frontdoor-dispatch-i64-vmul-live --overwrite`.
- [x] The live `evidence.json` records successful remote compile/link/run,
      output validation for source-built and bundle-object executable paths,
      branch coverage for guard false and true, runtime counts 7 and 16, and
      no performance claim.
- [x] The evidence schema or checks are hardened so dispatch runtime-guard and
      scalar fallback linkage are first-class evidence fields, not only
      implicit source comments or file-name inference.
- [x] Focused local lit coverage validates the live-capable front-door bundle
      evidence schema without contacting `ssh rvv`.
- [x] `git diff --check` passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
      passes.
- [x] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` passes if the
      script changes.
- [x] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passes if the
      script changes or schema helpers are added.
- [x] Focused lit with `--filter rvv-scalar-dispatch-bundle-e2e` passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes if C++/MLIR/exporter code changes, or if the focused script
      behavior affects shared runner coverage.
- [x] Trellis task validation passes before archive.

## Out Of Scope

- Generic RVV backend implementation or broad RVV lowering.
- New arithmetic families, broad shape/arithmetic matrices, or performance
  benchmarking.
- Compute semantics in `tcrv.exec`.
- RVV/scalar dispatch semantics hard-coded into generic core passes.
- Automatic hardware probing inside generated dispatch C; the dispatch guard
  remains an explicit host-provided runtime ABI/control parameter.
- Treating task metadata, prompt edits, helper-only reports, broad smoke tests,
  or stale cached evidence as the main deliverable.
- Committing generated `artifacts/tmp` evidence artifacts.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Focused lit for `rvv-scalar-dispatch-bundle-e2e`
- Representative live `ssh rvv` command named above
- Inspect/assert
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-frontdoor-dispatch-i64-vmul-live/evidence.json`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if C++/MLIR/exporter code changes or focused script changes warrant it
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-10-front-door-rvv-scalar-dispatch-bundle-ssh-evidence`
  after finish/archive

## Completion Evidence

- Dry-run evidence command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --run-id codex-frontdoor-dispatch-i64-vmul-dry --overwrite`
- Dry-run artifact:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-frontdoor-dispatch-i64-vmul-dry/evidence.json`
- Live ssh evidence command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --ssh-target rvv --run-id codex-frontdoor-dispatch-i64-vmul-live --overwrite`
- Live artifact:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-frontdoor-dispatch-i64-vmul-live/evidence.json`
- Live result:
  `status = success`, `ssh_evidence.success = true`,
  `ssh_evidence.remote_compile_succeeded = true`,
  `ssh_evidence.remote_link_succeeded = true`,
  `ssh_evidence.remote_run_succeeded = true`, and
  `ssh_evidence.output_validation_succeeded = true`.
- Live remote facts recorded:
  `architecture = riscv64`, `clang_path = /usr/bin/clang`, and
  `clang_version = Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Dispatch coverage recorded:
  `rvv_available = 0` selects the scalar fallback branch,
  `rvv_available = 1` selects the RVV dispatch case, and runtime counts
  `7` and `16` are checked with arithmetic output and overrun validation.
- Scope of the live claim:
  bounded RVV+scalar `i64-vmul` target-artifact bundle external caller
  correctness only; no performance, generic lowering, broad correctness, or
  generic RVV backend claim.
- Validation run:
  - `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
  - `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
  - `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
  - `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv . --filter rvv-scalar-dispatch-bundle-e2e`
    from `artifacts/tmp/tianchenrv-build/test`
  - `git diff --check`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
- Previous direct RVV bundle task PRD read:
  - `.trellis/tasks/archive/2026-05/05-10-front-door-rvv-binary-frontend-bundle-ssh-evidence/prd.md`
- Initial implementation surfaces inspected:
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `lib/Support/RuntimeABIParam.cpp`
  - `lib/Support/RuntimeABIContract.cpp`
  - `lib/Transforms/DispatchRuntimeGuard.cpp`
  - `lib/Transforms/ExecutionPlanCoherence.cpp`
