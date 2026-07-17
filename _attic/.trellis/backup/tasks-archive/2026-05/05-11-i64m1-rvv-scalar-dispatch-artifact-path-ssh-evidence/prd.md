# i64m1 RVV scalar dispatch artifact path with ssh rvv evidence

## Goal

Complete one bounded `i64-vmul` RVV+scalar dispatch artifact path from the
real MLIR/front-door planning pipeline through selected RVV metadata, scalar
fallback metadata, runtime ABI parameters, plugin-owned RVV+scalar dispatch
source/header/object bundle selection, and real `ssh rvv` compile/link/run
evidence. The proof is intentionally finite: it must not claim a generic RVV
backend, broad family matrix, dynamic runtime integration, performance, or
arbitrary linalg lowering.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean on `main` at
  `d5e2d63 feat(offload): register descriptor exporter via plugin bundle`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  module brief before source changes.
- Recent archived tasks show the previous module moved target artifact exporter
  ownership behind plugin bundles and recorded clean `check-tianchenrv`
  evidence.
- Earlier archived tasks already added finite i64 add/sub/mul RVV direct
  routes and i64 sub/mul scalar fallback dispatch routes; this round must prove
  one wider bounded i64m1 dispatch bundle path through the current compiler
  front door and real RVV host evidence.
- The selected single family for this round is `i64-vmul`, because the brief
  prefers a finite i64 family and the spec/archives already list
  `i64-vmul` as an intended supported route.

## Boundaries

- Compiler behavior stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only orchestrate compiler tools, parse bounded compiler-emitted
  artifacts, generate the external evidence caller from emitted ABI metadata,
  and record sanitized evidence. It must not own compiler semantics, dialects,
  passes, descriptor lookup, route selection, legality, lowering, emission, or
  runtime ABI contracts.
- `tcrv.exec` remains compute-free. i64 multiply semantics belong in finite
  RVV/scalar extension dialects and target/export descriptors.
- Generic planning, selected-boundary, emission-plan, target artifact, and
  bundle front doors must remain target-neutral; i64 family behavior must be
  carried by existing descriptor/manifest/plugin-owned route surfaces.
- Do not modify offload descriptor behavior except for a mechanical build fix.
- Do not commit generated artifacts under `artifacts/tmp`, raw ssh logs,
  credentials, private SSH material, or build outputs.

## Requirements

1. Starting from the real marked-linalg or existing generated module front
   door, carry `i64-vmul` through the existing execution-planning pipeline to a
   selected RVV dispatch case and scalar dispatch fallback.
2. The selected RVV side must preserve i64m1 compile-time metadata, i64
   runtime ABI identity, selected RVV callable route, and `int64_t` ABI
   parameter spellings.
3. The selected scalar fallback side must preserve the matching finite
   `i64-vmul` descriptor, scalar callable route, scalar runtime ABI identity,
   and `int64_t` ABI parameter spellings.
4. RVV+scalar dispatch source/header/object bundle records must be emitted by
   plugin-owned target exporter registration and must share coherent
   `component_group`, `component_role`, `external_abi_name`, runtime ABI
   kind/name, selected component paths, and ordered
   `runtime_abi_parameter[index]` signature metadata.
5. If the runner or tests currently hard-code i32 dispatch assumptions,
   generalize only the runner/evidence surface enough to consume the existing
   compiler-emitted `i64-vmul` descriptors and bundle metadata. Do not
   duplicate compiler route semantics in Python.
6. The real `ssh rvv` evidence must compile/link/run the generated dispatch
   source and generated external caller, and, when the bundle object is present,
   link/run the same caller against that object as well.
7. Evidence must exercise both dispatch branches where the script supports
   forcing them: `rvv_available = 0` for scalar fallback and
   `rvv_available = 1` for RVV, with bounded runtime `n` values.
8. Evidence JSON must record sanitized remote architecture/toolchain facts,
   selected family, artifact paths/hashes, branch coverage, compile/link/run
   success, and no throughput, latency, ratio, or performance fields.
9. Keep validation focused on this changed module behavior: i64-vmul
   front-door bundle path, RVV/scalar/dispatch artifact route coherence,
   runner assertions, focused lit/C++ tests if source changes, real `ssh rvv`
   evidence, and final `check-tianchenrv`.

## Acceptance Criteria

- [x] Trellis task exists, is current, and records this PRD plus spec context.
- [x] The selected family is exactly `i64-vmul`; no family matrix is added.
- [x] `i64-vmul` can be driven through the plan-and-export target artifact
      bundle front door or an equivalent existing real MLIR/front-door route.
- [x] Bundle/index validation proves source/header/object coherence for the
      `i64-vmul` dispatch external ABI and ordered `int64_t` runtime ABI
      signature.
- [x] The evidence runner, if changed, remains runner-only and rejects stale
      family metadata rather than hard-coding compiler route truth.
- [x] Focused lit/FileCheck or script tests cover the selected i64 dispatch
      bundle path and stale i32/vadd assumptions touched by the change.
- [x] Relevant C++ target/plugin tests pass if C++ surfaces are touched.
- [x] Real `ssh rvv` evidence succeeds for `i64-vmul` and records compile,
      link, run, both branch coverage, selected family, and artifact hashes.
- [x] `git diff --check` passes.
- [x] `tcrv-opt`, `tcrv-translate`, and relevant target artifact/export tests
      build.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before finish/archive.
- [x] Trellis task validation passes before finish/archive.

## Minimal Validation Plan

- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` if the runner
  changes.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Build/run focused C++ targets if touched, especially
  `tianchenrv-target-artifact-export-test` and relevant RVV/scalar plugin tests.
- Focused lit/FileCheck for the selected i64 RVV frontend/artifact path and
  RVV+scalar dispatch bundle path.
- Real evidence command, adjusted only if inspection shows a more precise
  existing i64-vmul fixture:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --ssh-target rvv --run-id codex-i64-vmul-dispatch-bundle-ssh --overwrite --timeout 120`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-i64m1-rvv-scalar-dispatch-artifact-path-ssh-evidence`

## Out of Scope

- Generic RVV backend or arbitrary linalg/RVV lowering claims.
- Broad family matrix, performance, throughput, latency, or ratio evidence.
- Moving computation semantics into `tcrv.exec`.
- Hard-coding RVV/scalar semantics into generic core passes or generic target
  artifact routing.
- Python-owned compiler descriptors, legality, route selection, lowering, or
  emission semantics.
- Offload descriptor changes unless required by a build break.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Archived task context read:
  - `.trellis/tasks/archive/2026-05/05-11-rvv-scalar-dispatch-bundle-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-sub-mul-scalar-fallback-dispatch-artifact-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence/prd.md`

## Result

- No compiler source change was required. Current HEAD already carries the
  finite `i64-vmul` RVV+scalar dispatch bundle path through the
  plan-and-export target artifact bundle front door.
- The completed evidence command was:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --ssh-target rvv --run-id codex-i64-vmul-dispatch-bundle-ssh --overwrite --timeout 120`.
- Evidence JSON:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i64-vmul-dispatch-bundle-ssh/evidence.json`.
- Evidence outcome: `status=success`, `mode=ssh`,
  `arithmetic_family=i64-vmul`, `vector_shape=i64m1`,
  `ssh_evidence_verified=true`, `runtime_success=true`,
  `remote_compile_succeeded=true`, `remote_link_succeeded=true`,
  `remote_run_succeeded=true`, and
  `output_validation_succeeded=true`.
- Remote host facts recorded by evidence: architecture `riscv64`, clang
  `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- The generated external caller exercised both `rvv_available=0` and
  `rvv_available=1` branches with runtime element counts `7` and `16`, and
  checked `lhs[index] * rhs[index]` using `int64_t` buffers.
- The selected bundle records used the `i64-vmul` routes
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-c`,
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-header`, and
  `tcrv-export-rvv-scalar-i64-vmul-dispatch-object`, with component group
  `rvv-scalar-i64-vmul-dispatch-external-abi.v1` and ordered ABI parameters
  `lhs`, `rhs`, `out`, `n`, and `rvv_available`.
- No throughput, latency, ratio, or performance claim was made.

## Validation Run

- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`: passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --run-id codex-i64-vmul-dispatch-bundle-dry --overwrite --timeout 120`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`: passed.
- Focused lit from `artifacts/tmp/tianchenrv-build/test`: 5 focused tests
  passed covering script bundle e2e, `i64-vmul` plan-and-export bundle,
  `i64-vmul` RVVScalarDispatch source/header/object routes, and linalg
  frontend RVV artifact path.
- `git diff --check`: passed.
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
  and `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-i64m1-rvv-scalar-dispatch-artifact-path-ssh-evidence`: passed.

## Continuation Rule If Unfinished

Keep this task open and record the exact failed boundary: MLIR lowering/
selection, plugin proposal or legality, selected lowering boundary, emission
plan, bundle index, source/header/object export, remote compile, remote link,
remote run, branch evidence, or final aggregate check. Do not archive and do
not claim RVV runtime correctness without real `ssh rvv` evidence.
