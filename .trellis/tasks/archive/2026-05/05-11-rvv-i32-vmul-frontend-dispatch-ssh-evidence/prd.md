# RVV i32-vmul frontend-to-dispatch ssh evidence slice

## Goal

Close one bounded `i32-vmul` dispatch handoff through the existing
TianChen-RV C++/MLIR path: marked finite linalg frontend or existing
`i32-vmul` dispatch fixture, selected RVV `i32-vmul` dispatch case, selected
scalar `i32-vmul` fallback, plugin-owned lowering boundaries, target-owned
dispatch artifact export, and fresh `ssh rvv` compile/link/run evidence from
the generated external self-check.

This is one dispatch boundary slice. It is not a finite-family matrix, not a
performance run, and not a generic RVV backend claim.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state was clean on `main` at
  `8a108de test(rvv): verify i32 vmul frontend ssh evidence`.
- No `.trellis/.current-task` existed before this task was created.
- The immediately previous task
  `rvv-i32-vmul-frontend-artifact-ssh-evidence` is archived at
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence/`
  and must not be reopened.
- That archived task proved the direct `i32-vmul` frontend-to-RVV
  source/header/object helper path with fresh `ssh rvv` evidence, but it did
  not make the dispatch boundary the active closure.
- Older dispatch evidence tasks already prove adjacent paths such as
  `i32-vmul` RVV+scalar dispatch bundle evidence, `i32m2` `i32-vsub`, and
  `i64m1` `i64-vsub`/`i64-vmul` front-door dispatch evidence. This round must
  revalidate the current HEAD path for the bounded `i32-vmul` dispatch slice.
- `test/Scripts/rvv-scalar-dispatch-e2e.test` already has dry-run dispatch
  coverage for `i32-vmul`; dry-run coverage must not be described as runtime
  correctness evidence.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only orchestrate compiler tools, artifact parsing, evidence
  capture, and remote compile/link/run checks. It must not implement compiler
  core, dialects, capability model, dispatch selection, lowering, emission,
  route selection, or family semantics.
- Use a real compiler path: either a marked finite `i32-vmul` linalg frontend
  input that feeds the existing execution-planning pipeline, or the existing
  `i32-vmul` dispatch fixture. Do not treat hand-authored final dispatch C as
  compiler truth.
- The selected RVV side must preserve `i32-vmul` descriptor metadata,
  selected vector-shape metadata, runtime ABI metadata, and a generated RVV
  intrinsic path containing `__riscv_vmul_vv_i32m1` unless the selected shape
  explicitly differs.
- The selected scalar fallback side must preserve `i32-vmul` descriptor
  metadata, scalar runtime ABI metadata, and scalar multiply behavior
  `lhs * rhs`. It must not inherit vadd/vsub or i64 names, route ids, ABI
  names, function stems, or arithmetic.
- Dispatch export must consume the selected RVV callable candidate plus the
  selected scalar fallback callable candidate through the existing dispatch
  artifact route and preserve the runtime dispatch guard as ABI/control data
  distinct from compile-time vector shape/config metadata and descriptor-local
  element counts.
- Generated source/header/object/bundle metadata must carry selected family,
  routes, runtime ABI kind/name, component group, external ABI name, ordered
  runtime ABI parameters, and conservative evidence roles consistently.
- Fresh `ssh rvv` evidence must compile, link, and run the generated dispatch
  external self-check successfully and record enough deterministic data to
  audit: run id, input path, selected family, selected kernel, selected routes,
  artifact hashes, compile/link/run status, and observed success marker.
- Focused lit/FileCheck regression must distinguish dry-run export/tooling
  evidence from real ssh runtime correctness evidence. Dry-run checks must not
  claim runtime success.
- If selected-plan route metadata or runtime ABI preflight is stale or missing
  for this path, it must fail closed through existing focused negative coverage
  or a new narrowly scoped regression.

## Acceptance Criteria

- A current HEAD `i32-vmul` dispatch input lowers/plans through real C++/MLIR
  passes to a selected dispatch with an RVV `i32-vmul` case and scalar
  `i32-vmul` fallback.
- Generated IR/artifacts visibly contain RVV `i32-vmul` selected metadata,
  scalar `i32-vmul` fallback metadata, runtime ABI/control metadata, and
  descriptor-local metadata in their proper layers.
- Generated dispatch source contains `__riscv_vmul_vv_i32m1` and a scalar
  fallback multiply branch, with no stale vadd/vsub or i64 symbol leakage.
- The existing `scripts/rvv_scalar_dispatch_e2e.py` workflow succeeds for the
  bounded `i32-vmul` dispatch slice in dry-run mode and in real
  `--ssh-target rvv` mode.
- The real ssh evidence JSON reports success only after compile/link/run
  success and observed bounded `i32-vmul` dispatch marker.
- A focused lit/FileCheck regression covers the current `i32-vmul` dispatch
  handoff/evidence fields without expanding into a broad family matrix.
- `git diff --check` passes.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-i32-vmul-frontend-dispatch-ssh-evidence` passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run if focused checks and build health make it practical.

## Non-Goals

- No broad dispatch matrix or all-family evidence sweep.
- No new RVV family, scalar family, dtype, LMUL, vector shape, tuning path,
  performance benchmark, throughput/latency claim, or generic RVV backend
  claim.
- No MLIR vector/scalable-vector lowering route.
- No IME, AME, Sophgo/offload, Template, or Toy work.
- No compute semantics in `tcrv.exec`.
- No Python compiler internals.
- No helper-only, report-only, fixture-only, smoke-only, or standalone
  evidence-packaging-only closeout if the dispatch path itself is broken.

## Validation Plan

- Build focused targets needed for the dispatch path:
  `tcrv-opt`, `tcrv-translate`, RVV/scalar plugin tests, target artifact export
  tests, and script dependencies as needed.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run the focused dry-run lit slice in
  `test/Scripts/rvv-scalar-dispatch-e2e.test`.
- Run `scripts/rvv_scalar_dispatch_e2e.py` for `i32-vmul` with
  `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door` and
  `--ssh-target rvv`, storing evidence under
  `artifacts/tmp/rvv_scalar_dispatch_e2e/...` or the script's current
  deterministic artifact root.
- Inspect the generated evidence JSON, source/header/bundle index, and command
  summaries for family-correct metadata and a bounded runtime success marker.
- Add or update one focused lit/FileCheck regression if current coverage does
  not already assert the needed `i32-vmul` dispatch evidence fields and dry-run
  boundary.
- Run `git diff --check`.
- Run the Trellis task validator.
- Run full `check-tianchenrv -j2` if practical after focused checks pass.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-09-i32-vmul-rvv-scalar-dispatch-ssh-rvv-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-vsub-frontdoor-dispatch-ssh-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-vmul-frontdoor-dispatch-ssh-evidence/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-11-i64m1-rvv-scalar-dispatch-artifact-path-ssh-evidence/prd.md`.
- Primary code/test surfaces to inspect:
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Scripts/rvv-scalar-dispatch-e2e.test`,
  `test/Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir`,
  `test/Target/TargetArtifactBundleExport/`,
  `test/Transforms/LinalgToExec/`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `lib/Transforms/VariantSelection.cpp`,
  `lib/Transforms/VariantDispatchSynthesis.cpp`,
  `lib/Transforms/LoweringBoundary.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Plugin/RVV/`,
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`, and
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.

## Definition Of Done

The task is done when the bounded current HEAD `i32-vmul` dispatch handoff
exports and runs through the selected RVV plus scalar fallback dispatch
artifact path, focused local regressions pass, fresh real `ssh rvv` evidence
supports exactly this bounded dispatch runtime correctness claim, Trellis task
validation passes, the task is finished/archived, and one coherent commit
records the work.

If unfinished, leave this task open and record the exact continuation point:
frontend fixture, execution planning, variant selection, dispatch synthesis,
RVV lowering boundary, scalar fallback boundary, emission readiness, route
metadata preflight, dispatch source export, dispatch object export, ssh
compile, ssh run, evidence capture, lit regression, Trellis validation, or
commit.

## Completion Notes

Completed in this round:

- Confirmed the current HEAD C++/MLIR path already carries bounded
  `i32-vmul` linalg frontend input through the plan-and-export bundle front
  door into a selected RVV `i32-vmul` dispatch case and selected scalar
  `i32-vmul` fallback.
- Confirmed the generated dispatch source contains the selected RVV intrinsic
  path `__riscv_vmul_vv_i32m1`, scalar fallback `lhs * rhs`, selected kernel
  `frontend_bundle_i32_vmul`, RVV callable symbol
  `tcrv_rvv_i32_vmul_microkernel_frontend_bundle_i32_vmul_rvv_first_slice`,
  and scalar callable symbol
  `tcrv_scalar_i32_vmul_microkernel_frontend_bundle_i32_vmul_scalar_fallback_first_slice`.
- Confirmed bundle metadata preserves the `i32-vmul` external ABI group:
  `rvv-scalar-i32-vmul-dispatch-external-abi.v1`, runtime ABI name
  `rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1`, source route
  `tcrv-export-rvv-scalar-i32-vmul-dispatch-c`, header route
  `tcrv-export-rvv-scalar-i32-vmul-dispatch-header`, and object route
  `tcrv-export-rvv-scalar-i32-vmul-dispatch-object`.
- Confirmed runtime ABI/control fields stay distinct from compile-time
  selected vector-shape metadata and descriptor-local fields: bundle records
  carry ordered `lhs`, `rhs`, `out`, runtime `n`, and
  `dispatch-availability-guard` ABI parameters, while selected-plan metadata
  separately records `tcrv_rvv.selected_binary_family = i32-vmul`,
  `tcrv_scalar.selected_binary_family = i32-vmul`, selected RVV shape
  `i32m1`, and descriptor-local element metadata.
- Added a focused lit regression in
  `test/Scripts/rvv-scalar-dispatch-e2e.test` for
  `scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul`.
  The regression checks the dry-run/non-runtime claim boundary, source/header/
  object bundle artifacts, `i32-vmul` source snippets, route metadata,
  runtime ABI roles, selected descriptor metadata, and absence of
  `runtime_success`/performance claims in dry-run evidence.
- No C++/MLIR/TableGen/CMake/compiler implementation fix was needed. RVV and
  scalar behavior remained plugin/target-owned, and `tcrv.exec` stayed
  compute-free.

Fresh ssh evidence:

- Command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul
  --expect-selected-kernel=frontend_bundle_i32_vmul --run-id
  codex-i32-vmul-dispatch-bundle-ssh-20260511 --overwrite --timeout 120
  --ssh-target rvv`.
- Evidence path:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i32-vmul-dispatch-bundle-ssh-20260511/evidence.json`.
- Result: `status=success`, `pass_fail_result=pass`, `mode=ssh`,
  `runtime_success=true`, `ssh_evidence.success=true`, and
  `ssh_evidence_verified=true`.
- Selected input:
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir`.
- Selected family and vector shape: `i32-vmul`, `i32m1`.
- Artifact hashes recorded:
  source `b276da89c75a889cc13972fd6cae30e5704e8a13c1d8b16e41a1e861cbe198c9`,
  header `7b28af5a863996b49a0a75cd1419c27fbbd1f02285ec259a596d4078d8d24eb1`,
  object `d92c374ce04f9969cb39e352527563252694cd49a5e5a28b3245e50e99237a34`,
  and generated external caller
  `049bb66eb8efd2993bfae7d7751da577c57657276f8f81338e1ea07004b78a93`.
- Remote host facts: `ssh rvv`, `riscv64`,
  `/usr/bin/clang`, `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Compile/link flags: compile `-O2 -march=rv64gcv -mabi=lp64d`, link
  `-O2 -march=rv64gcv -mabi=lp64d -no-pie`.
- Remote compile/link/run succeeded for both the source-built dispatch object
  and the compiler-emitted bundle object external caller paths. Both observed
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16
  branches=scalar_and_rvv`.
- Runtime correctness claim is bounded only to this generated RVV+scalar
  `i32-vmul` target-artifact bundle external caller path on `ssh rvv`; it is
  not a performance claim, broad dispatch matrix, generic high-level lowering
  claim, or arbitrary RVV backend claim.

Validation completed:

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-scalar-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i32-vmul
  --expect-selected-kernel=frontend_bundle_i32_vmul --run-id
  codex-i32-vmul-dispatch-bundle-dry --overwrite --timeout 120`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter rvv-scalar-dispatch-e2e`
  from `artifacts/tmp/tianchenrv-build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter 'plan-linalg-i32-vmul-and-export-target-artifact-bundle|rvv-scalar-i32-vmul-dispatch-generic-route'`
  from `artifacts/tmp/tianchenrv-build/test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-i32-vmul-frontend-dispatch-ssh-evidence`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  205/205 tests passed.

Spec update judgment:

- No `.trellis/spec/` update is needed. The existing specs already require
  the bounded RVV+scalar dispatch route manifest, runtime ABI/control layering,
  dry-run versus ssh evidence separation, and finite family-specific vsub/vmul
  stale-metadata rejection. This task added the missing current-head focused
  `i32-vmul` dispatch bundle evidence regression and fresh runtime evidence.
