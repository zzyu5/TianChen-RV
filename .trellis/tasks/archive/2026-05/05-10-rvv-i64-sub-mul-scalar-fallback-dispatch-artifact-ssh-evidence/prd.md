# RVV i64 sub/mul scalar fallback dispatch artifact and ssh evidence

## Goal

Carry the already completed RVV `i64-vsub` and `i64-vmul` frontend/direct
artifact slices through the same scalar fallback plus RVV/scalar host dispatch
artifact path that `i64-vadd` now uses, then prove both dispatch branches with
real `ssh rvv` evidence for compiler-generated artifacts.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree is clean on `main` at `dc9c6c5`.
- No `.trellis/.current-task` existed, so this task was created from the
  fallback Hermes brief after reading the latest supervisor audit and review
  input.
- Latest completed task `rvv-i64-vadd-scalar-fallback-dispatch-artifact-ssh-evidence`
  finished and archived the `i64-vadd` scalar fallback dispatch path with real
  `ssh rvv` branch evidence.
- Earlier task `rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence`
  already made `i64-vsub` and `i64-vmul` first-class RVV binary families for
  marked linalg frontend, RVV plugin materialization, target artifact export,
  and direct `ssh rvv` evidence.
- Current `include/TianChenRV/Target/RVVScalarBinaryFamily.h` builds the
  scalar/dispatch bridge from RVV family descriptors, but the exported bridge
  set currently contains only i32 add/sub/mul plus `i64-vadd`.
- Current scalar dialect/plugin/export code recognizes
  `tcrv_scalar.i64_vadd_microkernel` but not `i64_vsub` / `i64_vmul`.
- Current RVV+scalar dispatch manifest and direct wrappers register i32
  add/sub/mul and `i64-vadd`, not `i64-vsub` / `i64-vmul`.
- The module-sized continuation is therefore not another direct RVV path or a
  smoke/probe task; it is completing the sub/mul dispatch/fallback path across
  descriptor bridge, scalar plugin-local microkernel, target artifact export,
  bundle/front-door tests, runner support, and real RVV branch evidence.

## Boundaries

- Compiler behavior remains C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only act as runner/probe/export/evidence tooling. It must not own
  compiler IR, dialects, passes, descriptor lookup, plugin registry,
  capability model, lowering, emission, dispatch selection, or ABI contracts.
- `tcrv.exec` remains compute-free: only kernel/variant/dispatch/fallback,
  mem_window, runtime_param, capability, and diagnostics surfaces may be used.
- i64 subtract/multiply semantics belong in RVV/scalar extension dialects and
  target/export descriptors, not in core passes.
- Shared planning, selection, coherence, target artifact front doors, and
  bundle routing must remain target-neutral. Add family members through
  descriptor/manifest/plugin-local surfaces, not core RVV/scalar/dtype branches.
- Do not claim RVV runtime/correctness without fresh real `ssh rvv` evidence.
- Do not commit generated `artifacts/tmp`, build outputs, raw ssh logs,
  credentials, or private material.
- No performance claim, generic linalg lowering claim, arbitrary dtype support,
  offload/IME/AME work, or broad matrix run is in scope.

## Requirements

1. Extend the descriptor-owned RVV+scalar binary bridge so the family set
   contains i32 add/sub/mul and i64 add/sub/mul, with scalar route IDs,
   runtime ABI names, C types/operators, dispatch routes, self-check markers,
   external ABI component groups, and helper metadata derived from the RVV
   binary family descriptors.
2. Add bounded scalar plugin-local `tcrv_scalar.i64_vsub_microkernel` and
   `tcrv_scalar.i64_vmul_microkernel` surfaces as siblings of existing
   i64-vadd. They must verify the same selected-path, required-capability,
   source-kernel, role, and element-count contracts.
3. Extend scalar plugin proposal/materialization/readiness/plan/export routing
   to materialize and export i64 sub/mul scalar fallback callables from the
   selected descriptor, preserving i32 add/sub/mul and i64-vadd behavior.
4. Extend RVV+scalar dispatch route registration and direct wrappers so
   `i64-vsub` and `i64-vmul` dispatch source/header/object/self-check routes
   are visible to `tcrv-translate` through the manifest and to generic target
   artifact bundle export.
5. Starting from marked linalg `memref<?xi64>` `arith.subi` and `arith.muli`
   frontend inputs, the existing planning pipeline must produce one selected
   RVV dispatch case, one scalar fallback, typed runtime guard linkage,
   supported emission plans, selected lowering boundaries, and dispatch bundle
   artifacts for each family.
6. Generated dispatch C for sub must call RVV/scalar sub callables and validate
   `lhs - rhs`; generated dispatch C for mul must call RVV/scalar mul callables
   and validate `lhs * rhs`. Neither path may inherit stale vadd route IDs,
   symbols, intrinsics, ABI names, success markers, or `lhs + rhs` checks.
7. Extend `scripts/rvv_scalar_dispatch_e2e.py` only as evidence tooling so it
   can drive compiler-generated i64-vsub and i64-vmul dispatch artifacts through
   the same local dry-run and real `ssh rvv` evidence contract as i64-vadd.
8. Real evidence must compile generated dispatch artifacts on `ssh rvv` and run
   correctness harnesses that observe both `rvv_available=0` and
   `rvv_available=1` branches for i64-vsub and i64-vmul.
9. Keep validation focused on the changed module behavior: descriptor bridge,
   scalar dialect/plugin/export, dispatch exporter, bundle/front-door tests,
   runner self-test/dry-runs, and bounded real RVV evidence.

## Acceptance Criteria

- [x] RVV+scalar bridge lookup by family id, frontend lowering, lowering
      descriptor, scalar route id, and dispatch route id works for
      `i64-vsub` and `i64-vmul`.
- [x] Scalar dialect parses/verifies `tcrv_scalar.i64_vsub_microkernel` and
      `tcrv_scalar.i64_vmul_microkernel`.
- [x] Scalar plugin materializes descriptor-backed i64-vsub/i64-vmul fallback
      microkernels and emits supported scalar callable emission plans with
      `int64_t` ABI parameters and family-specific route/ABI metadata.
- [x] RVV+scalar dispatch source/header/object/self-check routes are registered
      for `i64-vsub` and `i64-vmul`, with generic target source/header/object
      and bundle front doors selecting the family-specific composite routes.
- [x] Marked linalg i64-vsub and i64-vmul frontend inputs lower through the
      existing planning pipeline to selected RVV + scalar dispatch artifacts.
- [x] Focused lit/FileCheck tests cover direct dispatch export, plan-and-export
      bundle output, stale-family negative checks, and i32/i64-vadd regressions
      where touched.
- [x] Focused C++ tests cover descriptor/registry, scalar plugin, and target
      artifact exporter route registration/preflight.
- [x] `scripts/rvv_scalar_dispatch_e2e.py --self-test` covers i64 sub/mul
      family metadata, self-check parsing, bundle index parsing, and generated
      external caller arithmetic.
- [x] Real `ssh rvv` evidence succeeds for compiler-generated i64-vsub and
      i64-vmul dispatch artifacts, records `status=success`, `mode=ssh`,
      `ssh_evidence_verified=true`, `runtime_success=true`, branch coverage
      for scalar and RVV, runtime counts, and generated callable/dispatcher
      symbols.
- [x] `check-tianchenrv` passes before archive when feasible.
- [x] Trellis task validation passes before archive.

## Minimal Validation Plan

- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` if the script
  changes
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused C++ tests:
  `tianchenrv-i32-binary-family-registry-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`
- Focused lit/FileCheck for scalar dialect, RVVScalarDispatch, target artifact
  bundle export, and script dry-run paths touched by i64 sub/mul.
- Real evidence commands, with exact input files adjusted to the final tests:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vsub --lower-linalg-frontend --input <i64-vsub-bundle-input> --run-id <bounded-id> --overwrite --timeout 120`
  and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vmul --lower-linalg-frontend --input <i64-vmul-bundle-input> --run-id <bounded-id> --overwrite --timeout 120`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-sub-mul-scalar-fallback-dispatch-artifact-ssh-evidence`

## Completion Evidence

- Direct dry-runs passed for `i64-vsub` and `i64-vmul`.
- Plan-and-export target artifact bundle dry-runs passed for `i64-vsub` and
  `i64-vmul`.
- Real `ssh rvv` evidence passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vsub --run-id codex-i64-vsub-ssh --overwrite --timeout 120`
- Real `ssh rvv` evidence passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vmul --run-id codex-i64-vmul-ssh --overwrite --timeout 120`
- Evidence JSON paths:
  `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-i64-vsub-ssh/evidence.json`
  and
  `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-i64-vmul-ssh/evidence.json`
  report `status=success`, `mode=ssh`, `ssh_evidence_verified=true`,
  `runtime_success=true`, scalar/RVV branch coverage, and runtime counts
  `[7, 16]`.
- Final aggregate validation passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  with 191/191 tests passed.

## Out Of Scope

- No new i8/i16/floating dtype support, generic linalg support, masks, new
  LMUL selection, performance benchmarking, offload, IME, AME, or future ISA
  work.
- No compute semantics in `tcrv.exec`.
- No core-pass family/dtype dispatch.
- No Python-owned compiler internals.
- No handwritten standalone C evidence that bypasses compiler-generated
  frontend/dispatch artifacts.
- No helper-only, PRD-only, report-only, broad smoke-matrix, or guardrail-only
  closeout.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/design-boundaries.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Required archive context read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-scalar-fallback-dispatch-artifact-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence/prd.md`
- Supervisor inputs read:
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0035-20260510T020335Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0035-20260510T020335Z/review_input.md`
  - `artifacts/tmp/hermes_codex_supervisor/asks/20260510T021737Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/asks/20260510T021737Z/review_input.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `lib/Dialect/Scalar/IR/ScalarDialect.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `test/Dialect/Scalar/`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Target/RVVScalarDispatch/`
  - `test/Target/TargetArtifactBundleExport/`
  - `test/Scripts/`

## Continuation Rule If Unfinished

Keep this task open and record the exact incomplete layer: descriptor bridge,
scalar dialect/plugin, scalar target export, dispatch route/export, frontend
bundle fixture, evidence runner, real `ssh rvv` evidence, or build/test
failure. Do not archive and do not claim RVV runtime correctness without real
`ssh rvv` evidence.
