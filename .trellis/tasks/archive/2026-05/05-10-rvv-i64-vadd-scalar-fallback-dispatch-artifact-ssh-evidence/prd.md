# RVV i64-vadd scalar fallback dispatch artifact and ssh evidence

## Goal

Complete the first dtype-aware i64 RVV plus scalar dispatch runtime ABI slice:
starting from a marked `linalg.generic` `memref<?xi64>` `arith.addi` frontend
input, the compiler must produce one selected RVV `i64-vadd` dispatch case,
one scalar `i64-vadd` fallback, generated dispatch artifacts, and real
`ssh rvv` evidence proving both runtime dispatch branches.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting state is clean on `main` at commit `178852e`.
- No `.trellis/.current-task` existed, so this task was created for the
  provided Hermes brief.
- The previous task
  `rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence` is archived at
  commit `178852e` and must not be reopened.
- Recent direct RVV i64 add/sub/mul work made the RVV-owned family registry and
  direct RVV frontend artifact path dtype-aware.
- The scalar fallback and RVV+scalar dispatch boundaries are still shaped
  around i32-family descriptors, scalar microkernel ops, and i32 dispatch
  route manifests.
- The required module boundary is not another direct RVV artifact: it is
  carrying the existing RVV `i64-vadd` selected variant across scalar fallback,
  dispatch metadata, generated source/header/object or self-check artifacts,
  and real RVV branch evidence.

## Boundaries

- Compiler behavior remains implemented in C++ / MLIR / LLVM / TableGen /
  CMake / lit / FileCheck.
- Python may only act as runner/probe/export/evidence tooling. It must not own
  compiler IR, dialects, passes, descriptor lookup, plugin registry,
  capability model, lowering, emission, or dispatch decisions.
- `tcrv.exec` remains compute-free. Scalar fallback surfaces remain under
  `tcrv.scalar` / scalar plugin-owned code; RVV surfaces remain under
  `tcrv.rvv` / RVV target/plugin code.
- Generic planning, selection, readiness, emission-plan, and target-artifact
  front doors must remain target-neutral and must not grow RVV/scalar/i64/vadd
  branches.
- Runtime branch control remains an explicit ABI/control parameter such as the
  dispatch guard. Hardware facts stay in target/capability records, and
  descriptor-local C names/types/operators stay in target/export descriptor
  data.
- Do not add i64-vsub or i64-vmul dispatch unless descriptor refactoring makes
  them trivial and focused tests plus `ssh rvv` evidence are also provided.
- Do not rename the lower-linalg-i32-binary pass or legacy i32 registry as the
  main result unless it directly blocks the i64-vadd dispatch behavior.
- Do not claim RVV runtime correctness without real `ssh rvv` evidence.
- Do not commit generated `artifacts/tmp` files, build outputs, raw ssh logs,
  credentials, or private material.

## Requirements

1. Add or refactor the smallest descriptor-owned C++ surface needed so scalar
   fallback and RVV+scalar dispatch metadata are not hard-coded to i32-only
   helpers.
2. Prefer deriving scalar and dispatch route names, ABI identities, C types,
   C operators, function stems, and route manifests from the existing
   `RVVBinaryFamilyRegistry` descriptor or a clearly bounded dtype-aware bridge.
3. Add the scalar plugin-owned `i64-vadd` fallback operation or metadata
   boundary required by the current architecture. It must preserve origin,
   selected variant, required scalar fallback capability, runtime element
   count, mem-window/ABI role requirements, and descriptor-local C type/name
   boundaries.
4. Extend scalar plugin proposal/materialization and target export only as
   needed for `i64-vadd` scalar fallback and RVV+scalar dispatch, preserving
   existing i32 add/sub/mul behavior.
5. From the marked linalg frontend input, the existing planning pipeline must
   produce a `tcrv.exec.kernel` with an RVV `i64-vadd` variant, scalar
   `i64-vadd` fallback variant, dispatch metadata, selected lowering
   boundaries, supported emission plans, and dispatch artifacts sufficient for
   export.
6. Generated dispatch C must call the RVV `i64-vadd` callable when the runtime
   guard indicates RVV availability and call the scalar `i64-vadd` callable
   otherwise.
7. Extend evidence tooling only as runner/probe/export helper support for this
   i64-vadd dispatch route.
8. Real evidence must compile compiler-generated artifacts on `ssh rvv` and
   execute a correctness harness that proves both scalar fallback and RVV
   dispatch branches.
9. Keep the evidence scope bounded to generated `i64-vadd` dispatch artifacts;
   no performance, generic lowering, arbitrary RVV support, or broad runtime
   integration claim.

## Acceptance Criteria

- [x] A descriptor-owned dtype-aware scalar/dispatch bridge exposes `i64-vadd`
      C type, pointer types, operator, function/header stems, route ids,
      runtime ABI identities, runtime glue roles, and external ABI component
      metadata without one-off i64 string piles.
- [x] The scalar dialect/plugin materializes and verifies the scalar
      `i64-vadd` fallback boundary or microkernel under `tcrv_scalar`, with
      stale route/descriptor mismatch failures.
- [x] The marked linalg `memref<?xi64>` `arith.addi` frontend input lowers
      through the existing planning pipeline to a selected RVV dispatch case
      plus scalar fallback dispatch path.
- [x] Target export emits deterministic dispatch source/header/object or
      self-check artifacts for `i64-vadd`, with `int64_t` ABI parameters,
      RVV branch callable symbol, scalar fallback callable symbol, runtime
      element-count parameter, and dispatch guard parameter.
- [x] Focused lit/FileCheck tests cover scalar `i64-vadd` fallback IR,
      frontend-to-dispatch IR, generated dispatch source/header/self-check,
      route mismatch or stale descriptor failure, and at least one i32
      dispatch regression.
- [x] Focused C++ tests cover changed scalar plugin/dialect, RVV plugin if
      touched, target artifact export, and RVV/scalar dispatch export behavior.
- [x] Real `ssh rvv` evidence succeeds for compiler-generated i64-vadd
      dispatch artifacts, records `status=success`, `mode=ssh`,
      `ssh_evidence=true`, branch-specific scalar/RVV observations or
      equivalent markers, and generated RVV/scalar function symbols.
- [x] `check-tianchenrv` passes before the task is archived.
- [x] Trellis task validation passes before archive.

## Completion Evidence

- Descriptor bridge: `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  owns the dtype-aware scalar fallback and RVV+scalar dispatch family surface
  for existing i32 families plus `i64-vadd`.
- Real RVV evidence:
  `artifacts/tmp/rvv_scalar_dispatch_e2e/rvv-i64-vadd-dispatch-ssh/evidence.json`
  records `status=success`, `mode=ssh`, `ssh_evidence_verified=true`,
  `runtime_success=true`, `rvv_available=0`, `rvv_available=1`, runtime
  element counts `[7, 16]`, and the generated RVV/scalar callable symbols.
- Generated runtime ABI recorded by evidence:
  `void tcrv_dispatch_i64_vadd_frontend_bundle_i64_vadd(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n, int rvv_available)`.
- Final validation passed:
  `git diff --check`;
  `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`;
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`;
  focused C++ tests for scalar plugin, binary family registry, and target
  artifact export;
  focused lit/FileCheck tests for scalar IR, dispatch export, target artifact
  bundle, and runner dry-run;
  `python3 scripts/rvv_remote_probe.py --run-id rvv-i64-vadd-dispatch-probe --timeout 90`;
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vadd --lower-linalg-frontend --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vadd-and-export-target-artifact-bundle.mlir --run-id rvv-i64-vadd-dispatch-ssh --overwrite --timeout 120`;
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`;
  and `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 187/187.

## Minimal Validation Plan

- `git diff --check`
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py` if that script changes
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` if that script
  changes
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused C++ tests for scalar plugin/dialect, RVV plugin if touched, target
  artifact export, and RVV/scalar dispatch export.
- Focused lit/FileCheck tests under `test/Dialect/Scalar`,
  `test/Transforms/LinalgToExec`, `test/Target/RVVScalarDispatch`,
  `test/Target/TargetArtifactBundleExport`, and `test/Scripts` as touched.
- Run an `ssh rvv` quick probe before making any RVV runtime claim.
- Run the bounded e2e command that lowers the linalg i64-vadd frontend,
  exports the RVV+scalar dispatch artifact or bundle, compiles/links on
  `ssh rvv`, and records evidence under `artifacts/tmp`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-vadd-scalar-fallback-dispatch-artifact-ssh-evidence`

## Out Of Scope

- No i8/i16/floating dtype support, generic linalg coverage, new hardware
  targets, IME/AME/offload behavior, broad smoke matrices, benchmarks, or
  performance claims.
- No compute semantics in `tcrv.exec`.
- No core-pass branches on RVV/scalar/dtype/family semantics.
- No Python-owned compiler internals or route selection.
- No hand-written standalone C evidence that bypasses compiler-generated
  frontend artifacts.
- No helper-only, PRD-only, report-only, or evidence-packaging-only closeout.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/design-boundaries.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Required archive PRDs read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-binary-family-registry-owner/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-dtype-axis-i64-vadd/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-scalar-vsub-dispatch-ssh-rvv-runtime-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-i32-vmul-rvv-scalar-dispatch-ssh-rvv-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-selected-rvv-scalar-dispatch-abi-artifact-boundary/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline/prd.md`
- Primary source/test surfaces to inspect:
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `lib/Dialect/Scalar/IR/ScalarDialect.cpp`
  - `include/TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `test/Dialect/Scalar/`
  - `test/Transforms/LinalgToExec/`
  - `test/Target/RVVScalarDispatch/`
  - `test/Target/TargetArtifactBundleExport/`
  - `test/Scripts/`

## Continuation Rule If Unfinished

Keep this task open and record the exact incomplete layer: scalar
dialect/plugin, descriptor bridge, frontend planning, dispatch export,
evidence runner, ssh evidence, or build/test failure. Do not archive and do not
claim RVV runtime correctness without real `ssh rvv` evidence.
