# RVV descriptor compute API exit

## Goal

Remove descriptor-owned arithmetic-expression and scalar-element-type helper
authority from the default RVV direct, scalar fallback, and RVV+scalar dispatch
artifact/export paths. Generated compute, self-check, and provenance behavior
must be derived from typed RVV/scalar family operations, generated EmitC route
provenance, and IR-backed runtime ABI boundaries.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is
  `32ab42d feat(rvv): derive self-check expectations from typed authority`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-rvv-descriptor-compute-api-exit/`.
- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-self-check-typed-authority/`
  already moved RVV direct and RVV+scalar self-check expected arithmetic/type
  away from descriptor helper calls.
- Live inventory still shows descriptor compute APIs and provenance leaks:
  `getCArithmeticCheckExpression`, descriptor `getCOperator`, descriptor
  `getScalarCType`, `cOperator` registration fields, default artifact
  `arithmetic_c_operator` comments, and scalar source emission reading scalar
  element C type from finite descriptor/family metadata.
- Specs define the current main route as extension family ops -> EmitC ->
  intrinsic/runtime C/C++, with descriptor-driven computation only as bounded
  migration debt.
- `tcrv.exec` must stay compute-free.

## Requirements

- Remove public descriptor APIs that format arithmetic expressions or expose
  descriptor-selected scalar element type from RVV descriptor helper types.
- Remove descriptor-style `getCOperator` naming from self-check expectation
  helpers so focused source search is not ambiguous with descriptor APIs.
- Stop printing default RVV direct artifact comments that claim
  `arithmetic_c_operator`; replace them with typed operation / EmitC route /
  runtime ABI provenance.
- Rewire scalar fallback EmitC source rendering so scalar element C type is
  derived from the IR-backed callable ABI parameter plan, not from descriptor
  scalar type metadata.
- Rewire RVV+scalar dispatch self-check arithmetic selection to use the
  validated selected family authority, not string suffix or descriptor operator
  data.
- Remove or quarantine `cOperator` fields from live RVV/scalar route metadata.
  If retained anywhere, it must be explicitly legacy mirror metadata and not
  used by default source, self-check, or provenance rendering.
- Update tests that currently require descriptor arithmetic helper APIs or
  `arithmetic_c_operator` comments.
- Add or preserve focused negative coverage showing stale descriptor mirrors
  fail closed and cannot override typed family-op / EmitC authority.

## Acceptance Criteria

- [ ] `include/`, `lib/`, and active `test/` inventory proves
      `getCArithmeticCheckExpression` is gone and descriptor
      `getCOperator` / `getScalarCType` are not public default-path APIs.
- [ ] Default RVV direct source/header/object/self-check paths do not use
      descriptor `cOperator` or descriptor `scalarCType` for compute,
      self-check, or provenance.
- [ ] Default scalar fallback source/header/object paths derive scalar element
      C type from the validated callable ABI plan and typed EmitC route.
- [ ] Default RVV+scalar dispatch source/header/object/self-check paths derive
      compute/self-check/provenance from selected RVV/scalar component
      authorities and dispatch ABI, not descriptor helper fields.
- [ ] Default generated artifacts no longer contain
      `arithmetic_c_operator`; any legacy descriptor mirror metadata remains
      explicitly labeled as legacy mirror only.
- [ ] Focused tests cover RVV direct, scalar fallback, RVV+scalar dispatch,
      and frontend-to-artifact default paths touched by this task.
- [ ] Focused negative coverage proves stale/mismatched descriptor mirrors fail
      closed for direct RVV and RVV+scalar dispatch paths.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] Focused build targets and lit tests pass; if public headers changed,
      `check-tianchenrv -j2` is run when the build tree is usable.
- [ ] Trellis validation passes before archive.
- [ ] One coherent commit records the completed module, or the task remains
      open with exact remaining APIs/check failures documented.

## Non-Goals

- No new high-level tensor/tile IR, generic RVV backend, broad coverage matrix,
  tuning, performance, runtime evidence, or ssh-only evidence round.
- No compiler core, dialects, passes, plugin registry, capability model,
  lowering, emission, or runtime ABI implemented as Python data structures.
- No compute semantics added to `tcrv.exec`.
- No removal of descriptors needed only for non-compute route registration,
  ABI names, intrinsic names, or explicitly quarantined legacy mirrors unless
  the local edit proves they are obsolete.
- No RVV runtime/correctness/performance claim without fresh `ssh rvv`
  evidence; this task should not need such a claim.

## Minimal Validation

- Run focused source inventory over live `include/`, `lib/`, and active
  `test/` for:
  `getCArithmeticCheckExpression`, `getCOperator`, `getScalarCType`,
  `cOperator`, `arithmetic_c_operator`, `scalarCType`,
  `selected_lowering_descriptor`, and `lowering_descriptor`.
- Build focused targets:
  `TianChenRVRVVTarget`, `TianChenRVScalarTarget`,
  `TianChenRVBuiltinTargetArtifactExporters`, `tcrv-translate`, `tcrv-opt`,
  and `tianchenrv-target-artifact-export-test` if present in the build tree.
- Run focused lit tests under `test/Target/RVVMicrokernel`,
  `test/Target/RVVScalarDispatch`, scalar artifact routes, and
  `test/Transforms/LinalgToExec` files touched by expectations.
- Run `git diff --check` and `git diff --cached --check`.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-rvv-descriptor-compute-api-exit`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Predecessor PRD read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-self-check-typed-authority/prd.md`
- Initial source inventory:
  - `include/TianChenRV/Target/BinarySelfCheckExpectation.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`

## Definition Of Done

The task is complete only when descriptor compute helper APIs are removed or
explicitly quarantined, default RVV direct/scalar/dispatch artifact generation
uses typed family-op + EmitC + IR-backed ABI authority, focused and available
broad checks pass, Trellis state is truthful, and one coherent commit records
the module. If unfinished, leave the task open and record the exact remaining
source function/test continuation point.
