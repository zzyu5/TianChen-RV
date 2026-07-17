# Descriptor mirror retirement from production source-authority surfaces

## Goal

Retire legacy lowering descriptor mirrors from the default production
source-authority and runtime-ABI planning surfaces now that direct RVV, scalar
fallback, and RVV+scalar dispatch source routes are backed by typed
extension-family ops and common EmitC materialization.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; task-start worktree was clean.
- Task-start HEAD is `49bacb3 feat(dispatch): emit rvv scalar wrapper via emitc`.
- `.trellis/.current-task` was absent, so this task was created as
  `.trellis/tasks/05-12-descriptor-mirror-retirement-from-production-surfaces/`.
- The previous task
  `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-emitc-source-authority-production-route/`
  is archived and must not be reopened.
- The recent direct RVV, scalar fallback, and RVV+scalar dispatch source routes
  already use typed family/body authority and common MLIR EmitC / Cpp emitter
  source authority.
- The current Support runtime ABI callable planning API still imports target
  descriptor/registration headers and exposes descriptor-shaped overloads.
- Existing target registries may keep finite route-registration records, but
  production selected source/ABI planning must treat descriptor fields only as
  quarantined legacy mirrors or compatibility validation inputs.

## Requirements

- Remove target descriptor headers from the default Support runtime ABI
  callable-planning API surface.
- Introduce or reuse a generic Support-layer finite binary callable/runtime ABI
  contract that carries only selected family id, callable parameter shape,
  mem_window/runtime_param requirements, runtime ABI identities, and optional
  dispatch identity.
- Update RVV, scalar, dispatch, and offload call sites so they translate their
  selected typed family/registration data into the generic Support contract
  before invoking callable ABI planning or mirror validation.
- Keep target-owned finite registration tables where they are still needed for
  bounded family enumeration, route registration, intrinsic names, selected
  shape, and explicit legacy mirror validation.
- Ensure positive/default RVV, scalar, and dispatch source/bundle paths do not
  require `tcrv_rvv.selected_lowering_descriptor`,
  `tcrv_scalar.selected_lowering_descriptor`,
  `tcrv_rvv.lowering_descriptor`, or `tcrv_scalar.lowering_descriptor` to
  choose compute semantics, callable ABI shape, component calls, or route source
  authority.
- Keep stale descriptor mirror inputs fail-closed when explicitly accepted as
  legacy mirrors; a stale descriptor must not alter arithmetic, dtype/vector
  shape, ABI order, or selected component calls.
- Update focused positive tests to assert typed family/body authority, selected
  operation/family id, runtime ABI contracts, EmitC source route metadata, and
  descriptorless source authority instead of descriptor-positive metadata.
- Quarantine remaining descriptor assertions as legacy negative or
  mirror-validation tests.
- Keep `tcrv.exec` compute-free and keep Python tooling-only.

## Acceptance Criteria

- [x] `include/TianChenRV/Support/RuntimeABICallablePlan.h` no longer includes
      target descriptor or family registry headers.
- [x] Support callable planning default overloads accept Support-owned generic
      runtime ABI contracts, not RVV/scalar descriptor structs.
- [x] Focused C++ tests prove descriptorless Support callable planning and
      runtime ABI mirror validation for at least i32 shared and direct RVV
      finite binary contracts.
- [x] Direct RVV, scalar fallback, and RVV+scalar dispatch production source
      artifacts still route through common EmitC source authority and still
      export coherent artifacts.
- [x] Positive/default lit/FileCheck fixtures no longer require selected
      lowering descriptor mirror metadata.
- [x] Explicit stale descriptor mirror fixtures remain fail-closed and are
      labeled/worded as legacy non-authoritative mirror validation.
- [x] Generated default source/bundle comments prefer typed family/body
      authority, selected family id, runtime ABI, EmitC route/source authority,
      vector shape/config, and artifact route id.
- [x] `git diff --check`, Trellis task validation, focused C++/lit checks, and
      `check-tianchenrv -j2` pass if the build tree is usable.
- [x] One coherent commit records the completed module, or the task remains
      open with the exact descriptor surface left.

## Non-Goals

- No new RVV/scalar family, dtype, LMUL, benchmark, throughput, latency, or
  performance claim.
- No new generic RVV backend, MLIR vector/scalable-vector route, IME, AME,
  Sophgo/offload, Template, or Toy work.
- No compute semantics in `tcrv.exec`.
- No Python compiler implementation.
- No helper-only, metadata-only, wrapper-only, smoke-only, report-only, or
  standalone ssh-evidence closeout.
- No deletion of finite family registration data that still enumerates the
  supported bounded family set; it may be renamed or narrowed only after the
  default path stops treating it as compute/source/ABI authority.
- No fresh RVV runtime claim unless a bounded `ssh rvv` slice is rerun through
  the changed production path.

## Technical Approach

Add a Support-owned finite binary runtime ABI contract shape that is independent
of target descriptor structs. The contract will own the reusable callable ABI
parameter order, mem_window/runtime_param role requirements, selected family id,
and source/runtime identity strings. Support callable planning and parameter
mirror validation will consume that contract only.

Target-owned RVV/scalar/dispatch code will remain responsible for selecting a
typed family/body and for consulting finite registration records where needed.
Those target layers will build Support generic contracts from selected typed
family registration data, then pass the generic contract to Support planning.
Descriptor mirror fields may still be checked after typed authority is known,
but they will no longer be accepted by Support as the default callable-planning
authority.

## Decision (ADR-lite)

**Context**: Source production has moved to typed extension-family ops and
common EmitC, but the Support runtime ABI callable planning API still exposes
target descriptor-shaped overloads.

**Decision**: Make Support callable planning descriptorless by default through a
Support-owned generic finite binary runtime ABI contract, and move target
descriptor/registration-to-contract adaptation into target/plugin code.

**Consequences**: Call sites that previously passed RVV or i32 descriptor
records must explicitly build/pass the selected ABI contract. Tests should
assert contract shape, typed authority, and fail-closed legacy mirror behavior
instead of descriptor-positive authority.

## Minimal Validation

- Build focused owners:
  `TianChenRVSupport`, `TianChenRVConversionEmitC`, `TianChenRVRVVTarget`,
  `TianChenRVScalarTarget`, `TianChenRVBuiltinTargetArtifactExporters`,
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-runtime-abi-callable-plan-test`, and
  `tianchenrv-target-artifact-export-test` as applicable.
- Run focused C++ tests for runtime ABI callable planning, target artifact
  export, RVV/scalar plugin planning, and i32 registry behavior as touched.
- Run focused lit/FileCheck under `test/Transforms/LinalgToExec`,
  `test/Target/ArtifactExport`, `test/Target/TargetArtifactBundleExport`, and
  `test/Target/RVVScalarDispatch` for changed fixtures.
- Run direct and dispatch dry-run validation only if source/object export
  fallout requires it.
- Run `git diff --check`, Trellis task validation, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- Prior PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-emitc-source-authority-production-route/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-scalar-emitc-source-authority-production-route/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-emitc-source-authority-production-route/prd.md`
- Initial source focus:
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `include/TianChenRV/Support/RuntimeABIContract.h`
  - `lib/Support/RuntimeABIContract.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
  - focused tests under `test/Support`, `test/Target`, `test/Plugin`, and
    `test/Transforms/LinalgToExec`.

## Definition Of Done

This task is complete only when the default Support callable ABI planning API is
descriptorless, production RVV/scalar/dispatch source and bundle paths remain
typed-family/EmitC authoritative, legacy descriptors are quarantined as
non-authoritative validation mirrors, focused checks pass, Trellis state is
truthful, and one coherent commit records the module. If unfinished, leave the
task open with the exact remaining descriptor surface.

## Completion Notes

- Added Support-owned `FiniteBinaryRuntimeABIContract` /
  `FiniteBinaryRuntimeABIContractSpec` and made `I32BinaryRuntimeABIContract` a
  descriptorless Support contract keyed by selected family id.
- Removed target descriptor and i32 family registry includes from
  `RuntimeABICallablePlan.h` and `RuntimeABIContract.h`.
- Retired descriptor-shaped Support callable-plan overloads. The active
  planning and runtime ABI mirror validation APIs now consume the generic
  Support contract or a selected i32 family id.
- Reworked target-owned RVV runtime ABI contracts to derive from the generic
  Support finite-binary contract while keeping RVV family registration identity
  target-local.
- Updated RVV, scalar fallback, RVV+scalar dispatch, and offload descriptor
  call sites to pass selected-family runtime ABI contracts instead of
  descriptor/registration records into Support planning.
- Updated focused C++ coverage to prove i32 callable planning is keyed by
  selected family id and direct RVV i64 planning uses the target-adapted generic
  contract.
- Updated the lowering/runtime spec so future work treats descriptor-shaped
  Support callable-planning overloads as a descriptor-exit regression.
- No fresh `ssh rvv` execution was performed; this task makes no RVV runtime,
  correctness, or performance claim.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVSupport tianchenrv-runtime-abi-callable-plan-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVBuiltinTargetArtifactExporters TianChenRVOffloadTarget tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test tianchenrv-scalar-extension-plugin-test tianchenrv-rvv-binary-planning-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv . --filter 'Target/ArtifactExport|Target/TargetArtifactBundleExport|Target/RVVScalarDispatch|Transforms/LinalgToExec'` from `artifacts/tmp/tianchenrv-build/test`, 50/50 passed.
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vmul --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --tcrv-opt artifacts/tmp/tianchenrv-build/bin/tcrv-opt --tcrv-translate artifacts/tmp/tianchenrv-build/bin/tcrv-translate --artifact-root artifacts/tmp/descriptor_mirror_retirement_dryrun --run-id direct-i32-vmul --overwrite`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family i32-vmul --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --tcrv-opt artifacts/tmp/tianchenrv-build/bin/tcrv-opt --tcrv-translate artifacts/tmp/tianchenrv-build/bin/tcrv-translate --artifact-root artifacts/tmp/descriptor_mirror_retirement_dryrun --run-id dispatch-i32-vmul --overwrite`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-descriptor-mirror-retirement-from-production-surfaces`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`, 209/209 passed.
