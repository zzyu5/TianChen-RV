# remove-i32-vadd-runtime-abi-compat-wrappers

## Goal

Remove the temporary `I32VAdd*` runtime ABI compatibility wrapper APIs from the
support layer now that active RVV, scalar, dispatch, offload, and target
artifact paths consume the family-aware `I32Binary*` runtime ABI contract
directly. The bounded i32 binary ABI remains compiler-owned C++ support-layer
code; this task narrows the public ABI surface so add/sub/mul owners cannot
accidentally re-enter the obsolete add-only compatibility path.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `79c6096 feat(target): quarantine finite descriptor registries`.
- Worktree was clean at task start and `.trellis/.current-task` was absent.
- Latest Hermes audit/review input
  `artifacts/tmp/hermes_codex_supervisor/runs/20260511T-resume-descriptor-exit-2-r0017-20260511T220848Z/{repo_audit.md,review_input.md}`
  says the previous descriptor-registry quarantine task finished, archived, and
  committed.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` still allows
  `getI32VAddRuntimeABIContract`, `getI32VAddRuntimeABIParameters`,
  `getI32VAddRuntimeABIRoleRequirements`,
  `getI32VAddDispatchRuntimeABIParameters`,
  `getI32VAddBufferMemWindowSpecs`,
  `getI32VAddRuntimeElementCountParamSpec`,
  `getI32VAddDispatchAvailabilityGuardParamSpec`, and
  `buildI32VAddCallableABIPlan` as temporary add-descriptor compatibility
  entry points.
- Current source search found those `I32VAdd*` runtime ABI wrappers only in
  support-layer declarations/definitions, not in active `include/`, `lib/`, or
  `test/` production call sites. Current code already uses
  `I32BinaryRuntimeABIContract`, `buildI32BinaryCallableABIPlan`,
  `getI32BinaryBufferMemWindowSpecs`, and related family-aware APIs.
- Specs require compiler functionality to remain in C++/MLIR/LLVM/TableGen/CMake
  and require descriptor-driven computation/direct descriptor-to-C surfaces to
  shrink toward extension family ops -> common EmitC route.

## Requirements

- Delete the unused `I32VAdd*` runtime ABI support wrappers from public headers:
  `RuntimeABI.h`, `RuntimeABIContract.h`, `RuntimeABIMemWindow.h`,
  `RuntimeABIParam.h`, and `RuntimeABICallablePlan.h`.
- Delete the matching wrapper definitions from
  `lib/Support/RuntimeABIContract.cpp` and
  `lib/Support/RuntimeABICallablePlan.cpp`.
- Keep the family-aware `I32Binary*` APIs intact as the production/default
  support-layer contract for bounded i32 add/sub/mul callable ABI shape,
  role binding, mem windows, runtime params, dispatch guard params, and callable
  ABI plan validation.
- Keep i32-vadd microkernel dialect ops, frontend marker names, route ids, and
  family registration records where they are real bounded family members; this
  task only removes obsolete runtime ABI compatibility APIs, not the vadd
  family itself.
- Update lowering-runtime spec text so it no longer authorizes the deleted
  compatibility wrappers and instead states that active add/sub/mul owners use
  `I32Binary*` APIs directly.
- Add or update focused support-layer tests so the public contract proves:
  family-aware add/sub/mul ABI contracts remain usable, and deleted wrappers
  are not required by production behavior.
- Do not implement compiler core, dialects, passes, plugin registry, ABI
  contracts, lowering, or emission as Python data structures.

## Non-Goals

- No RVV runtime/correctness/performance claim and no required `ssh rvv` run.
- No new family, dtype, LMUL, operation, dispatch route, offload route,
  benchmark, tuning, or source artifact.
- No deletion of valid `i32-vadd` family support, dialect ops, test fixtures,
  or route registration records.
- No descriptor-to-C exporter and no fallback to descriptor-derived compute or
  ABI authority.
- No broad smoke/probe-only task completion.

## Acceptance Criteria

- [x] `rg` over tracked source/test files shows no
  `getI32VAddRuntimeABI*`, `appendI32VAddRuntimeABI*`,
  `makeI32VAddDispatchAvailabilityGuard`,
  `bindI32VAddCallableRuntimeABIParametersByRole`,
  `getI32VAddBufferMemWindowSpecs`,
  `getI32VAddRuntimeElementCountParamSpec`,
  `getI32VAddDispatchAvailabilityGuardParamSpec`,
  `getI32VAddRuntimeElementCountParamSpecs`,
  `getI32VAddDispatchRuntimeParamSpecs`,
  `buildI32VAddCallableABIPlan`, or
  `validateI32VAddCallableABIParameterMirror` declarations/definitions/usages.
- [x] Family-aware `I32Binary*` ABI helpers and
  `I32BinaryRuntimeABIContract` remain available and covered for add/sub/mul.
- [x] Runtime ABI support tests pass after wrapper deletion.
- [x] At least the focused C++ support test target
  `tianchenrv-runtime-abi-callable-plan-test` passes.
- [x] `git diff --check` passes.
- [x] Trellis task validation passes before finish/archive.
- [x] If the existing build tree remains usable, run the focused build target
  and a narrow additional target that catches affected support compilation.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/index.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/plugin-protocol/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- Source audit entry points:
  - `include/TianChenRV/Support/RuntimeABI.h`
  - `include/TianChenRV/Support/RuntimeABIContract.h`
  - `include/TianChenRV/Support/RuntimeABIMemWindow.h`
  - `include/TianChenRV/Support/RuntimeABIParam.h`
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `lib/Support/RuntimeABIContract.cpp`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `test/Support/RuntimeABICallablePlanTest.cpp`

## Task Status Notes

- Created this task because no current Trellis task existed and the latest
  Hermes audit showed the previous descriptor-registry quarantine task was
  already archived.
- Planning evidence: current source search found temporary `I32VAdd*` runtime
  ABI wrappers only in support-layer declarations/definitions; production
  owners already use family-aware `I32Binary*` APIs.
- Removed the obsolete `I32VAdd*` runtime ABI wrapper declarations and
  definitions from support headers/sources while keeping real i32-vadd family
  registration, dialect, route, and microkernel support intact.
- Added runtime ABI callable-plan coverage proving the `I32BinaryRuntimeABIContract`
  remains family-aware for add/sub/mul identity and shared callable parameter
  shape.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` so the
  stable contract no longer permits temporary add-only ABI wrappers.
- Validation completed:
  - wrapper-name `rg` check over `include`, `lib`, `test`, and relevant spec
    files found no deleted compatibility wrapper names.
  - `git diff --check`
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-remove-i32-vadd-runtime-abi-compat-wrappers`
  - `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-runtime-abi-callable-plan-test -j2`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
  - `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test -j2`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
    207/207 passed.
