# Descriptor-backed i32 binary linalg frontend lowering contract

## Goal

Replace the active bounded linalg frontend lowering owner from the stale
vadd-named surface to an i32 binary family contract. The frontend lowering
boundary must publicly and internally describe the finite i32 add/sub/mul
family while preserving the existing explicitly marked `linalg.generic` wrapper
shape and the `tcrv.exec` kernel, target/profile, `mem_window`, `runtime_param`,
and frontend family metadata consumed by the planning pipeline.

## What I already know

- The previous descriptor-backed i32 binary runtime ABI task is archived and
  current HEAD is `b65f930`.
- The repository currently has no active Trellis task and starts this task from
  a clean `main` worktree.
- The existing implementation in the old
  `lib/Transforms/LowerLinalgI32VAddToExec.cpp` owner already accepted the finite
  marked frontend family values `i32-vadd`, `i32-vsub`, and `i32-vmul` and
  checks the matching `arith.addi`, `arith.subi`, or `arith.muli` body.
- The public pass, factory, built-in `tcrv-opt` registration,
  `tcrv-translate` plan-and-export frontend step, Python bridge orchestration,
  README examples, specs, and many lit RUN lines still use
  `LowerLinalgI32VAddToExec` or
  `--tcrv-lower-linalg-i32-vadd-to-exec`.
- The frontend pass must remain a bounded lowering boundary into `tcrv.exec`;
  plugin proposal, legality, selection, selected-boundary materialization,
  emission plans, and target artifact exports remain owned by the existing
  planning/export pipeline.

## Requirements

- Add a family-named pass/factory surface such as
  `createLowerLinalgI32BinaryToExecPass` and public option
  `--tcrv-lower-linalg-i32-binary-to-exec`.
- Refactor active implementation naming, generated pass class naming, source
  file ownership, diagnostics, comments, built-in tool calls, script calls, and
  README examples away from vadd-only ownership.
- Preserve the accepted finite frontend family markers `i32-vadd`, `i32-vsub`,
  and `i32-vmul`.
- Preserve source IR semantics and output IR structure except for public names,
  diagnostics, and comments required by this module.
- The pass must continue to lower only explicitly marked hand-written/test
  `linalg.generic` wrappers with the existing constrained one-arithmetic-op
  shape into the existing `tcrv.exec` ABI boundary.
- Keep the old `--tcrv-lower-linalg-i32-vadd-to-exec` option only as a
  deprecated compatibility alias if needed, implemented as a wrapper around the
  family-named pass.
- Active scripts, new/updated tests, README examples, and plan/export front
  doors touched in this task must use the family-named pass.
- If the compatibility alias remains, add one focused compatibility test and
  make remaining retained references explicitly compatibility-only.
- Update durable specs where the public frontend contract still names the
  vadd-only pass.
- Search for remaining `LowerLinalgI32VAddToExec` and
  `--tcrv-lower-linalg-i32-vadd-to-exec` references and either migrate them or
  record why they are retained.

## Acceptance Criteria

- [x] Public C++ API exposes `createLowerLinalgI32BinaryToExecPass`.
- [x] Public pass option `--tcrv-lower-linalg-i32-binary-to-exec` works for
      add, sub, and mul marked frontend wrappers.
- [x] `tcrv-opt` built-in pass registration and `tcrv-translate`
      plan-and-export frontend lowering use the family-named factory.
- [x] `scripts/rvv_scalar_dispatch_e2e.py --lower-linalg-frontend` uses the
      family-named pass option.
- [x] Direct lowering, execution-planning, target artifact/bundle, and script
      dry-run tests touched by this module exercise the family-named pass for
      add/sub/mul where those paths already exist.
- [x] Any old vadd-named pass option remains only as a deprecated alias and has
      one focused compatibility test.
- [x] Remaining vadd-named frontend lowering references are only
      compatibility/documentation notes or family marker values such as
      `i32-vadd`, not active pass ownership.
- [x] `git diff --check` passes.
- [x] Local CMake configure with the MLIR toolchain under
      `artifacts/tmp/tianchenrv-build` succeeds.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes, or failures are repaired/rerun or documented with the exact next
      continuation point.

## Non-goals

- Do not infer arbitrary linalg semantics or make this a generic linalg
  lowering pass.
- Do not add compute semantics to `tcrv.exec`.
- Do not change descriptor family facts, runtime ABI shapes, dispatch guard
  semantics, plugin proposal/selection rules, emission backends, target export
  routes, or generated arithmetic semantics.
- Do not add new RVV/scalar/offload kernels, new descriptor families, new
  variant selection rules, or new runtime evidence claims.
- Do not implement compiler core, dialects, passes, plugin registry, capability
  model, lowering, or emission in Python.
- Do not claim RVV runtime/correctness/performance without fresh `ssh rvv`
  evidence. This task should not require `ssh rvv`.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task PRD read:
  `.trellis/tasks/archive/2026-05/05-09-descriptor-backed-i32-binary-runtime-abi-contract/prd.md`.
- Primary source surfaces inspected:
  `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/LowerLinalgI32VAddToExec.cpp` before the source owner was
  renamed to `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`,
  `lib/Transforms/CMakeLists.txt`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Transforms/LinalgToExec`,
  `test/Target/RVVScalarDispatch`,
  `test/Target/TargetArtifactBundleExport`,
  `test/Scripts/rvv-scalar-dispatch*.test`,
  and README frontend sections.

## Completion Notes

- Active frontend lowering ownership is now `LowerLinalgI32BinaryToExec`,
  `createLowerLinalgI32BinaryToExecPass`, and
  `--tcrv-lower-linalg-i32-binary-to-exec`.
- The old `LowerLinalgI32VAddToExec` /
  `--tcrv-lower-linalg-i32-vadd-to-exec` surface remains only as a deprecated
  compatibility alias delegating to the same family-named implementation.
- The retained vadd-named frontend-lowering references are:
  `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`, and
  `tools/tcrv-opt/tcrv-opt.cpp` for the compatibility alias;
  `test/Transforms/LinalgToExec/linalg-i32-binary-compat-vadd-alias.mlir` for
  the focused alias test; README/spec/task text documenting deprecation; and
  ordinary `i32-vadd` family marker/descriptor references that are not pass
  ownership.
- Validation:
  `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`,
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (162/162 passed), focused lit from `artifacts/tmp/tianchenrv-build/test`
  over `Transforms/LinalgToExec`, `Target/RVVScalarDispatch`,
  `Target/TargetArtifactBundleExport`, and the two
  `Scripts/rvv-scalar-dispatch*` tests (26/26 passed), and
  `git diff --check`.
