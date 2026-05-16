# Orphan legacy lit executable erasure

## Goal

Delete orphan lit executable references for the removed i32 binary-family
registry path and removed RVV lowering-boundary compatibility-wrapper test
path, so `check-tianchenrv` cannot pass by invoking stale binaries left in an
old build directory.

## Context

- Current HEAD is `b66cc28 chore(rvv): erase derived lane capability` with a
  clean worktree before this task.
- `test/lit.cfg.py` still registers substitutions for
  `tianchenrv-i32-binary-family-registry-test` and
  `tianchenrv-rvv-lowering-boundary-test`.
- `test/Target/I32BinaryFamilyRegistry/registry.test` still runs the deleted
  i32 binary-family registry executable.
- `test/Transforms/LoweringBoundary/rvv-lowering-boundary.test` still runs the
  deleted standalone RVV lowering-boundary compatibility executable.
- The live build directory may contain stale copies of those executables, so
  the test suite can appear green without active source ownership.

## Scope

### In Scope

- Delete lit `.test` files whose only behavior is invoking
  `tianchenrv-i32-binary-family-registry-test` or
  `tianchenrv-rvv-lowering-boundary-test`.
- Remove the corresponding lit substitutions from `test/lit.cfg.py`.
- Remove directly related active comments/docs only if they specifically
  preserve the deleted executable tests or deleted binary-family /
  compatibility-wrapper behavior.
- Keep current target-neutral lowering-boundary materialization and validation
  tests.
- Keep active RVV extension-family dialect tests unless they directly depend on
  the stale executable names.

### Out of Scope

- No recreation of either legacy executable.
- No compatibility wrapper, replacement descriptor/binary-family tests, or
  replacement direct-C/source route.
- No new RVV lowering, config, EmitC, source, or executable plugin work.
- No deletion of active generic lowering-boundary interfaces or active
  target-neutral selected-boundary validation.

## Acceptance Criteria

- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer mention
      `tianchenrv-i32-binary-family-registry-test`.
- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer mention `I32BinaryFamilyRegistry`.
- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer mention
      `i32 binary family registration registry test passed`.
- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer mention
      `tianchenrv-rvv-lowering-boundary-test`.
- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer mention `RVVLoweringBoundaryTest`.
- [ ] Active source, test, CMake, lit, and spec surfaces outside archived
      Trellis task records no longer include the standalone
      `rvv-lowering-boundary.test` compatibility executable test.
- [ ] Affected lit directories run without depending on stale `build/bin`
      copies of deleted executables.
- [ ] If deletion exposes a missing new-architecture check, it is reported as a
      rebuild gap instead of restoring the old executable or adding a wrapper.

## Minimal Validation

- Focused active-surface ref-scans for:
  - `tianchenrv-i32-binary-family-registry-test`
  - `I32BinaryFamilyRegistry`
  - `i32 binary family registration`
  - `tianchenrv-rvv-lowering-boundary-test`
  - `RVVLoweringBoundaryTest`
  - `rvv-lowering-boundary.test`
- Run affected lit directories after deletion.
- Run `cmake --build build --target check-tianchenrv` if the build remains
  coherent.
- Run `python3 ./.trellis/scripts/task.py validate` for this task.
- Run `git diff --check`.

## Technical Notes

- This is a deletion-only Wrong Logic Deletion Campaign round.
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- Directly inspected files:
  - `test/lit.cfg.py`
  - `test/Target/I32BinaryFamilyRegistry/registry.test`
  - `test/Transforms/LoweringBoundary/rvv-lowering-boundary.test`
  - `test/Transforms/LoweringBoundary/`
