# RVV i32 family intrinsic prefix contract

## Goal

Make the bounded RVV i32 add/sub/mul family registry describe only the
family-level RVV arithmetic intrinsic prefix, while the final emitted intrinsic
name is derived by the RVV target exporter from that family prefix plus the
selected vector-shape suffix (`i32m1` or `i32m2`). This keeps family semantics
and selected compile-time vector shape separate after the selected-shape
cleanup at `e2f8814`.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Session start state was clean at HEAD `e2f8814`.
- No active `.trellis/.current-task` existed; the latest completed task was
  archived as
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-vector-shape-config-boundary-cleanup/`.
- Latest supervisor audit input was
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0021-20260509T174517Z/`.
- Current code already emits selected i32m2 add/sub/mul intrinsics correctly
  when the target profile selects i32m2.
- The remaining boundary issue is structural: `I32BinaryFamilyRegistry.h`
  still stores RVV `intrinsicName` values such as `__riscv_vadd_vv_i32m1`,
  even though `i32m1` is selected-shape metadata, not family identity.
- `lib/Target/RVV/RVVMicrokernel.cpp` currently carries a separate local switch
  for arithmetic intrinsic prefixes, duplicating family knowledge that belongs
  in the i32 family registry.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Do not implement compiler semantics in Python; Python scripts may remain
  evidence consumers only if touched.
- Replace the registry field that currently looks like a full m1 intrinsic name
  with a family-level RVV arithmetic intrinsic prefix.
- Make RVV target source emission derive the final intrinsic spelling from:
  family intrinsic prefix + selected vector-shape suffix.
- Preserve emitted source for existing i32m1 and i32m2 paths.
- Preserve selected vector-shape metadata as the only source of `i32m1`/`i32m2`
  suffix selection.
- Do not add new dtype, LMUL, arithmetic family, performance claim, or broad
  smoke/test matrix.
- Do not choose a test-only or docs-only outcome.

## Acceptance Criteria

- `I32BinaryFamilyRegistry.h` no longer records RVV add/sub/mul intrinsic
  strings with a baked-in `i32m1` suffix.
- RVV source emission uses the registry-owned family prefix plus selected
  vector-shape suffix to print the exact same generated intrinsic names as
  before.
- Existing i32m1 vadd/vsub/vmul routes continue to emit `...i32m1`.
- Existing i32m2 vsub route continues to emit `...i32m2`.
- A focused test verifies family registry intrinsic prefixes are suffix-free and
  unique across add/sub/mul.
- Focused lit/FileCheck coverage verifies emitted source still carries selected
  shape metadata and matching intrinsic spelling.
- `git diff --check` passes.
- Focused C++/lit checks pass.

## Out of Scope

- Adding new shape coverage for add/vmul beyond existing compiler support.
- Collecting new `ssh rvv` runtime evidence unless emitted source semantics
  change in a claim-bearing way.
- Reworking runtime ABI parameter shape, dispatch semantics, or linalg frontend
  lowering.
- Adding broad benchmark, performance, or full matrix validation.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/capability-model/capability-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Primary code surfaces:
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - focused RVV microkernel/linalg source-export lit fixtures

## Validation Plan

- Build focused target:
  `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test -j2`
- Run focused C++ test:
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Run focused lit tests for RVV source emission covering i32m1 and i32m2.
- Run `git diff --check`.

## Current Status

Completed.

## Completion Notes

- Replaced the RVV family descriptor's full selected-shape-looking
  `intrinsicName` with suffix-free `arithmeticIntrinsicPrefix` values for
  add/sub/mul.
- Removed the RVV target emitter's local arithmetic intrinsic prefix switch.
  The generated arithmetic intrinsic now comes from:
  family `arithmeticIntrinsicPrefix` + selected vector-shape suffix.
- Preserved generated source spelling for existing i32m1 paths and the i32m2
  vsub selected-shape path.
- Added C++ registry assertions that RVV arithmetic intrinsic prefixes are
  pairwise distinct, use the RVV intrinsic namespace, end at the suffix
  boundary, and do not bake in `i32m1` or `i32m2`.
- Updated RVV plugin and lowering/runtime specs to state the family-prefix
  versus selected-shape-suffix contract.

## Validation Results

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir
  Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir
  Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir
  Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir`
  from `artifacts/tmp/tianchenrv-build/test` (4/4 passed)
- [OK] `python3 ./.trellis/scripts/task.py validate
  05-10-rvv-i32-family-intrinsic-prefix-contract`
- No new `ssh rvv` run was collected; this round changed compiler ownership of
  intrinsic prefix composition without changing emitted source semantics or
  making a new runtime/correctness/performance claim.
