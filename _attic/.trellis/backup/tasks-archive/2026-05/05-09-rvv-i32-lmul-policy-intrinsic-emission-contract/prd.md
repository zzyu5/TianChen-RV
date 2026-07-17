# RVV i32 LMUL/policy intrinsic emission contract

## Goal

Make the bounded RVV i32 add/sub/mul microkernel C source/header/object exporter
derive its intrinsic vector type, setvl intrinsic, load intrinsic, arithmetic
intrinsic, store intrinsic, and policy handling from the structured
`tcrv_rvv.setvl` / `tcrv_rvv.with_vl` / i32 dataflow body instead of silently
using the current fixed e32/m1 helper surface. Preserve the existing e32/m1
add/sub/mul behavior, fail closed for stale descriptor/body mismatches, and add
one bounded compiler-visible shape only if it fits the same coherent emitter
submodule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD before this task is `397f2b6 feat(rvv): add direct helper ssh evidence handoff`.
- The worktree was clean and `.trellis/.current-task` was absent before this
  task was created.
- The previous task `rvv-i32-family-direct-helper-ssh-evidence-handoff` is
  finished and archived under
  `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-helper-ssh-evidence-handoff/`;
  it must not be reopened.
- The latest archived route-helper work already aligned add/sub/mul direct
  source/header/object helper route ids and collected real `ssh rvv` evidence
  for direct helper handoff.
- The next bottleneck is the RVV target-owned emitter: route helpers and
  evidence exist, but generated C still risks relying on fixed e32/m1 helper
  assumptions instead of validating the typed RVV body metadata.
- The relevant specs require compiler behavior to stay in C++/MLIR/LLVM/
  TableGen/CMake/lit/FileCheck, keep `tcrv.exec` compute-free, and keep RVV
  emission behavior target/plugin-owned.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Do not implement compiler core, dialects, lowering, capability, or emission
  truth in Python.
- Preserve `tcrv.exec` as compute-free. RVV-specific behavior must stay under
  `tcrv_rvv` dialect, RVV plugin, or RVV target/export code.
- The RVV i32 microkernel exporter must validate SEW, LMUL, and finite
  tail/mask policy metadata from typed RVV IR before emitting source, header,
  object, or bundle artifacts.
- Descriptor-local i32 family metadata must agree with the typed RVV body:
  selected family, microkernel op family, dataflow arithmetic op, SEW, LMUL,
  and policy must match before artifact bytes are emitted.
- Existing e32/m1 add/sub/mul source/header/object/direct helper behavior must
  remain stable.
- Unsupported or missing SEW/LMUL/policy metadata must fail closed with bounded
  diagnostics naming the route, selected family, and invalid metadata category.
- If bounded e32/m2 support is small enough for one coherent round, add it for
  the existing i32 add/sub/mul family using matching RVV C intrinsic suffixes
  and vector types. If it proves too large, finish the body-driven m1 refactor
  and record m2 as the exact continuation.
- Do not add new dtypes beyond i32.
- Do not broaden into generic MLIR vector lowering, generic RVV lowering, broad
  runtime ABI changes, performance claims, or standalone evidence packaging.

## Acceptance Criteria

- Source emission derives vector type spelling, vsetvl intrinsic, load
  intrinsic, family arithmetic intrinsic, and store intrinsic from validated
  body metadata instead of fixed m1 constants.
- e32/m1 add/sub/mul outputs still contain the existing `i32m1` RVV intrinsic
  family and ABI behavior.
- Malformed or missing `tcrv_rvv.setvl` SEW, LMUL, or policy metadata fails
  before source/header/object output.
- Mismatched `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` SEW/LMUL/policy metadata
  fails before artifact output.
- Descriptor/body stale-family mismatches fail before artifact output and name
  the route plus selected family.
- Focused lit/FileCheck coverage exists for positive add/sub/mul emission and
  the new SEW/LMUL/policy diagnostics.
- Relevant C++ tests continue to cover target artifact route registration and
  i32 binary family registry behavior.
- `git diff --check` passes.
- Relevant build targets pass:
  `tcrv-translate`, `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-i32-binary-family-registry-test`.
- Focused changed tests pass, and `check-tianchenrv` is run before finishing if
  the local build state supports it.
- No RVV runtime/correctness claim is made unless new real `ssh rvv`
  compile/run evidence is collected.

## Out of Scope

- New dtypes beyond i32.
- Generic MLIR vector lowering or LLVM/RISC-V lowering.
- New high-level tensor/tile compute semantics.
- New core `tcrv.exec` compute operations or RVV branches in core orchestration
  passes.
- Performance benchmarks, broad runtime claims, or evidence packaging as the
  main result.
- Runtime ABI role changes unless the emitter refactor proves an existing role
  is wrong; any such change must be documented and tested separately.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/validation/index.md`
- Relevant archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-microkernel-route-helpers/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-helper-ssh-evidence-handoff/prd.md`
- Likely implementation surfaces:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/EmissionManifest.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `test/Dialect/RVV/`
  - `test/Target/RVVMicrokernel/`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`

## Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit tests under `test/Dialect/RVV` and `test/Target/RVVMicrokernel`
  covering source/header/object emission and SEW/LMUL/policy diagnostics.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  before finish/archive if supported by the local build state.

## Completion Evidence

- Completed submodule: body-driven e32/m1 RVV i32 microkernel intrinsic
  emission contract.
- Source emission now derives the RVV vector C type, intrinsic suffixes, setvl
  suffix, load intrinsic, arithmetic intrinsic, store intrinsic, and emitted
  policy comment from validated `tcrv_rvv.setvl` / `tcrv_rvv.with_vl`
  metadata captured in `RVVIntrinsicConfig`.
- Existing e32/m1 add/sub/mul source/header/object behavior is preserved.
- Descriptor/body stale-family validation remains target-owned and now the
  typed intrinsic metadata diagnostic names the active route and selected
  microkernel family for policy/config mismatches.
- `m2` was not implemented in this round. The current concrete dialect and
  plugin materializer still expose only `!tcrv_rvv.i32m1`, m1 verifiers, m1
  first-slice capability ids, and m1 plugin materialization. The exact
  continuation for e32/m2 is to add a finite `i32m2` dialect type plus bounded
  verifier support, capability ids/proposal legality, plugin materialization,
  C intrinsic suffix generation, source/header/object tests, and real
  `ssh rvv` compile/run evidence before making any runtime/correctness claim.
- Spec update judgment: no `.trellis/spec/` edit was needed because the
  existing RVV plugin, emission-runtime, testing, and validation specs already
  require target-owned typed `setvl` / `with_vl` body validation, bounded i32
  RVV C intrinsic export, fail-closed diagnostics, and no runtime claim without
  real `ssh rvv` evidence.
- Validation run:
  - `git diff --check`
  - `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
  - Focused lit filter `rvv-microkernel-(auto-materialization|family-sub|family-mul|control-body-policy-mismatch-fails)` passed 6/6.
  - Focused lit filter `(Dialect/RVV|Target/RVVMicrokernel)` passed 29/29.
  - `python3 ./.trellis/scripts/task.py validate 05-09-rvv-i32-lmul-policy-intrinsic-emission-contract`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 164/164.
  - No new `ssh rvv` run was collected; this task makes no new RVV
    runtime/correctness/performance claim.
