# RVV i32 family direct helper SSH evidence handoff

## Goal

Extend the bounded RVV microkernel evidence workflow so non-add i32 family direct helper routes are intentionally exercised from real compiler metadata through generated source/header/object artifacts into focused `ssh rvv` evidence. This round switches from route-helper implementation to evidence handoff for already registered RVV direct helper routes.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is `2e10006 feat(rvv): add family direct microkernel route helpers`.
- The worktree was clean before this task was created.
- `.trellis/.current-task` was absent; the previous task `rvv-i32-family-direct-microkernel-route-helpers` is archived under `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-microkernel-route-helpers/` and must not be reopened.
- The previous task added direct `tcrv-translate` helpers for vsub/vmul source/header/object route ids and stale-family fail-closed coverage.
- `scripts/rvv_microkernel_e2e.py` already has bounded family specs for `i32-vadd`, `i32-vsub`, and `i32-vmul`, manifest stale-family rejection, source validation, bundle-mode external caller construction, and optional `ssh rvv` execution.
- The current non-bundle direct evidence path still exports source through the legacy generic direct helper and exports header/object through generic target artifact front doors; it does not yet prove the family-specific direct source/header/object helper route handoff.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
- Use Python only as evidence runner/orchestrator, artifact parser, and remote command driver.
- Do not add new arithmetic families beyond bounded `i32-vadd`, `i32-vsub`, and `i32-vmul`.
- Do not move compiler route selection, lowering, emission, capability decisions, or runtime ABI truth into Python.
- For non-add direct helper evidence, select the source/header/object route ids from the existing compiler-facing family route metadata already mirrored by `tcrv-translate` helper registration.
- Fail closed when selected manifest/source/header/object metadata belongs to a stale arithmetic family.
- Persist sanitized evidence identifying arithmetic family, direct helper route ids, source/header/object artifacts when produced, hashes, command summaries, exact `ssh rvv` commands, and bounded claim scope.
- Record no runtime/correctness/pass claim unless real `ssh rvv` compile/link/run evidence exists.
- At minimum, run real `ssh rvv` evidence for `i32-vsub` direct helper mode. If `i32-vmul` is bounded and reachable in the same module, run it too; otherwise record the exact continuation point.

## Acceptance Criteria

- Direct non-bundle evidence mode uses `tcrv-export-rvv-i32-vsub-microkernel-c`, `tcrv-export-rvv-i32-vsub-microkernel-header`, and `tcrv-export-rvv-i32-vsub-microkernel-object` for `i32-vsub`.
- Direct non-bundle evidence mode uses the corresponding `i32-vmul` direct source/header/object helpers when `--arithmetic-family=i32-vmul`.
- Evidence JSON records direct helper route ids and produced artifact paths/hashes without raw logs, URLs, credentials, runtime success strings, throughput, latency, or performance claims.
- Dry-run coverage for `i32-vsub` and `i32-vmul` proves source direct helper selection and stale-family rejection without contacting `ssh rvv`.
- Focused lit coverage continues to cover the existing RVV microkernel runner and, where local RVV object clang is available, bundle/object helper behavior.
- A real `ssh rvv` command for `i32-vsub` direct helper mode compiles/links/runs the generated external caller against generated direct helper artifacts and saves artifacts under `artifacts/tmp/...`.
- If feasible, the same real `ssh rvv` proof is collected for `i32-vmul`; otherwise this task remains open or records a truthful continuation point.
- `git diff --check`, Python compile/self-test, focused dry-runs, focused lit, tool build, and feasible `check-tianchenrv` validation pass before finish/archive.

## Out of Scope

- New RVV arithmetic families, generic RVV lowering, MLIR vector lowering, LLVM/RISC-V intrinsic lowering, or broad runtime integration.
- Performance claims, broad correctness claims, benchmarks, or test matrices beyond the changed direct helper evidence path.
- README/spec-only work as the main result.
- RVV+scalar dispatch bundle expansion unless needed to keep the direct helper evidence path honest.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/validation/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- Prior task context read:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-microkernel-route-helpers/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-microkernel-route-helpers/implement.jsonl`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-family-direct-microkernel-route-helpers/check.jsonl`
- Likely implementation surfaces:
  - `scripts/rvv_microkernel_e2e.py`
  - `test/Scripts/rvv-microkernel-e2e.test`
  - `test/Scripts/rvv-microkernel-bundle-e2e.test` if bundle evidence output changes.
- Existing compiler route surfaces are expected to remain source of truth:
  - `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-family-sub-object.test`
  - `test/Target/RVVMicrokernel/rvv-microkernel-family-mul-object.test`

## Completion Evidence

- `i32-vsub` direct helper ssh evidence:
  - `artifacts/tmp/rvv_microkernel_e2e/20260509-rvv-i32-vsub-direct-helper-ssh/evidence.json`
  - Source route: `tcrv-export-rvv-i32-vsub-microkernel-c`
  - Header route: `tcrv-export-rvv-i32-vsub-microkernel-header`
  - Object route: `tcrv-export-rvv-i32-vsub-microkernel-object`
  - Remote compile/run marker: `tcrv_rvv_i32_vsub_microkernel_external_abi_ok`
- `i32-vmul` direct helper ssh evidence:
  - `artifacts/tmp/rvv_microkernel_e2e/20260509-rvv-i32-vmul-direct-helper-ssh/evidence.json`
  - Source route: `tcrv-export-rvv-i32-vmul-microkernel-c`
  - Header route: `tcrv-export-rvv-i32-vmul-microkernel-header`
  - Object route: `tcrv-export-rvv-i32-vmul-microkernel-object`
  - Remote compile/run marker: `tcrv_rvv_i32_vmul_microkernel_external_abi_ok`
- Spec update judgment: no `.trellis/spec/` edit was needed because
  `.trellis/spec/testing/mlir-testing-contract.md` already contains the
  `RVV Direct I32 Binary Microkernel Evidence Bridge` contract covering this
  runner behavior.
- Final validation:
  - `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
  - `python3 scripts/rvv_microkernel_e2e.py --self-test`
  - Focused vsub/vmul dry-runs with stale-family rejection through lit.
  - Focused lit filter `rvv-microkernel-e2e|rvv-microkernel-bundle-e2e|rvv-microkernel-family-(sub|mul)` passed 6/6.
  - `git diff --check`
  - `python3 ./.trellis/scripts/task.py validate 05-09-rvv-i32-family-direct-helper-ssh-evidence-handoff`
  - `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 164/164.
