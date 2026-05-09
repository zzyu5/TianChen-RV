# Manifest-driven RVV i64 vadd microkernel ssh evidence

## Goal

Carry the descriptor-backed `i64-vadd` / `i64m1` RVV microkernel from the
current MLIR-generated TianChen-RV artifact path to real `ssh rvv`
compile/link/run evidence. The evidence workflow must consume compiler-emitted
manifest and artifact metadata instead of reimplementing descriptor, ABI,
shape, or family truth in Python.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting inspected state is clean on `main` at HEAD `9fc133c`.
- No current Trellis task existed, so this new task owns the bounded runtime
  evidence round.
- The previous `rvv-dtype-axis-i64-vadd` task is complete and archived at
  commit `9fc133c`; this task must not reopen it or become another
  descriptor-only/source-artifact-only round.
- The previous task added dtype-aware `RVVBinaryDescriptor`,
  `tcrv_rvv.i64_*` microkernel/source export behavior, plugin
  proposal/legality, and generated i64 RVV C source evidence, but explicitly
  did not collect real `ssh rvv` runtime/correctness evidence.
- Existing direct RVV microkernel evidence style is represented by archived
  i32/i32m2 tasks and `scripts/rvv_microkernel_e2e.py`: local dry-runs prove
  tooling and manifest handoff only; runtime/correctness claims require real
  `ssh rvv` compile/link/run artifacts under `artifacts/tmp`.

## Boundaries

- Compiler implementation remains C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may invoke tools, parse compiler-emitted manifests/indexes, generate
  bounded callers from emitted ABI surfaces, run remote compile/link/run, and
  write sanitized evidence JSON.
- Python must not decide compiler lowering, legality, selected descriptor,
  selected shape, intrinsic spelling, runtime ABI, route selection, or
  correctness semantics independently of compiler-emitted artifact metadata.
- `tcrv.exec` stays compute-free; RVV behavior remains RVV dialect/plugin/
  target owned.
- Runtime evidence is bounded only to the generated `i64-vadd` / `i64m1`
  microkernel artifact and tested runtime element counts.
- No performance, throughput, generic RVV lowering, generic linalg lowering,
  full runtime integration, or broad correctness claim is allowed.
- Do not commit `artifacts/tmp`, build outputs, raw ssh logs, private SSH
  material, credentials, tokens, or transient generated files.

## Requirements

1. Start from `test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir` or an
   equivalent bounded MLIR input and use the existing `tcrv-opt` /
   `tcrv-translate` compiler/export route to produce the runtime-callable i64
   RVV C artifact.
2. If the compiler-exported handoff lacks enough metadata for safe evidence
   discovery, add the smallest target/export C++ metadata needed so the runner
   can identify dtype, arithmetic family, selected vector shape, function
   symbol, ABI parameter types/order, route identity, and selected capability
   facts from compiler output.
3. Extend `scripts/rvv_microkernel_e2e.py` as evidence orchestration only. It
   may add a bounded `i64-vadd` / `i64m1` evidence mode, invoke compiler tools,
   parse manifest fields, construct a runner/caller from emitted ABI data, and
   collect local/remote evidence.
4. The i64 path must validate generated C against compiler-emitted identity:
   `int64_t` buffers, `i64-vadd`, `i64m1`, the selected function symbol, and
   RVV intrinsics such as `__riscv_vsetvl_e64m1`,
   `__riscv_vle64_v_i64m1`, `__riscv_vadd_vv_i64m1`, and
   `__riscv_vse64_v_i64m1`.
5. The i64 path must fail closed before a runtime claim on stale i32/vsub/vmul
   route identity, stale i32 ABI types, missing/ambiguous function symbol,
   missing manifest fields, or mismatched shape/dtype/family metadata.
6. Run real `ssh rvv` compile/link/run evidence for `i64-vadd` / `i64m1`.
   Validate at least two runtime `n` counts: one non-multiple/tail case and
   one larger multi-iteration case when the generated callable supports runtime
   `n`.
7. Persist sanitized evidence JSON under `artifacts/tmp` with at least:
   repo commit, MLIR input, artifact paths, selected dtype/family/shape,
   function symbol, selected shape/capability metadata, compiler path, compile
   /link/run exit codes, stdout success marker, tested runtime counts,
   bounded claim scope, and a no-performance-claim marker.
8. Add focused local/lit coverage for the new i64 dry-run/evidence runner path
   and at least one regression proving the old i32 path still works.
9. Preserve existing i32 add/sub/mul, i32m1/i32m2, source/header/object export,
   and previous ssh evidence behavior.

## Acceptance Criteria

- [ ] PRD and Trellis context files identify this bounded runtime-evidence
      task and do not reopen the archived descriptor/source-artifact task.
- [ ] The i64 evidence route consumes compiler-emitted manifest/artifact fields
      for dtype/family/shape/function/ABI discovery; any Python family table is
      limited to runner expectations and rejects stale compiler output.
- [ ] Local generated artifact validation proves i64 ABI buffers, i64m1 RVV
      intrinsics, selected i64 metadata, and absence of stale i32/vsub/vmul
      identity.
- [ ] A focused dry-run writes claim-free evidence for `i64-vadd` / `i64m1`.
- [ ] A focused dry-run or self-test regression proves an existing i32 route
      remains compatible.
- [ ] Real `ssh rvv` compile/link/run succeeds for `i64-vadd` / `i64m1`, with
      at least two runtime element counts including a tail case and a larger
      multi-iteration case when supported.
- [ ] Sanitized evidence JSON records the required bounded fields and contains
      no secrets, credentials, private SSH material, raw unrelated environment,
      or performance claim.
- [ ] Focused lit/script tests for the changed evidence workflow pass.
- [ ] Focused compiler/export checks and `git diff --check` pass.
- [ ] `check-tianchenrv` passes if practical; otherwise the task remains open
      unless focused checks plus real ssh evidence are adequate and the
      unrun scope is explicitly justified.
- [ ] Trellis task validation passes before finish/archive.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused local generation/FileCheck for `i64-vadd` source artifact.
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- New/updated dry-run for `i64-vadd` / `i64m1`.
- Existing i32 dry-run regression.
- Focused lit under `test/Scripts/` for the evidence workflow and any touched
  target-export tests.
- Real `ssh rvv` i64 evidence command with sanitized JSON artifact.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-vadd-ssh-evidence`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks.

## Out Of Scope

- No i64 sub/mul, i16, i8, floating point, additional LMULs, broad dtype
  matrix, or new compiler descriptor expansion beyond consuming the existing
  `i64-vadd` / `i64m1` path.
- No new `tcrv.exec` compute semantics.
- No RVV-specific branches in shared core/generic transforms.
- No hand-written standalone RVV C as the source of hardware evidence.
- No broad smoke matrix or benchmark/performance run.
- No committed generated artifacts, raw ssh logs, secrets, or build outputs.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/validation/experiment-reference.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
- Prior task/context read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-dtype-axis-i64-vadd/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-typed-microkernel-ssh-evidence-handoff/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence/prd.md`
  - `.trellis/workspace/codex/journal-1.md` entries for prior direct
    microkernel and i32m2 ssh evidence conventions.
- Primary source/test surfaces to inspect:
  - `test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `scripts/rvv_remote_probe.py`
  - `test/Scripts/rvv-microkernel-e2e.test`
  - `test/Scripts/rvv-microkernel-bundle-e2e.test`
