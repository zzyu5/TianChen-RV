# Linalg i64 vadd frontend to RVV artifact and ssh evidence

## Goal

Carry one bounded high-level MLIR frontend path, marked `linalg.generic`
`memref<?xi64>` vector add, through TianChen-RV exec planning, RVV plugin
selection, descriptor-backed lowering/emission, generated source/header/object
artifact, and real `ssh rvv` runtime correctness evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting state is clean on `main` at HEAD `1a31347`.
- No current Trellis task existed, so this task owns the bounded
  frontend-to-runtime round.
- The archived `rvv-dtype-axis-i64-vadd` task completed the target/RVV
  descriptor and i64 source-artifact path.
- The archived `rvv-i64-vadd-ssh-evidence` task completed microkernel-fixture
  SSH evidence for `i64-vadd` / `i64m1`.
- This task must not reopen either archive and must not close as another
  descriptor-only, microkernel-fixture-only, runner-only, or report-only round.
- Current frontend lowering is still visibly i32-centered through
  `LowerLinalgI32BinaryToExec`, i32 add/sub/mul markers, and i32 ABI helpers.

## Boundaries

- Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
- Python may only orchestrate tools, parse compiler-emitted artifacts, run
  remote evidence, and write sanitized evidence JSON.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not add i64 sub/mul, i16/i8/fp, arbitrary linalg lowering, generic tensor
  semantics, performance benchmarking, or broad dtype matrices.
- Do not add RVV-specific semantics to generic core transforms when the
  existing RVV descriptor/plugin/exporter is the owner.
- Do not commit `artifacts/tmp`, build outputs, raw ssh logs, credentials,
  secrets, or transient generated files.

## Requirements

1. Generalize the bounded linalg binary frontend pass just enough to accept the
   existing i32 add/sub/mul markers plus exactly one new marker:
   `tcrv_frontend_lowering = "i64-vadd"`.
2. Preserve the public pass spelling
   `--tcrv-lower-linalg-i32-binary-to-exec` and the deprecated
   `--tcrv-lower-linalg-i32-vadd-to-exec` alias unless the implementation
   proves a compatible alias is unsafe.
3. Validate the i64 source fail-closed before materializing a kernel: exactly
   two input buffers and one output buffer, `memref<?xi64>` operands, one
   single-block `linalg.generic`, three i64 scalar region arguments, one
   `arith.addi` of the first two scalar args, and one `linalg.yield` of that
   result.
4. Keep unsupported or mismatched marked linalg bodies fail-closed before any
   `tcrv.exec.kernel` is created.
5. Materialize `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` ABI
   boundaries with i64 C types for the new path: `const int64_t *` lhs/rhs,
   `int64_t *` output, and runtime `size_t n`.
6. Feed the existing RVV plugin/descriptor path by preserving the frontend
   lowering marker and using existing descriptor-owned metadata for i64 ABI
   requirements. Do not rediscover i64 route strings in the frontend pass.
7. Produce a generated frontend source artifact containing `int64_t` buffers,
   `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`,
   `__riscv_vadd_vv_i64m1`, and `__riscv_vse64_v_i64m1`, with no stale
   i32/vsub/vmul identity.
8. Preserve existing i32 add/sub/mul frontend behavior and focused regression
   coverage.
9. Extend the evidence runner only if needed to accept the new frontend MLIR
   input. Runner changes must remain orchestration/artifact-validation only.

## Acceptance Criteria

- [x] PRD and Trellis context identify this bounded frontend-to-runtime task.
- [x] A marked `linalg.generic` `i64-vadd` wrapper lowers to `tcrv.exec.kernel`
      with i64 ABI mem windows/runtime count parameter.
- [x] The lowered i64 exec kernel reaches RVV proposal/selection,
      `tcrv_rvv.lowering_boundary`, `tcrv_rvv.i64_vadd_microkernel`, supported
      emission-plan metadata, and target source export.
- [x] Missing i64m1 capability facts fail closed without materializing the RVV
      i64 microkernel or emission plan.
- [x] At least one existing i32 frontend route still lowers and emits unchanged.
- [x] Focused lit/FileCheck tests cover positive i64 lowering/export, negative
      missing i64m1 facts, and i32 regression.
- [x] Real `ssh rvv` evidence runs on the compiler-generated frontend artifact
      for at least runtime counts `[7, 16]` or equivalent tail plus
      multi-iteration counts.
- [x] Sanitized evidence JSON is written under `artifacts/tmp` and is not
      committed.
- [x] Focused build, focused lit, `git diff --check`, task validation, and
      practical full `check-tianchenrv` pass before archive.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused lit for new and touched `test/Transforms/LinalgToExec/` tests.
- Focused lit for touched RVV microkernel/evidence runner tests if changed.
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py` if the runner changes.
- Real SSH evidence command:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vadd --input <new linalg i64 vadd test input> --run-id <bounded-id> --overwrite --timeout 120 --ssh-target rvv`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-linalg-i64-vadd-frontend-rvv-ssh-evidence`

## Completion Evidence

- Focused build passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused FileCheck coverage passed for:
  - `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`
  - `test/Transforms/LinalgToExec/linalg-i64-vadd-rvv-i64m1-missing-config-fails.mlir`
  - `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
  - `test/Scripts/rvv-microkernel-e2e.test` i64 frontend dry-run/source/evidence prefixes
- Python runner syntax check passed:
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- Real SSH evidence passed:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vadd --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --run-id codex-i64-vadd-frontend-ssh --overwrite --timeout 120 --ssh-target rvv`
- Sanitized evidence JSON:
  `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vadd-frontend-ssh/evidence.json`
- Evidence runtime counts:
  `[7, 16]`
- Full check passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Out Of Scope

- No new RVV dtype/op expansion beyond i64 vadd.
- No performance or throughput claims.
- No generic linalg lowering for arbitrary ops, ranks, tensor semantics, or
  shapes.
- No compute semantics inside `tcrv.exec`.
- No extension-specific behavior in generic core transforms when plugin or
  descriptor interfaces are the correct owner.
- No hand-written standalone RVV C evidence that bypasses the
  compiler-generated frontend artifact.
- No broad smoke matrix, dashboard, report-only, or wrapper-only closeout.

## Technical Notes

- Required specs are listed in `implement.jsonl` and `check.jsonl`.
- Required archive context read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-dtype-axis-i64-vadd/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-ssh-evidence/prd.md`
- Primary source/test surfaces:
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `include/TianChenRV/Transforms/Passes.td`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `test/Transforms/LinalgToExec/`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir`
  - `test/Scripts/rvv-microkernel-e2e.test`
