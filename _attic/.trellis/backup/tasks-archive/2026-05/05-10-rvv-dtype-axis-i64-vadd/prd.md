# RVV dtype-axis intrinsic descriptor with first i64 vadd artifact path

## Goal

Generalize the RVV target/plugin binary intrinsic descriptor path from the
current finite i32 add/sub/mul demo into an explicit dtype/SEW-owned axis, then
prove one bounded non-i32 route: i64 vector add, preferably `i64m1`, from RVV
plugin proposal/materialization through plugin-local lowering-boundary metadata
and target-owned generated C source artifact.

## What I Already Know

- The previous `05-10-rvv-vector-shape-selection` task is complete and archived
  at commit `65bd441`; this task must not reopen it or produce another
  i32-only shape/evidence round.
- The archived descriptor-registry PRD records that the current
  `RVVI32BinaryIntrinsicDescriptor` composes i32 add/sub/mul family metadata
  with `RVVVectorShape` i32m1/i32m2 shape metadata.
- The current architecture rules require compute semantics to stay out of
  `tcrv.exec`; RVV compute/lowering surfaces belong under `tcrv_rvv` and
  target/RVV code.
- Generic planning, selection, emission readiness, and target artifact routing
  must remain target-neutral. Dtype/family/shape decisions stay RVV
  plugin/target-local.
- Runtime, correctness, or performance claims require real `ssh rvv` evidence.
  This round may stop at compiler/source artifact evidence if no bounded remote
  run is performed.

## Boundaries

- Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
- Python may only orchestrate validation or parse artifacts; it must not become
  compiler descriptor, dialect, plugin, lowering, or emission logic.
- Do not add generic high-level dtype expansion, broad dtype matrices, i64
  sub/mul, i16/i8/floating point, new performance tuning, or new `tcrv.exec`
  compute semantics.
- Do not add RVV-specific branches to core/generic orchestration.
- Do not commit `artifacts/tmp`, build outputs, raw ssh logs, secrets, or
  transient generated files.

## Requirements

1. Introduce or refactor a target/RVV-owned binary intrinsic descriptor API so
   dtype/SEW is structured data rather than an i32-specific name baked into
   each route. Preserve existing i32 add/sub/mul and i32m1/i32m2 behavior
   through compatibility helpers if useful.
2. Seed exactly one non-i32 dtype path: i64 vadd, preferably shape `i64m1`.
   The descriptor must derive vector type, intrinsic suffix, `vsetvl` suffix,
   load/store/arithmetic intrinsic names, selected shape capability ids,
   route/component identity, callable ABI name, runtime ABI C parameter types,
   and target artifact marker text from structured dtype/family/shape data.
3. Add the minimal RVV extension-dialect ODS/C++ surface needed for the i64
   vadd microkernel path. Prefer a small generic typed binary RVV microkernel
   operation if practical; otherwise add the smallest i64-specific surface and
   document the intended dtype-axis follow-up here.
4. Make RVV plugin proposal, materialization, legality, selected lowering
   boundary, and emission-readiness recognize the i64 vadd descriptor only
   when target capabilities carry the required i64/SEW64/LMUL/policy facts.
   Missing facts must fail closed with a bounded diagnostic or non-selection,
   not silently reuse i32 capability facts.
5. Make target artifact export consume the i64 descriptor for generated C
   source. The generated source must use `<riscv_vector.h>` intrinsics such as
   `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`,
   `__riscv_vadd_vv_i64m1`, and `__riscv_vse64_v_i64m1`, with `int64_t`
   ABI buffers and no stale i32/vsub/vmul identity.
6. Add focused tests for descriptor composition, plugin proposal/legality,
   selected lowering-boundary metadata, emission-readiness, generated i64
   source text, and a negative missing-i64-capability case.
7. Preserve existing i32 add/sub/mul, i32m1/i32m2, scalar fallback, and
   RVV+scalar dispatch behavior.

## Acceptance Criteria

- [x] A target/RVV descriptor API has an explicit dtype/SEW axis and still
      derives all existing i32 add/sub/mul x i32m1/i32m2 names and metadata.
- [x] A bounded i64 vadd descriptor derives `i64m1` vector type, suffixes,
      intrinsic names, capability ids, route/component identity, callable ABI,
      and C ABI parameter types from structured descriptor data.
- [x] RVV plugin proposal/materialization selects the i64 vadd path only when
      `rvv`, i64m1 SEW/LMUL/tail/mask policy, toolchain/probe, and other
      existing required RVV evidence facts are available.
- [x] Missing i64 config facts fail closed while available i32 facts alone do
      not select or emit the i64 path.
- [x] Planned IR / lowering-boundary / emission-plan metadata for i64 vadd is
      internally coherent and target-owned.
- [x] Generated RVV C source for i64 vadd contains `int64_t` ABI buffers and
      the i64m1 RVV intrinsics, with FileCheck coverage rejecting stale
      i32/vsub/vmul identity.
- [x] Existing focused i32 descriptor, shape-selection, export, and dispatch
      regression tests still pass.

## Completion Notes

- Implemented a target/RVV `RVVBinaryDescriptor` surface and kept existing
  i32 descriptor/export behavior compatible.
- Added the first non-i32 path: descriptor-backed `i64-vadd` / `i64m1`
  proposal, legality, lowering-boundary, microkernel op, emission plan, and
  generated RVV C source export.
- Added focused descriptor/plugin C++ coverage, positive/negative i64
  microkernel lit coverage, and updated affected registry/setvl regression
  tests.
- Validation completed with focused commands and full `check-tianchenrv`.
- No `ssh rvv` runtime/correctness/performance evidence was run or claimed;
  this round stops at compiler/source-artifact evidence.

## Minimal Validation

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused C++ descriptor/plugin tests for dtype-aware RVV binary descriptors
  and i64 capability selection.
- Focused lit/FileCheck tests for i64 vadd selected boundary/lowering-boundary
  materialization and generated source artifact.
- Focused regression tests for existing i32 descriptor/shape/export behavior,
  including at least one i32m2 path.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks.
- If this round claims i64 runtime/correctness, run one bounded real
  `ssh rvv` compile/run evidence command and record the untracked artifact
  directory. If not run, explicitly state source-artifact evidence only.

## Out Of Scope

- No broad dtype matrix.
- No i64 sub/mul, i16/i8/floating point, masks beyond existing policy
  metadata, or multiple LMULs unless required by the descriptor abstraction and
  kept test-light.
- No generic linalg dtype expansion unless it is the smallest safe way to feed
  the i64 path.
- No performance or benchmark claim.
- No Python compiler implementation.
- No report-only, smoke-only, fixture-only, or evidence-only closeout.

## Technical Notes

- Required specs are listed in `implement.jsonl` and `check.jsonl`.
- Historical context:
  - `.trellis/tasks/archive/2026-05/05-10-05-10-rvv-vector-shape-selection/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32-binary-descriptor-registry/prd.md`
- Primary source surfaces to inspect:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - focused tests under `test/Plugin/`, `test/Transforms/EmissionReadiness/`,
    `test/Target/RVVMicrokernel/`,
    `test/Target/TargetArtifactBundleExport/`, and
    `test/Scripts/rvv-microkernel-e2e.test`
