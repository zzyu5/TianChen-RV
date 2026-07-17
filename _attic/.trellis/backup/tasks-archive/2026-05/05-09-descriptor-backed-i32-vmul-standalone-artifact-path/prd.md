# Descriptor-backed i32-vmul standalone artifact path

## Goal

Add the first new descriptor-backed finite i32 binary arithmetic family,
`i32-vmul`, through the active local compiler/plugin/export path for standalone
RVV and scalar microkernel artifacts. This proves
`I32BinaryFamilyRegistry` is a bounded extension interface rather than only an
add/sub refactor.

## Why Now

The previous plugin descriptor-consumption module archived cleanly at commit
`5c109b9`. Add/sub are now descriptor-backed across target/export and plugin
proposal/materialization surfaces. The next bottleneck is to add one bounded
new family end-to-end without turning `tcrv.exec` into a compute dialect or
expanding into dispatch/runtime evidence work.

## Requirements

- Extend `I32BinaryFamilyRegistry` with stable `i32-vmul` family facts distinct
  from add/sub:
  - family id and frontend lowering identity;
  - finite lowering descriptor;
  - RVV/scalar microkernel op identities;
  - RVV arithmetic op and intrinsic identity;
  - scalar C operator;
  - route ids, emission kinds, runtime ABI fields, runtime glue roles, function
    stems, header guard stems, result variable names, and mismatch identity.
- Extend the RVV extension dialect with a finite plugin-local
  `tcrv_rvv.i32_mul` dataflow op and `tcrv_rvv.i32_vmul_microkernel` op, or an
  equivalent bounded family op required by the existing typed RVV body/export
  path.
- Extend the scalar extension dialect with the matching finite plugin-local
  scalar `i32_vmul_microkernel` op needed by scalar descriptor
  materialization/export.
- Extend bounded linalg frontend lowering so a marked `linalg.generic` i32
  multiply body using `arith.muli` lowers to an execution-planning kernel with
  `tcrv_frontend_lowering = "i32-vmul"`.
- Extend RVV plugin proposal, materialization, readiness, and emission-plan
  logic so `i32-vmul` is selected through registry-backed descriptor fields
  while preserving RVV plugin-local capability, policy, required march,
  capacity-derived element-count, and diagnostics.
- Extend scalar plugin proposal, materialization, readiness, and emission-plan
  logic so `i32-vmul` is selected through registry-backed descriptor fields
  while preserving scalar fallback role, policy, and diagnostics.
- Extend RVV standalone microkernel export so `i32-vmul` emits multiply
  semantics using the correct RVV intrinsic or a clearly bounded RVV intrinsic
  path, and never reuses add/sub operation identity.
- Extend scalar standalone microkernel export so `i32-vmul` emits
  `out[index] = lhs[index] * rhs[index]`, and never reuses add/sub arithmetic
  semantics.
- Add focused MLIR/lit/C++ coverage proving registry membership, frontend
  lowering, RVV/scalar plugin proposal/materialization, and RVV/scalar
  standalone artifact emission for `i32-vmul`.

## Acceptance Criteria

- [x] Registry tests prove the finite registry covers exactly `i32-vadd`,
      `i32-vsub`, and `i32-vmul`, with distinct vmul descriptor, route, ABI,
      intrinsic/op identity, scalar operator, and function stems.
- [x] RVV dialect parse/verify coverage proves `tcrv_rvv.i32_mul` is admitted
      only in the matching bounded vmul microkernel body and stale add/sub
      bodies fail closed.
- [x] Scalar dialect parse/verify coverage proves
      `tcrv_scalar.i32_vmul_microkernel` is plugin-local and distinct from
      add/sub.
- [x] Frontend lowering accepts a bounded marked i32 multiply linalg body with
      `arith.muli` and preserves `tcrv_frontend_lowering = "i32-vmul"` on the
      generated `tcrv.exec.kernel`.
- [x] RVV plugin proposal/materialization tests prove `i32-vmul` produces the
      vmul descriptor, RVV microkernel op, RVV arithmetic op, readiness, and
      supported emission plan through registry-backed family facts.
- [x] Scalar plugin proposal/materialization tests prove `i32-vmul` produces
      the vmul descriptor, scalar microkernel op, readiness, and supported
      emission plan through registry-backed family facts.
- [x] RVV target artifact tests prove generated vmul source contains the vmul
      RVV intrinsic/path and not add/sub intrinsic/path.
- [x] Scalar target artifact tests prove generated vmul source contains `*`
      multiply semantics and not add/sub semantics.
- [x] Focused add/sub regression tests touched by the migration remain passing.
- [x] `git diff --check` passes.
- [x] CMake configure with repository LLVM/MLIR paths passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before the task is archived, unless a narrower blocker is recorded
      truthfully.
- [x] If complete, Trellis validation/finish/archive is performed, the journal
      is updated, and one coherent commit is created.

## Completion Notes

- `i32-vmul` is implemented across registry, RVV dialect/body/export,
  scalar dialect/export, bounded linalg frontend lowering, RVV plugin, scalar
  plugin, and focused tests.
- The task intentionally does not add an `i32-vmul` RVV+scalar dispatch bundle
  or remote `ssh rvv` runtime evidence. Those remain later modules after
  standalone artifacts are stable.
- Validation completed with `git diff --check`, CMake configure against
  `/usr/lib/llvm-20`, focused C++ plugin/registry/exporter binaries, focused
  lit filter for vmul plus touched add/sub tests, and full `check-tianchenrv`.

## Scope Boundary

This task completes standalone RVV and scalar artifact support for `i32-vmul`.
It does not add the full RVV+scalar dispatch bundle, host dispatch source/header
/object routes, remote runtime evidence, performance measurements, or any
generic arithmetic in `tcrv.exec`.

Tiny shared route plumbing is allowed only when needed to keep existing
registry/plugin/export APIs coherent and must be focused-tested. Dispatch
consumer behavior for add/sub must be preserved, but dispatch is not the main
deliverable this round.

## Non-goals

- No new `ssh rvv` runtime correctness claim unless a fresh focused remote
  compile/link/run self-check is added and executed.
- No performance benchmarking or ratio claims.
- No i64/e64, masks, widening/narrowing, new RVV policy families,
  dynamic-shape frontend expansion, StableHLO/TOSA lowering, generic RVV
  lowering, or generic scalar lowering.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No Python implementation of compiler registry, dialects, passes, plugin
  proposal, lowering, emission, route selection, runtime ABI decisions, or
  source generation.
- No docs-only, smoke-only, wrapper-only, report-only, helper-only, or
  metadata-only closeout.

## Technical Notes

- Current repo state before this task:
  - repo root: `/home/kingdom/phdworks/TianchenRV`
  - HEAD: `5c109b9 feat(plugin): consume i32 binary family registry`
  - worktree: clean
  - no active Trellis task before this task was created
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Prior module PRDs:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-binary-family-descriptor-registry/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-plugin-i32-binary-family-descriptor-consumption/prd.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Dialect/Scalar/IR/ScalarDialect.cpp`
  - `lib/Transforms/LowerLinalgI32VAddToExec.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `tools/tcrv-opt/tcrv-opt.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - focused tests under `test/Plugin`, `test/Transforms`, `test/Target/RVV`,
    `test/Target/Scalar`, and `test/Target/I32BinaryFamilyRegistry`

## Continuation Rule If Unfinished

Keep this task open. Record which layer is complete and which is not:
registry, RVV ODS/op, scalar ODS/op, frontend lowering, RVV plugin, scalar
plugin, RVV exporter, scalar exporter, tests, task archive, or commit. Do not
archive or claim `i32-vmul` standalone artifact support is complete until both
RVV and scalar standalone artifact paths are implemented and checked.
