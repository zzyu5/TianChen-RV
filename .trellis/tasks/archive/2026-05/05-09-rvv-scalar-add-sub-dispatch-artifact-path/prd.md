# RVV/scalar add-sub dispatch artifact path

## Goal

Complete the bounded RVV-primary plus scalar-fallback dispatch artifact path for
i32 add/sub. A frontend-lowered i32-sub module must be able to carry matching
RVV vsub and scalar vsub fallback variants through selected dispatch,
emission-plan, target artifact, and bundle export without stale vadd route,
runtime ABI, glue role, component metadata, function naming, or arithmetic.
Existing i32-add dispatch behavior must remain compatible.

## Scope

- Extend the existing target-owned RVV/scalar dispatch artifact surface to be
  i32 add/sub family-aware.
- Preserve the existing vadd source/header/object/bundle routes and direct
  user-facing behavior.
- Add vsub dispatch source/header/object/bundle routes only through the same
  RVV/scalar dispatch target exporter surface.
- Pair RVV vsub only with scalar vsub fallback and fail closed on mixed
  add/sub candidate families before emitting dispatch artifacts.
- Reuse the existing runtime ABI support utilities for mem_window,
  runtime_param, callable ABI role binding, and dispatch guard validation.
- Keep `tcrv.exec` orchestration-only; arithmetic remains in `tcrv_rvv`,
  `tcrv_scalar`, and target-owned generated C.

## Non-Goals

- No Python compiler internals.
- No new `tcrv.exec` compute operations.
- No new arithmetic families beyond bounded i32 add/sub.
- No i64/e64, mask, dynamic-shape, StableHLO/TOSA, generic linalg, performance,
  or RVV runtime correctness claims.
- No broad smoke matrix or dashboard/report-only work.
- No RVV runtime/correctness/performance evidence claim unless fresh `ssh rvv`
  evidence is collected separately.

## What Is Already Known

- `LowerLinalgI32VAddToExec.cpp` already accepts marked i32 add/sub linalg
  wrappers and preserves `tcrv_frontend_lowering = "i32-vadd"` or `"i32-vsub"`
  on the generated `tcrv.exec.kernel`.
- `RVVExtensionPlugin.cpp` and `ScalarExtensionPlugin.cpp` already choose
  matching add/sub descriptors from `tcrv_frontend_lowering` and materialize
  `tcrv_rvv.i32_vadd_microkernel` / `tcrv_rvv.i32_vsub_microkernel` and
  `tcrv_scalar.i32_vadd_microkernel` / `tcrv_scalar.i32_vsub_microkernel`.
- `RVVMicrokernel.cpp` and `ScalarMicrokernel.cpp` already export direct
  add/sub runtime-callable C source routes with family-specific ABI metadata and
  arithmetic.
- The current gap is `lib/Target/Builtin/RVVScalarDispatch.cpp`: the composite
  RVV/scalar dispatch exporter is still hard-coded to i32-vadd candidate route
  IDs, runtime ABI identities, component group, function names, diagnostics, and
  self-check arithmetic.

## Acceptance Criteria

- A bounded i32-vsub frontend input can lower through
  `--tcrv-lower-linalg-i32-vadd-to-exec` and
  `--tcrv-execution-planning-pipeline` into a selected dispatch with one RVV
  vsub dispatch case and one scalar vsub dispatch fallback when the target
  capability set requires a runtime RVV availability guard.
- Generic target source artifact export emits the RVV+scalar vsub dispatch
  route, includes `__riscv_vsub_vv_i32m1`, includes scalar subtract semantics
  `out[index] = lhs[index] - rhs[index]`, and does not contain stale vadd route,
  ABI, glue-role, function-name, or addition metadata for the vsub path.
- Generic target artifact bundle export records vsub-specific source/header/
  object component routes, component group, external ABI name, runtime ABI name,
  and ordered ABI parameters for the dispatch external ABI.
- A focused stale-family mismatch case fails closed before dispatch artifact
  output when RVV and scalar callable candidates are not from the same add/sub
  family.
- Existing i32-vadd dispatch source/header/object/bundle behavior remains
  compatible.

## Minimal Validation

- Add or update focused lit/FileCheck tests for vsub dispatch source/bundle
  behavior and one stale-family mismatch rejection if not already covered.
- Run `git diff --check`.
- Run CMake configure with the repo LLVM/MLIR paths.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
- Validate the Trellis task before archiving if this round finishes.

## Notes

- Do not widen local artifact checks into RVV runtime/correctness/performance
  claims.
- If object export is skipped by local toolchain feature gating in lit, report
  the exact local condition rather than claiming runtime evidence.

## Completion Summary

- Implemented family-aware RVV/scalar dispatch artifact selection for bounded
  i32 add/sub, including vsub-specific source/header/object composite routes,
  component group, external ABI name, runtime ABI identity, generated symbol
  names, self-check arithmetic, and stale add/sub family mismatch rejection.
- Preserved the existing i32-vadd direct dispatch helper surface while allowing
  generic target artifact and bundle selection to choose the vsub composite
  route from selected RVV vsub plus scalar vsub fallback candidates.
- Added focused vsub dispatch source/header/mismatch and bundle/object-symbol
  lit coverage, and updated target artifact exporter C++ registry coverage for
  the expanded vadd/vsub dispatch composite route set.
- Validation completed locally with `git diff --check`, CMake configure using
  `/usr/lib/llvm-20` LLVM/MLIR paths, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
  No `ssh rvv` runtime correctness or performance claim was made.
