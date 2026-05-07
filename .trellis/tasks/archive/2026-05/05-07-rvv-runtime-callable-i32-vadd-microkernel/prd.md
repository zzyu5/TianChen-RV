# RVV Runtime-Callable I32 VAdd Microkernel ABI

## Goal

Make the existing bounded RVV i32 vector-add microkernel C export expose a
stable runtime-callable C ABI function backed by RVV intrinsics, while keeping
the existing self-check `main` only as a harness that calls that ABI.

## Scope

- Modify RVV target-owned microkernel emission, primarily
  `lib/Target/RVV/RVVMicrokernel.cpp`.
- Keep the deterministic generated symbol policy derived from the selected
  kernel and variant.
- Generate a callable ABI shaped as:
  `void <generated_name>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)`.
- Implement the callable function body with RVV intrinsic-bearing code:
  `__riscv_vsetvl_e32m1`, `__riscv_vle32_v_i32m1`,
  `__riscv_vadd_vv_i32m1`, and `__riscv_vse32_v_i32m1`.
- Preserve the self-check `main`, but make it initialize bounded arrays, call
  the generated callable ABI function, and verify output.
- Update RVV emission-plan/runtime-ABI metadata and focused tests only where
  the callable ABI changes the expected handoff text.
- Preserve automatic `tcrv_rvv.i32_vadd_microkernel` materialization from the
  finite selected descriptor; positive auto-materialization tests must not need
  a hand-authored microkernel input op.

## Non-Goals

- No generic RVV lowering, arbitrary RVV op lowering, object generation,
  linking, benchmark harness, or performance pipeline.
- No Python compiler internals.
- No RVV-specific branches in generic transforms, target-neutral coherence,
  `tcrv-translate` routing, or generic target artifact export code.
- No changes to scalar fallback, offload descriptor, IME, Sophgo, AME, or
  future plugin behavior except preserving existing tests.
- No RVV hardware compile/run/correctness/performance claim unless existing
  `ssh rvv` evidence path is explicitly run.

## Acceptance

- Focused FileCheck coverage verifies the generated source contains the
  runtime-callable ABI signature, RVV intrinsic loop, and a self-check harness
  that calls the ABI function.
- Existing RVV plugin, RVV dialect, lowering-boundary, execution planning,
  emission manifest, target artifact export, and `tcrv-translate` tests pass.
- Required checks run:
  `git diff --check`,
  CMake configure into `artifacts/tmp/tianchenrv-build`, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
