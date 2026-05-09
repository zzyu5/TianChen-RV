# RVV i32m2 vsub dispatch artifact path

## Goal

Make the bounded frontend-generated `i32-vsub` RVV-primary plus scalar-fallback
dispatch evidence path intentionally support the already implemented selected
RVV `i32m2` shape. The task is not to invent new compiler semantics: the
current C++ planner/exporter can already generate an `i32m2` dispatch source
when given an `i32m2` target/profile. The missing module boundary is that the
repo-owned dispatch evidence/front-door surface still assumes `i32m1`, so real
`ssh rvv` evidence for this dispatch path cannot be collected truthfully.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting inspected state is clean at HEAD `eeddc80`.
- There is no active Trellis task, so this task was created explicitly as
  `.trellis/tasks/05-10-rvv-i32m2-vsub-dispatch-artifact-path/`.
- Latest supervisor audit/review input is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0023-20260509T190035Z/`.
  It records that the previous task was completed and archived, and it did not
  provide a new Hermes module brief because review JSON was malformed.
- The prior completed task preserved selected-plan and scalar fallback metadata
  at the RVV+scalar dispatch artifact boundary for `i32-vsub`, but only the
  checked-in positive dispatch fixture is `i32m1`.
- A direct stdin probe of the current compiler showed that changing the
  `i32-vsub` dispatch fixture target/profile from `i32m1` to `i32m2` already
  produces:
  - `rvv_selected_plan_metadata` with `tcrv_rvv.selected_vector_shape = i32m2`;
  - `selected_vector_shape_config: shape=i32m2`;
  - `__riscv_vsetvl_e32m2`;
  - `__riscv_vsub_vv_i32m2`;
  - the same dispatcher ABI with explicit `rvv_available`.
- The analogous direct RVV microkernel path already has an archived
  `--vector-shape=i32m2` evidence mode and real `ssh rvv` evidence. The
  dispatch evidence bridge does not yet have that shape selector.

## Requirements

- Keep compiler truth in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate compiler tools, validate generated
  artifacts, build external callers, and write sanitized evidence.
- Do not add new `tcrv.exec` compute semantics and do not move arithmetic,
  route selection, selected-shape legality, runtime ABI shape, or source
  generation into Python.
- Extend `scripts/rvv_scalar_dispatch_e2e.py` with an explicit bounded
  `--vector-shape` selector for the existing finite `i32m1` and `i32m2`
  selected RVV shapes.
- For `--arithmetic-family=i32-vsub --vector-shape=i32m2`, the dispatch
  evidence bridge must validate compiler-emitted selected-shape metadata before
  recording success:
  - selected shape `i32m2`;
  - SEW 32, LMUL m2, agnostic tail/mask policy;
  - vector type `vint32m2_t`;
  - vector suffix `i32m2`;
  - setvl suffix `e32m2`;
  - selected-shape capability IDs `rvv.i32_m2.*`;
  - RVV intrinsic `__riscv_vsub_vv_i32m2`.
- Add or update checked-in MLIR/lit fixtures so a normal linalg-origin
  `i32-vsub` dispatch path with an `i32m2` target/profile reaches:
  - post-planning dispatch IR;
  - RVV callable source;
  - scalar fallback callable source;
  - dispatch library source;
  - dispatch header;
  - dispatch object or target artifact bundle records when local object clang
    is available.
- Preserve the existing `i32m1` add/sub/mul dispatch evidence behavior and
  existing family-specific stale-route rejection.
- If `ssh rvv` is reachable, collect one real bounded dispatch evidence run for
  the `i32-vsub i32m2` path through the generated compiler artifacts. If remote
  access fails, leave the task open or report the exact non-secret blocker
  without claiming runtime correctness.

## Acceptance Criteria

- `scripts/rvv_scalar_dispatch_e2e.py --self-test` covers shape validation and
  still passes.
- A local dry-run for
  `--arithmetic-family=i32-vsub --vector-shape=i32m2` records evidence with
  `rvv_config.shape = "i32m2"` or equivalent selected-shape evidence, selected
  compile flags from compiler output, `ssh_evidence = false`, and no runtime or
  performance claim.
- A negative local dry-run with `--vector-shape=i32m2` over an `i32m1` input
  fails before success evidence is accepted.
- Focused lit coverage proves the checked-in `i32m2` dispatch fixture emits
  `__riscv_vsub_vv_i32m2`, selected-shape comments, subtract scalar fallback
  semantics, dispatcher ABI, and no hidden `main` in the library artifact.
- Bundle-mode dry-run for the `i32m2` dispatch path validates the compiler
  bundle index, source/header/object records, runtime ABI parameter signature,
  selected dispatch component roles, generated external caller, and selected
  `i32m2` shape metadata.
- If remote evidence is collected, evidence JSON records both source-built and
  bundle-object external caller runs with the bounded success marker observed
  for scalar fallback and RVV guard cases.
- `git diff --check`, script self-test, focused lit, and a focused or full
  project check pass before archiving.

## Out of Scope

- New arithmetic families, dtypes, LMULs beyond the already implemented finite
  `i32m1` / `i32m2` slice.
- New compiler lowering or source-emission semantics if current C++ behavior
  already supports the selected shape.
- Performance measurement.
- Broad runtime matrices.
- Treating local dry-run, lit, or Python self-test as RVV runtime/correctness
  evidence.
- Generic RVV lowering correctness claims.

## Technical Notes

- Specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/capability-model/capability-contract.md`
- Prior task references read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-typed-microkernel-ssh-evidence-handoff/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-scalar-vsub-dispatch-ssh-rvv-runtime-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-i32-vmul-rvv-scalar-dispatch-ssh-rvv-evidence/prd.md`
- Primary implementation surfaces:
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `test/Target/RVVScalarDispatch/`
  - `test/Target/TargetArtifactBundleExport/`
- C++ exporter files should remain read-only unless the new checked-in fixture
  exposes a real selected-shape or dispatch export bug.

## Validation Plan

- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Local direct dry-run for `i32-vsub i32m2`.
- Negative local dry-run proving `i32m2` requested over `i32m1` generated
  source fails closed.
- Local bundle dry-run for `i32-vsub i32m2`.
- Focused lit for `rvv-scalar-dispatch-e2e` and
  `rvv-scalar-dispatch-bundle-e2e`.
- Focused lit for the checked-in `RVVScalarDispatch` and bundle fixtures.
- Real `ssh rvv` run for the `i32-vsub i32m2` dispatch bundle path if remote is
  reachable.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if local build time/state permits.

## Current Status

Completed. The direct `i32-vsub i32m2` dispatch self-check and the
target-artifact bundle external caller both passed real `ssh rvv` validation:

- `artifacts/tmp/rvv_scalar_dispatch_e2e/20260510-rvv-scalar-vsub-i32m2-dispatch-ssh/evidence.json`
- `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/20260510-rvv-scalar-vsub-i32m2-dispatch-bundle-ssh/evidence.json`
