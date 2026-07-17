# Bounded linalg i32 binary frontend to selected RVV scalar artifact pipeline

## Goal

Complete one bounded frontend-to-artifact proof slice for the existing
`i32-vsub` + `i32m2` path. The input is a marked hand-written
`linalg.generic` i32 binary subtract wrapper. The accepted route is the C++ /
MLIR owner chain:

```text
tcrv-translate --tcrv-plan-and-export-target-artifact-bundle
  -> --tcrv-lower-linalg-i32-binary-to-exec
  -> tcrv-execution-planning-pipeline
  -> selected RVV dispatch case + scalar fallback
  -> target artifact bundle exporter
```

If the full route is already present, this task strengthens the focused lit
proof and removes the remaining ambiguity that a hand-authored selected dispatch
fixture could be doing the compiler's work.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting inspected state is clean at `61df2eb`.
- The previous selected-shape ownership task is archived and must not be
  reopened.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` already runs
  bounded linalg lowering, execution planning, and target artifact bundle
  export in one tool route.
- Existing `i32-vsub` + `i32m2` RVV runtime evidence remains the prior
  `5db3128` evidence; this task does not make a new runtime/correctness/
  performance claim unless generated runtime-visible behavior changes and is
  run on `ssh rvv`.

## Requirements

- Keep compiler behavior in C++ / MLIR / TableGen / CMake / lit / FileCheck.
- Do not implement lowering, selection, dispatch, ABI, or export semantics in
  Python.
- Do not add new arithmetic families, dtypes, LMULs, RVV ops, scalar ops, or
  generic vector lowering.
- Keep `tcrv.exec` compute-free; RVV and scalar behavior must remain
  plugin/target-local.
- Use the existing `i32-vsub` + `i32m2` descriptor path as the acceptance
  fixture.
- Prove the route starts from marked bounded `linalg.generic`, not from a
  pre-authored selected exec dispatch fixture.
- Prove the generated bundle preserves:
  - selected RVV shape metadata from the C++ RVV descriptor;
  - selected RVV capability continuity for the finite `i32m2` config;
  - runtime ABI mem-window roles for lhs/rhs/out;
  - runtime ABI param roles for `n` and `rvv_available`;
  - selected dispatch case plus scalar fallback component records;
  - scalar fallback callable boundary and generated dispatch C/header/object
    records.
- Add one focused negative check for a mismatched/stale frontend marker or
  boundary that must fail before a misleading selected artifact/bundle appears.

## Pipeline Boundary For This Round

- Input form: module-level target profile plus a marked `func.func` wrapper
  containing one `linalg.generic` with `tcrv_frontend_lowering = "i32-vsub"`,
  `tcrv_frontend_kernel`, and `tcrv_frontend_target`.
- Pass/tool route: the one-command translation route
  `--tcrv-plan-and-export-target-artifact-bundle`; direct `tcrv-opt |`
  `tcrv-translate` remains valid supporting evidence but is not the primary
  acceptance route for this task.
- Selected representation: `tcrv.exec.variant @rvv_first_slice`,
  `tcrv.exec.variant @scalar_fallback_first_slice`, `tcrv.exec.dispatch`,
  `tcrv.exec.case @rvv_first_slice`, and
  `tcrv.exec.fallback @scalar_fallback_first_slice`.
- Runtime ABI boundary: executable artifact parameters mirror
  `tcrv.exec.mem_window` roles `lhs-input-buffer`, `rhs-input-buffer`,
  `output-buffer`, `tcrv.exec.runtime_param` role `runtime-element-count`,
  and target/export-owned dispatch guard role `dispatch-availability-guard`.
- Selected-shape metadata: bundle/source records must show `i32m2`, SEW 32,
  LMUL `m2`, agnostic tail/mask policy, vector type `vint32m2_t`, vector suffix
  `i32m2`, setvl suffix `e32m2`, and the four finite `rvv.i32_m2.*`
  capability ids.
- Artifact evidence: bundle index plus source/header/object records for
  `tcrv-export-rvv-scalar-i32-vsub-dispatch-{c,header,object}`.

## Acceptance Criteria

- The `i32-vsub` + `i32m2` bundle lit test starts from bounded linalg input and
  proves the one-command front door emits the selected RVV+scalar dispatch
  source/header/object bundle.
- The positive test checks selected RVV shape metadata, finite `i32m2`
  capability continuity, runtime ABI mem-window roles, runtime param roles,
  selected dispatch component records, scalar fallback boundary, and artifact
  records.
- A focused negative lit test proves a mismatched frontend marker/body fails
  during the one-command front door before a bundle index or selected artifact
  can be produced.
- `git diff --check` passes.
- When the build directory exists, build at least `tcrv-opt` and
  `tcrv-translate`.
- Run focused lit tests touched by this task.
- Because this task only strengthens compiler/front-door proof and does not
  change runtime-visible generated C/object/ABI behavior, no new `ssh rvv`
  evidence is required; `5db3128` remains the latest runtime evidence for the
  affected i32m2 vsub dispatch path.

## Out Of Scope

- New families, dtypes, LMULs, RVV ops, scalar ops, generic vector lowering, or
  broad smoke matrices.
- Python compiler semantics or Python-owned lowering/selection/export
  decisions.
- New runtime correctness/performance claims.
- Editing or reopening the archived selected-shape ownership task.
- Committing `artifacts/tmp`, raw remote logs, credentials, or build outputs.

## Technical Notes

- Specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
- Prior task reference read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-selected-shape-descriptor-ownership-cxx-target-backend/prd.md`
- Primary implementation/test surfaces inspected:
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `lib/Transforms/ExecutionPlanningPipeline.cpp`
  - `lib/Transforms/VariantSelection.cpp`
  - `lib/Transforms/VariantDispatchSynthesis.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `tools/tcrv-opt/tcrv-opt.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `test/Transforms/`
  - `test/Target/RVVScalarDispatch/`
  - `test/Target/TargetArtifactBundleExport/`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`

## Current Status

Completed. The full one-command bounded linalg `i32-vsub` + `i32m2`
frontend-to-artifact bundle route already existed, so this task tightened the
focused proof instead of adding duplicate compiler plumbing.

Completed changes:
- Strengthened the positive bundle lit test to prove selected RVV `i32m2`
  shape metadata, selected-plan metadata, runtime ABI parameter roles,
  dispatch mem-window roles, dispatch guard linkage, RVV capability continuity,
  scalar fallback selected boundary, and source/header/object bundle records.
- Added a focused negative one-command front-door lit test proving an
  `i32-vsub` frontend marker with an `arith.addi` body fails during bounded
  linalg lowering before any bundle index or selected RVV+scalar artifact is
  produced.

Validation completed:
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate 05-10-bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'plan-linalg-i32m2-vsub-and-export-target-artifact-bundle|plan-linalg-i32-vsub-marker-mismatch-no-bundle'`
  from `artifacts/tmp/tianchenrv-build/test` (2/2 passed)
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (172/172 passed)

No new `ssh rvv` evidence was produced because this task changed compiler
front-door proof coverage only, not runtime-visible generated source/object/ABI
behavior. Commit `5db3128` remains the latest runtime evidence for the
affected `i32-vsub i32m2` dispatch path.
