# RVV pre-realized arithmetic executable ABI gate

## Goal

Prove the just-completed RVV pre-realized add/sub/mul selected-body realization
path at the executable ABI boundary. The evidence path must start from the
pre-realized selected-body fixtures, run through public selected-boundary
materialization, emission planning, target object/header bundle export, an
external C ABI harness, and real `ssh rvv` compile/link/run.

## Current Facts

- Current HEAD at task creation is `faa4543`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.
- The archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-arithmetic-pre-realized-realization-closure`
  closed local route-supported pre-realized add/sub/mul realization and
  generated local bundles, but it intentionally made no runtime or correctness
  claim.
- `scripts/rvv_generated_bundle_abi_e2e.py` already validates generated
  object/header bundles and can run an external C ABI harness on `ssh rvv`.
  Its default evidence input is still the fully realized explicit selected-body
  fixture; `--source-seed` is a legacy explicit-only mode.
- The pre-realized add/sub/mul fixtures exist under `test/Target/RVV/` and
  require `--tcrv-materialize-selected-lowering-boundaries` before emission
  planning/export.

## Requirements

1. Add a bounded evidence-runner mode for the pre-realized selected-body
   fixtures:
   `test/Target/RVV/pre-realized-selected-body-artifact-{add,sub,mul}.mlir`.
2. In that mode, the runner must use the public selected-boundary
   materialization pass before emission planning and bundle export.
3. The generated bundle and harness checks must confirm the selected variant,
   runtime ABI name, function name, ordered `lhs,rhs,out,n` runtime ABI
   parameters, runtime AVL/VL metadata, selected-body operation, and realized
   typed compute op for add/sub/mul.
4. The evidence must be unable to pass through fully realized explicit-body
   fixtures, legacy source seed input, descriptor/direct-C/source-export
   residue, route-id authority, artifact-name authority, or common EmitC RVV
   semantic branching.
5. Common EmitC/materializer and target export must remain neutral; if a local
   exporter issue appears, repair the owning path rather than teaching common
   code RVV semantics.
6. Collect real `ssh rvv` compile/link/run evidence for add/sub/mul with
   bounded nontrivial runtime counts such as `7,16,23`. If remote access or
   toolchain blocks the run, record the exact blocker and do not claim
   executable status.
7. Keep checks focused on the changed evidence runner, pre-realized fixtures,
   RVV plugin target path, target artifact export, and bounded residue scans.

## Acceptance Criteria

- [x] `scripts/rvv_generated_bundle_abi_e2e.py` has an explicit
      pre-realized selected-body input mode and records that mode in evidence.
- [x] The pre-realized mode runs
      `--tcrv-materialize-selected-lowering-boundaries` before
      `--tcrv-materialize-emission-plans` and target bundle export.
- [x] Dry-run evidence for add/sub/mul proves selected variants
      `pre_realized_body_rvv_i32_{add,sub,mul}`, runtime ABI names
      `rvv-i32m1-{add,sub,mul}-callable-c-abi.v1`, function prototypes, ordered
      ABI parameters, runtime AVL/VL metadata, and typed compute ops
      `tcrv_rvv.i32_{add,sub,mul}`.
- [x] Real `ssh rvv` evidence for add/sub/mul passes counts `7,16,23`, or the
      exact blocker is recorded without an executable claim.
- [x] Focused lit/self-test coverage prevents confusing pre-realized evidence
      with fully realized explicit-body input or legacy source-seed input.
- [x] Focused build/test checks for touched RVV plugin/export behavior pass.
- [x] Bounded residue scans over touched files/artifacts show no restored
      descriptor/direct-C/source-export path, no source-front-door default
      authority, no route/artifact metadata authority, and no common EmitC RVV
      semantic branch.

## Non-Goals

- No compare/select, reductions, conversions, broadcast expansion, new dtype,
  new LMUL, new source shapes, new intrinsic families, Linalg/Vector/StableHLO
  frontend work, or one-intrinsic wrapper dialects.
- No common EmitC/materializer/target export RVV semantic branching.
- No source-front-door default authority restoration.
- No dashboard, artifact-index campaign, broad smoke matrix, performance
  claim, or helper-only/report-only completion.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.

## Validation Plan

- Build focused targets:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Run focused lit for the script self-test/dry-run coverage and the
  pre-realized add/sub/mul target fixtures plus existing negative coverage.
- Run local pre-realized dry-run evidence for add/sub/mul with counts
  `7,16,23`.
- Run real `ssh rvv` pre-realized evidence for add/sub/mul with counts
  `7,16,23`.
- Run bounded residue scans over touched runner/test files and relevant RVV
  plugin/common EmitC surfaces.
- Run `python3 ./.trellis/scripts/task.py validate <task-dir>` and
  `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-arithmetic-pre-realized-realization-closure/prd.md`

## Initial Code Surface

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`
- `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir`
- `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`

## Implementation Notes

- Added `--pre-realized-selected-body` to
  `scripts/rvv_generated_bundle_abi_e2e.py`. The mode selects the
  pre-realized add/sub/mul fixtures and is mutually exclusive with
  `--source-seed`.
- Added pre-realized expected facts for selected variants
  `pre_realized_body_rvv_i32_add`, `pre_realized_body_rvv_i32_sub`, and
  `pre_realized_body_rvv_i32_mul`, with matching external C ABI function names.
- In pre-realized mode, local generation now runs
  `tcrv-opt <fixture> --tcrv-materialize-selected-lowering-boundaries
  --tcrv-materialize-emission-plans` before
  `tcrv-translate --tcrv-export-target-artifact-bundle`.
- Added a materialized MLIR check that confirms the selected variant,
  `tcrv_rvv.with_vl`, the realized typed compute op, and absence of
  `tcrv_rvv.i32_binary_pre_realized_body` after public selected-boundary
  materialization.
- Added focused lit coverage in
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-dry-run.test` to
  prove the dry-run evidence uses pre-realized input, not fully realized
  explicit-body input or the legacy source-front-door seed.
- Common EmitC/materializer, RVV provider, and target exporter code were not
  changed.

## Evidence

- Local dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-pre-realized-arithmetic-executable-abi-gate-dry`.
- Real `ssh rvv` artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-pre-realized-arithmetic-executable-abi-gate`.
- Each op evidence records:
  - selected variants
    `pre_realized_body_rvv_i32_add`,
    `pre_realized_body_rvv_i32_sub`, and
    `pre_realized_body_rvv_i32_mul`;
  - typed compute ops
    `tcrv_rvv.i32_add`,
    `tcrv_rvv.i32_sub`, and
    `tcrv_rvv.i32_mul`;
  - `pre_realized_body_consumed = true`;
  - ordered runtime ABI parameters `lhs,rhs,out,n`;
  - runtime AVL metadata `runtime_abi:n`;
  - external C ABI harness consuming only the generated header and object.
- Remote compile facts for add/sub/mul:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote run output:

```text
add case n=7 ok
add case n=16 ok
add case n=23 ok
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23

sub case n=7 ok
sub case n=16 ok
sub case n=23 ok
tcrv_rvv_generated_bundle_abi_sub_ok counts=7,16,23
PASS op=sub counts=7,16,23

mul case n=7 ok
mul case n=16 ok
mul case n=23 ok
tcrv_rvv_generated_bundle_abi_mul_ok counts=7,16,23
PASS op=mul counts=7,16,23
```

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-pre-realized-arithmetic-executable-abi-gate-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-pre-realized-arithmetic-executable-abi-gate --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`
  passed 3/3 after fixing FileCheck ordering in the new test.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/pre-realized-selected-body-artifact-add.mlir ../test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir ../test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir ../test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
  passed 7/7.
- Residue scan over the real pre-realized evidence artifact for
  `legacy-rvv-source-seed`, `explicit_selected_body`,
  `source-artifact-front-door-pipeline`,
  `tcrv-rvv-materialize-i32m1-vector-source-front-door`,
  `rvv_arithmetic_op`, `descriptor`, `direct-C`, `direct_c`,
  `source-export`, and `source_export` found no matches.
- Residue scan over the touched runner/test for
  `source-artifact-front-door-pipeline`, `rvv_arithmetic_op`,
  `RVVI32M1ArithmeticRouteSpec`, `RVVI32M1ArithmeticSlice`, and
  `collectRVVI32M1ArithmeticSlice` found no matches.
- Common EmitC materializer scan for `rvv`, `tcrv_rvv`, `__riscv`, and `i32_`
  found no matches in
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-rvv-pre-realized-arithmetic-executable-abi-gate`
- `git diff --check`

## Status

Completed locally with real `ssh rvv` runtime/correctness evidence for the
pre-realized add/sub/mul generated artifacts. Ready for Trellis archive and one
coherent commit.
