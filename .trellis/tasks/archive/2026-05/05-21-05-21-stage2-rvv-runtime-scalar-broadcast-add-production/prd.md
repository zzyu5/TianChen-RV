# Stage2 RVV runtime scalar broadcast add production evidence

## Goal

Execute the Hermes direction brief against current HEAD without creating another metadata-only closeout. The bounded module behavior is:

```text
out_i32[i] = in_i32[i] + scalar_i32
```

for runtime `n`, using e32m1 as an ordinary instance of the corrected generic typed RVV surface.

Current repository inspection shows that the core production path already exists: selected/pre-realized `tcrv_rvv.typed_binary_pre_realized_body` accepts `memory_form = "rhs-scalar-broadcast"` only for `op_kind = "add"` and SEW32/LMUL m1; selected-body realization materializes `setvl/with_vl/load/splat/binary/store`; route planning/provider recognize `scalar_broadcast_add`; target artifact export and `scripts/rvv_generated_bundle_abi_e2e.py` contain a generated-bundle path. This round must therefore avoid duplicating the route and instead close the active acceptance gap: generated-bundle evidence must exercise at least two runtime scalar addend values, not a single hard-coded scalar.

## What I Already Know

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean.
- Current HEAD before this task: `9a4d58c7 trellis: close scalar broadcast stale task`.
- No `.trellis/.current-task` existed before this task was created.
- Archived stale task `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice/` closed as current-head validation and recorded prior `ssh rvv` evidence, but only for `rhs_scalar=-37`.
- Current source already has `rhs-scalar-value` ABI role, generic `tcrv_rvv.splat`, pre-realized `rhs-scalar-broadcast`, route operation `scalar_broadcast_add`, and generated-bundle dry-run support.
- Specs require the authority chain to stay: `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin selected-body realization / legality / route provider -> common EmitC materializer -> target artifact -> `ssh rvv` evidence.

## Requirements

- Do not add a duplicate scalar-broadcast route or revive old `i32m1` route authority.
- Preserve existing production authority placement:
  - `tcrv.exec` declares ABI/runtime roles only.
  - `tcrv_rvv` body structurally carries lhs mem window, RHS scalar runtime parameter, output mem window, runtime `n`/AVL, SEW/LMUL/policy, operation kind, and scalar-broadcast memory form.
  - RVV selected-body realization owns `setvl/with_vl/load/splat/binary/store`.
  - RVV route planning/provider derive C types, header facts, ABI order, operation, and diagnostics from typed body/config/runtime facts.
  - Common EmitC/export remains neutral.
- Extend the generated-bundle ABI harness so `scalar_broadcast_add` can run the same generated artifact over multiple runtime scalar addends.
- Keep default behavior compatible for existing non-scalar-broadcast generated-bundle users.
- Positive evidence must cover runtime counts `7`, `16`, and `23` and at least two scalar values.
- Negative evidence must keep fail-closed coverage for missing scalar runtime role, unsupported scalar-broadcast op kind, invalid memory form, missing/incomplete typed body, invalid policy/config, mixed RHS buffer broadcast with RHS scalar splat, and stale route-id/legacy authority where applicable.

## Acceptance Criteria

- [x] Current positive selected-body/FileCheck coverage still shows `rhs-scalar-broadcast` realization into `setvl/with_vl/load/splat/binary/store`.
- [x] Current route-planning/provider coverage still shows `scalar_broadcast_add` support derived from typed body/config/runtime facts.
- [x] Generated-bundle script accepts repeated runtime scalar values for `scalar_broadcast_add` and records them in evidence.
- [x] Generated-bundle dry-run passes for counts `7`, `16`, `23` with at least two scalar values.
- [x] Real `ssh rvv` execution passes for counts `7`, `16`, `23` with at least two scalar values, proving runtime scalar use, runtime `n`, and tail sentinel preservation.
- [x] Negative fail-closed tests remain present for scalar role/op-kind/memory-form/mixed-broadcast/incomplete structure/legacy-authority cases.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/source-export, route-id authority, artifact-name authority, or common/export RVV semantic authority.
- [x] Focused script/lit/build checks pass, followed by `check-tianchenrv`, `git diff --check`, and clean final `git status --short`.

## Non-Goals

- No broad broadcast framework.
- No dtype or LMUL clone batch.
- No masked/strided broadcast, compare/select expansion, contraction, matmul, Linalg, or frontend lowering.
- No source-front-door positive route.
- No one-intrinsic wrapper dialect.
- No dashboard/report-only/helper-only work as the claimed compiler-path behavior.
- No new descriptor-driven computation or direct-C exporter path.
- No revival of legacy i32 route authority. The i32 scalar add case is allowed only as an ordinary bounded instance of the generic typed RVV surface.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
- Relevant archived task context read:
  - `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice/prd.md`
  - `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-vector-select-executable-slice/prd.md`
- Current implementation surfaces inspected:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - scalar-broadcast FileCheck tests under `test/Target/RVV`, `test/Conversion/EmitC`, `test/Transforms/LoweringBoundary`, and `test/Scripts`

## Validation Performed

- `pwd`
- `git status --short`
- `git log --oneline -8`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-05-21-stage2-rvv-runtime-scalar-broadcast-add-production`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-broadcast-multiscalar-dryrun --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Manual FileCheck equivalents for the scalar-broadcast script test:
  - ROOT evidence check
  - SCALAR evidence check with `--implicit-check-not="descriptor"`, `--implicit-check-not="direct-C"`, `--implicit-check-not="source-export"`, and `--implicit-check-not="tcrv_rvv.i32_"`
  - HARNESS check with `--implicit-check-not="descriptor"`, `--implicit-check-not="direct-C"`, and `--implicit-check-not="source-export"`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=REALIZED`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=PLAN`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=HEADER`
- `build/bin/tcrv-opt test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir --tcrv-materialize-emitc-lowerable-routes | /usr/lib/llvm-20/bin/FileCheck test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir`
- `build/bin/tcrv-opt test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries`
- `build/bin/tcrv-opt test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir --split-input-file --tcrv-materialize-emitc-lowerable-routes 2>&1 | /usr/lib/llvm-20/bin/FileCheck test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir`
- `build/bin/tcrv-opt test/Dialect/RVV/generic-stage2-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-stage2-dataflow.mlir`
- Real RVV evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-broadcast-multiscalar-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
- `git diff --check`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `cmake --build build --target check-tianchenrv -j2`

`llvm-lit` was not available as `/usr/lib/llvm-20/bin/llvm-lit`, `llvm-lit`, or `lit` in this environment, so the scalar-broadcast script lit RUN lines were executed manually with their FileCheck commands. `check-tianchenrv` still discovered and passed 248 tests.

Fresh `ssh rvv` output:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-multiscalar-ssh
[scalar_broadcast_add] scalar_broadcast_add case n=7 ok rhs_scalar=-37
scalar_broadcast_add case n=16 ok rhs_scalar=-37
scalar_broadcast_add case n=23 ok rhs_scalar=-37
scalar_broadcast_add case n=7 ok rhs_scalar=91
scalar_broadcast_add case n=16 ok rhs_scalar=91
scalar_broadcast_add case n=23 ok rhs_scalar=91
tcrv_rvv_generated_bundle_abi_scalar_broadcast_add_ok counts=7,16,23 rhs_scalars=-37,91
PASS op=scalar_broadcast_add counts=7,16,23 rhs_scalars=-37,91
```

Evidence artifacts:

- Dry-run: `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-multiscalar-dryrun`
- Real RVV: `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-multiscalar-ssh`

Self-repair performed:

- Initial manual HARNESS FileCheck failed because the new check order expected the generated ABI call after the outer scalar loop. The generated call is correctly inside `run_case`, before `main`. The test expectations were reordered and the HARNESS FileCheck passed.

## Final Status

Completed. The existing production scalar-broadcast route remains the route authority; this round added multi-scalar generated-bundle evidence support and refreshed dry-run, FileCheck, fail-closed, real `ssh rvv`, authority scan, and full `check-tianchenrv` validation.

## Spec Update Judgment

No `.trellis/spec/**` update is needed. The long-term contract already requires runtime scalar values to be explicit typed body/runtime ABI facts and requires `ssh rvv` evidence for runtime correctness claims. This round only strengthened a generated-bundle evidence harness and its test coverage.
