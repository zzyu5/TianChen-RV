# Stage2 RVV executable artifact closure for generic typed elementwise routes

## Goal

Close executable artifact evidence for the already route-supported generic
typed RVV selected-body routes. This task must prove that the existing
generic typed arithmetic baseline, RHS broadcast add, and compare/select
selected bodies flow through RVV plugin-owned route construction, common EmitC
materialization, generated object/header bundles, runtime ABI `n`/AVL binding,
and real `ssh rvv` correctness runs.

This is an executable artifact closure task, not a coverage expansion task.

## Current Facts

- Git was clean before task creation and HEAD was
  `8175bcc rvv: add generic stage2 elementwise predicate routes`.
- No `.trellis/.current-task` existed before this task was created.
- The archived Stage2 route coverage PRD completed generic typed
  `tcrv_rvv.load`, `tcrv_rvv.broadcast_load`, `tcrv_rvv.binary`,
  `tcrv_rvv.compare`, `tcrv_rvv.select`, and `tcrv_rvv.store` route support,
  but intentionally made no `ssh rvv` runtime/correctness claim.
- `scripts/rvv_generated_bundle_abi_e2e.py` is evidence tooling only. It
  starts from explicit selected `tcrv.exec` / `tcrv_rvv` body fixtures,
  materializes emission plans, exports generated target artifact bundles,
  builds an external C ABI harness, and optionally compiles/runs that harness
  on `ssh rvv`.
- Current bounded fixtures already exist for:
  - arithmetic selected-body baseline: `explicit-selected-body-artifact-add`,
    `sub`, and `mul`;
  - RHS broadcast selected-body add:
    `explicit-selected-body-artifact-broadcast-add.mlir`;
  - compare/select selected body:
    `explicit-selected-body-artifact-cmp-select.mlir`.
- Long-term specs require RVV dtype, SEW, LMUL, policy, memory form,
  operation kind, runtime ABI use, and intrinsic mapping to be structural in
  the typed `tcrv_rvv` body/config/runtime facts or derived by the RVV plugin.
  Common EmitC/export must stay neutral.

## Requirements

1. Generate and verify target artifact bundles for exactly the bounded
   existing generic typed routes:
   - arithmetic baseline add/sub/mul;
   - RHS broadcast add;
   - compare/select.
2. Preserve runtime ABI parameter order `lhs, rhs, out, n`, `runtime_abi:n`
   AVL source metadata, and actual runtime `n` use across multi-count harness
   runs.
3. Ensure generated C/RVV artifacts use RVV intrinsic leaves supplied by the
   RVV plugin after generic typed body/config/runtime validation. Do not infer
   route behavior from route ids, artifact names, ABI strings, test names,
   status fields, or emission-plan mirrors.
4. If harness, script, bundle, or target export glue is missing or stale,
   repair it at the runtime/artifact boundary. Do not invent computation
   semantics in common export or restore source-front-door/descriptor-driven
   authority.
5. Unsupported or non-executable cases must fail closed with clear diagnostics
   rather than producing object/header bundles from metadata-only or legacy
   route authority.
6. Record `ssh rvv` compile/run evidence under the existing artifact root for
   the bounded fixtures, with sanitized command/output records and generated
   evidence JSON.
7. Keep active RVV authority on the generic typed selected-body path. Do not
   reintroduce `RVVI32M1*` slices, finite `tcrv_rvv.i32_*` helpers,
   `!tcrv_rvv.i32m*` positive types, source-front-door positive routes,
   descriptor computation, or common/export RVV semantic branches.

## Acceptance Criteria

- Local dry-run artifact generation passes for arithmetic add/sub/mul selected
  bodies and verifies generated bundle metadata, object/header artifacts,
  runtime ABI order, `n`/AVL metadata, and harness generation.
- Local dry-run artifact generation passes for RHS broadcast add and proves
  `tcrv_rvv.broadcast_load` plus `rhs[0]` correctness expectation without
  legacy broadcast helper ops.
- Local dry-run artifact generation passes for compare/select and proves
  `tcrv_rvv.compare` + `tcrv_rvv.select` predicate dataflow without legacy
  select helper ops.
- Real `ssh rvv` compile/link/run passes for the same bounded fixture set:
  arithmetic add/sub/mul, RHS broadcast add, and compare/select, each with
  multiple distinct runtime `n` counts including a non-one-vector count.
- Generated evidence includes remote compile success, remote run success, and
  harness PASS markers for every bounded op in scope.
- Focused local checks for touched script/provider/materializer paths pass. If
  no source code changes are needed, the evidence commands themselves are the
  focused behavioral validation for this closure.
- Active-authority scan over `include lib test scripts .trellis/spec` confirms
  no new legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m*`, source-front-door, descriptor, or common/export RVV
  semantic authority was introduced by this task.
- Trellis task context is valid, task status is truthful, and the task is
  archived only if all bounded executable evidence is complete.

## Non-Goals

- No reductions, conversions, contraction/FMA expansion, new dtype/LMUL clone
  batches, new Stage2 route classes, or one-intrinsic wrapper growth.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive route restoration.
- No Scalar, IME, Offload, TensorExt, Template/Toy, dashboard, tuning DB,
  readiness machine, broad smoke matrix, or performance claim.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python remains evidence
  tooling only.

## Validation Plan

1. Validate task context and build `tcrv-opt` / `tcrv-translate`.
2. Run the script self-test and focused lit tests for selected-body,
   RHS-broadcast, and compare/select dry-run modes.
3. Run explicit local dry-runs with repo-local build tools:
   - selected-body add/sub/mul;
   - RHS broadcast add;
   - compare/select.
4. Run the corresponding non-dry `ssh rvv` evidence commands with
   `--timeout 180 --connect-timeout 10`.
5. Inspect evidence JSON and remote stdout for PASS markers and runtime counts.
6. Run an active-authority scan and `git diff --check`.
7. Run `check-tianchenrv` only if source/script/shared test behavior is
   changed enough to justify the broader suite.

## Initial Code Surface

- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-rhs-broadcast-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`

## Definition Of Done

- The bounded selected-body routes are executable as generated target
  artifacts on `ssh rvv` with correctness evidence.
- Any runtime/artifact glue fixes needed for that closure are implemented and
  focused tests pass.
- No new route coverage or legacy authority is introduced.
- The task context and PRD reflect actual completed behavior.
- The task is finished/archived and one coherent commit is created when
  complete. If an executable blocker remains, the task stays open with the
  exact blocker and next continuation point.

## Implementation Notes

- No compiler, provider, materializer, script, or test source change was
  required after inspection and validation. The existing runtime/artifact
  boundary already carries the bounded generic typed routes to executable
  target artifacts.
- The module behavior completed in this round is executable evidence closure:
  generated object/header bundles for arithmetic add/sub/mul, RHS broadcast
  add, and compare/select were compiled and run on `ssh rvv` with correctness
  harness PASS markers.
- The task did not add route coverage, did not add new `tcrv_rvv` ops, did not
  enable source-front-door positives, and did not change common EmitC/export
  semantics.

## Evidence

### Local Checks

- `pwd`
- `git status --short`
- `git log --oneline -8`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-05-19-stage2-rvv-executable-artifact-closure`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-selected-body-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --rhs-broadcast-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-rhs-broadcast-add-dry --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-cmp-select-dry --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-rhs-broadcast-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test` from `build/test`: 4/4 passed.
- `git diff --check`

### SSH RVV Evidence

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-selected-body-rvv --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
  - artifact root: `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-exec-closure-selected-body-rvv`
  - result: `success`, `ssh_evidence=true`
  - PASS markers: `PASS op=add counts=7,16,23`, `PASS op=sub counts=7,16,23`, `PASS op=mul counts=7,16,23`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --rhs-broadcast-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-rhs-broadcast-add-rvv --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
  - artifact root: `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-exec-closure-rhs-broadcast-add-rvv`
  - result: `success`, `ssh_evidence=true`
  - PASS marker: `PASS op=add counts=7,16,23`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-exec-closure-cmp-select-rvv --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
  - artifact root: `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-exec-closure-cmp-select-rvv`
  - result: `success`, `ssh_evidence=true`
  - PASS marker: `PASS op=cmp_select counts=7,16,23`

### Active-Authority Scan

- `git diff --name-only -- include lib test scripts .trellis/spec` produced no
  paths, so this task introduced no changes to active compiler/provider/test
  authority surfaces.
- Repository-wide legacy-token scan over `include lib test scripts
  .trellis/spec` still reports pre-existing specs, negative/deprecated
  fixtures, mirror route/runtime ABI labels, source-front-door negative or
  deprecated materializers, and RVV intrinsic leaves derived after generic
  provider validation. No new active authority was introduced in this task.

## Completion Status

- Complete for this bounded executable artifact closure.
- No continuation point is open.
