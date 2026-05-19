# Stage2 RVV compare/select executable ABI closure

## Goal

Close the executable ABI evidence gap for the already route-supported
pre-realized RVV compare/select selected-body path. The task starts from the
typed `tcrv_rvv.typed_compare_select_pre_realized_body` path added in the
previous round and proves that the generated bundle, external C ABI harness,
runtime `n`/AVL handling, expected-result checking, and real `ssh rvv`
execution all consume the same explicit compare/select facts.

This is an evidence and ABI closure task for one existing production path. It
is not a new Stage2 coverage class.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `09ca76d0 rvv: realize pre-realized compare select`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-20-stage2-rvv-cmp-select-executable-abi`.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-compare-select-route/`
  completed route-supported compare/select realization, positive/negative lit
  coverage, generated-bundle dry-run, and `check-tianchenrv 179/179`, but made
  no runtime correctness claim and did not run `ssh rvv`.
- `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require the authority chain:
  selected `tcrv.exec` RVV variant -> typed/realized `tcrv_rvv` body -> RVV
  plugin legality / selected-body realization / provider-built
  `TCRVEmitCLowerableRoute` -> neutral common EmitC/export -> target artifact
  -> real `ssh rvv` evidence for runtime/correctness claims.
- The current `scripts/rvv_generated_bundle_abi_e2e.py` already has a
  pre-realized `cmp_select` expectation and dry-run lit coverage for counts
  `7,16,23`.
- The current `cmp_select` harness expression is
  `(lhs[index] == rhs[index] ? lhs[index] : rhs[index])`, matching the explicit
  `eq` predicate and `select-lhs-when-mask-else-rhs` layout. Because true lanes
  satisfy `lhs == rhs`, output values cannot distinguish the selected branch on
  those lanes; the harness should therefore ensure the input set contains both
  true and false predicate lanes where feasible and keep expected-result
  checking tied to the explicit predicate/select facts.

## Requirements

1. Keep the compiler authority unchanged: compare/select semantics stay in the
   typed `tcrv_rvv` body and RVV plugin/provider path. Do not move semantics
   into common EmitC/export, descriptors, route ids, artifact names, or source
   metadata.
2. If executable closure needs code changes, keep them in evidence tooling,
   harness generation, or missing ABI support for the generated bundle path.
3. Ensure the `cmp_select` generated harness derives inputs, expected values,
   runtime counts, and result checks from the same explicit operation,
   predicate, mask, select-layout, memory-form, and runtime ABI facts used by
   the typed body path.
4. For representative counts `7`, `16`, and `23`, produce real `ssh rvv`
   compile/link/run evidence for the pre-realized compare/select bundle.
5. The expected-result checker must demonstrate both predicate-true and
   predicate-false lanes when feasible. For the current `eq(lhs, rhs)` plus
   select `lhs` else `rhs` semantics, true-lane output equality is inherently
   branch-indistinguishable, so the harness should explicitly count/require
   both predicate outcomes for multi-lane cases.
6. Missing ABI, harness, or target-artifact support must fail closed with
   targeted diagnostics.
7. Preserve the existing generated-bundle dry-run evidence for
   `--pre-realized-selected-body --op-kind cmp_select`.
8. Confirm the task did not reintroduce active legacy authority:
   `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_` route authority,
   `!tcrv_rvv.i32m*` lowerable authority, source-front-door/source-seed
   authority, descriptors/direct-C/source-export authority, or common/export
   RVV semantic authority.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata match this bounded
      compare/select executable ABI closure task.
- [x] Generated-bundle dry-run for pre-realized `cmp_select` still passes for
      counts `7`, `16`, and `23`.
- [x] Real `ssh rvv` generated-bundle ABI evidence passes for pre-realized
      `cmp_select` at counts `7`, `16`, and `23`.
- [x] The harness/evidence shows both predicate-true and predicate-false lanes
      for multi-lane `cmp_select` counts, or records a truthful infeasibility
      reason if the selected semantics cannot make branch values differ.
- [x] Expected-result checking is derived from explicit compare/select facts,
      not from route ids, artifact names, descriptor residue, C strings, test
      names, or common EmitC/export guesses.
- [x] Any touched script/lit/compiler behavior has focused validation.
- [x] Active-authority scan shows no reintroduced legacy/source/descriptor/common
      semantic authority.
- [x] `git diff --check` passes. Worktree cleanliness is checked after
      archive/commit.

## Non-Goals

- No new predicate kinds, select layouts, dtype/LMUL expansion, conversion,
  broadcast, reduction, macc, matmul, or other Stage2 coverage side quests.
- No high-level frontend lowering or source-front-door positive route.
- No performance claim.
- No descriptor-driven computation, descriptor-driven C export, direct-C source
  export, or compatibility wrapper preserving legacy i32 authority.
- No broad smoke matrix unless shared behavior changes require it.

## Validation Plan

1. Validate Trellis task context.
2. Run focused script checks:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
3. Run focused pre-realized compare/select dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root /tmp/tianchenrv-pre-realized-cmp-select --run-id dryrun --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`.
4. Run the same generated-bundle ABI flow without `--dry-run` to collect real
   `ssh rvv` evidence.
5. Run the lit dry-run fixture if tools are available.
6. Run focused build/lit checks for touched behavior if compiler or test files
   change; run `check-tianchenrv` only if shared compiler behavior changes.
7. Run active-authority scans over the changed script/test surfaces and active
   RVV include/lib/script/test paths.
8. Run `git diff --check`.

## Implementation Notes

- If the current path already runs on `ssh rvv`, keep code changes minimal and
  record the result as evidence replay for the previous production path.
- If the harness does not exercise both predicate outcomes, adjust input
  generation and checking in the evidence script only; do not reinterpret
  compare/select semantics outside the typed body facts.

## Implementation Results

- Confirmed the existing pre-realized `cmp_select` generated bundle already
  compiles and runs on `ssh rvv`; the active blocker was evidence completeness,
  not missing compiler ABI support.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so `cmp_select` generated
  harness input includes both equality and non-equality predicate lanes:
  indices divisible by 4 make `lhs == rhs`, other indices make `lhs != rhs`.
- Added a `cmp_select`-specific harness checker that counts
  `predicate_true_lanes` and `predicate_false_lanes`, requires both for
  multi-lane cases, and keeps expected output as
  `(lhs[index] == rhs[index] ? lhs[index] : rhs[index])`.
- Recorded a harness-level `predicate_coverage_contract` in evidence JSON and
  updated the focused dry-run lit test to assert the new contract and generated
  harness checks.
- Did not change C++/MLIR dialects, selected-body realization, route planning,
  RVV provider, common EmitC/export, or target artifact exporter code.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-cmp-select-executable-abi`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root /tmp/tianchenrv-pre-realized-cmp-select --run-id dryrun --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] Focused lit:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run'`
      from `build/test`: 1/1 passed.
- [x] Real `ssh rvv` executable evidence:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root /tmp/tianchenrv-pre-realized-cmp-select-ssh --run-id sshrun --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
- [x] `ssh rvv` output:
      `cmp_select case n=7 ok predicate_true_lanes=2 predicate_false_lanes=5`;
      `cmp_select case n=16 ok predicate_true_lanes=4 predicate_false_lanes=12`;
      `cmp_select case n=23 ok predicate_true_lanes=6 predicate_false_lanes=17`;
      `PASS op=cmp_select counts=7,16,23`.
- [x] Active-authority scan over the changed script/test diff found no added
      `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, source-front-door/source-seed authority,
      descriptor/direct-C/source-export authority, or common/export RVV
      semantic authority.
- [x] `git diff --check`
