# Stage2 RVV selected-body artifact ABI closeout

## Goal

Close out the existing uncommitted Stage 2 RVV selected-body-to-artifact
runtime ABI boundary diff from the previous run. The goal is repository
coherence: review the dirty diff, preserve or repair the completed
`widening_macc_add` selected-body-to-generated-bundle evidence path, rerun the
small focused checks needed for confidence, finish/archive this repair task,
create one coherent commit, and leave `git status --short` clean.

This task must not start a new RVV feature owner. It is a bounded closeout for
the previous completed production path:

```text
selected tcrv.exec RVV variant
  -> RVV selected-body realization
  -> realized typed tcrv_rvv body
  -> RVV provider-owned route facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV target artifact bundle
  -> external runtime ABI harness
```

## What I Already Know

* No `.trellis/.current-task` existed at session start, while the working tree
  contained the previous run's uncommitted closeout diff.
* The dirty files are `scripts/rvv_generated_bundle_abi_e2e.py`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/workspace/codex/index.md`,
  `.trellis/workspace/codex/journal-20.md`, and the untracked archived task
  `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/`.
* The previous archived task records successful dry-run, direct-pre-realized
  fail-closed, real `ssh rvv` runtime ABI correctness evidence for counts
  `0,1,16,17,257`, focused binary tests, `git diff --check`, and
  `check-tianchenrv` 465/465.
* The previous run intentionally did not commit because its finish workflow
  barred that worker from running `git commit`; this brief explicitly asks for
  one coherent commit.
* `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  require RVV selected-body realization and provider route facts to remain
  plugin-owned, with common EmitC/export acting as neutral consumers.
* `.trellis/spec/lowering-runtime/emitc-route.md` forbids common EmitC from
  inferring RVV dtype/config/operation/runtime ABI semantics from metadata.
* `.trellis/spec/testing/mlir-testing-contract.md` now needs to describe the
  current selected-boundary positive path and retired direct route-entry
  negative mode coherently.

## Requirements

* Review the uncommitted diff and classify whether each changed file belongs to
  this closeout.
* Preserve the positive selected-boundary generated-bundle path for
  `widening_macc_add`; do not silently drop prior functional evidence.
* Preserve the direct pre-realized route-entry fail-closed guard for
  `widening_macc_add`.
* Repair only real inconsistencies needed for repository coherence.
* Keep common EmitC/export neutrality and RVV plugin authority layering intact.
* Refresh task/workspace metadata truthfully for this closeout round.
* Finish/archive the closeout task, create one coherent commit, and verify the
  final worktree is clean.

## Acceptance Criteria

* [x] Current uncommitted diff is reviewed against the brief and relevant specs.
* [x] The previous archived runtime ABI task validates or any invalid part is
  explicitly repaired with evidence.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes.
* [x] Focused generated-bundle dry-run for `widening_macc_add` passes, or a
  precise preservation decision explains why rerunning is unnecessary.
* [x] Direct pre-realized route-entry negative check for `widening_macc_add`
  remains fail-closed.
* [x] Real `ssh rvv` evidence is rerun if practical; otherwise the previous
  final evidence path is checked and preserved without claiming new runtime
  evidence.
* [x] The smallest affected compiled tests pass, or an exact blocker is
  recorded.
* [x] `git diff --check` passes.
* [x] A bounded old-authority scan over touched files shows no new positive
  route authority from `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, source-front-door/source-artifact markers, descriptors, or
  metadata/status fields.
* [x] Trellis validation passes for the closeout task.
* [x] The task is finished and archived truthfully.
* [x] One coherent commit is created and `git status --short` is clean.

## Definition of Done

* The repository records the completed Stage 2 selected-body-to-artifact runtime
  ABI boundary as a coherent committed state.
* Any checks not rerun are named with the preservation reason and the exact
  prior evidence path.
* No new compiler owner, feature family, matrix expansion, source-front-door
  route, descriptor-driven path, or common EmitC semantic inference is added.

## Technical Approach

1. Read the dirty diff, archived task, relevant specs, and previous
   `last_message.md`.
2. Repair only closeout inconsistencies, if any.
3. Rerun focused script/build/test checks and the bounded authority scan.
4. Update Trellis task status and workspace journal for this closeout.
5. Archive the task, commit the coherent diff, then verify clean status.

## Completion Notes

* The diff review found the previous script/spec/workspace changes aligned
  with the brief and relevant specs. The only closeout repair was removing
  stale `_example` template rows from the archived runtime ABI task's
  `implement.jsonl` and `check.jsonl`.
* Positive dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-dry-run/20260601T134535Z`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/direct-fail/20260601T134543Z`;
  the command exited 1 with the expected retired shortcut diagnostic for
  `widening_macc_add`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-ssh-rvv/20260601T134557Z`;
  counts `0,1,16,17,257` passed signed widening product accumulation and tail
  preservation checks.
* The bounded old-authority scan found only the testing spec's negative
  absence requirement and PRD/status guardrail hits; no new positive route
  authority was added.
* The task was archived with `--no-commit` so the final commit can include the
  previous runtime ABI boundary, this closeout task, workspace notes, and
  metadata repair as one coherent repository state. The final report records
  the resulting commit hash and clean `git status --short`.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-selected-body-artifact-abi-closeout`
* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-dry-run`
* [x] Direct route-entry negative command exited 1 with the expected retired direct route-entry diagnostic.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-ssh-rvv`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] Bounded old-authority scan over touched tracked diff and task directories.
* [x] `rtk git diff --check`

## Decision (ADR-lite)

**Context**: The previous task appears functionally complete and archived but
left the repository dirty because it did not create a commit.

**Decision**: Treat this as a closeout/repair task rather than a new RVV
feature owner. Prefer preserving prior `ssh rvv` runtime evidence and rerunning
focused local checks unless evidence review exposes a real semantic gap.

**Consequences**: The commit should be small and provenance-heavy. It may
include the previous archived task, spec/script regression, workspace journal,
and this closeout task/archive, but must not expand RVV coverage or rewrite the
artifact/runtime path beyond necessary coherence repairs.

## Out of Scope

* New RVV feature families or coverage expansion.
* Broad test matrices or performance claims.
* High-level frontend, source-front-door, descriptor-driven, direct-C, or
  metadata-authority routes.
* Dtype/LMUL clone batches, dashboards, tuning databases, or readiness state
  machines.
* Rewriting common EmitC/export to infer RVV semantics.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/prd.md`.
* Read: `artifacts/tmp/hermes_codex_supervisor/runs/20260601T050142Z-r0021-20260601T132403Z/last_message.md`.
