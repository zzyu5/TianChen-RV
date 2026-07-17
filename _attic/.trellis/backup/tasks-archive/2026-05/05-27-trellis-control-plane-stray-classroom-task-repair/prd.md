# control-plane: repair stray classroom task residue

## Goal

Restore a clean and truthful Trellis/git control-plane state for the serial
Codex loop before selecting any new RVV compiler owner. This task is bounded to
proving and removing the stray untracked classroom task directory:

```text
.trellis/tasks/05-27-classroom-bitwise-xor-slice-workflow/
```

No compiler, RVV route, generated artifact, experiment, or runtime behavior is
part of this cleanup.

## Direction Source

- Direction title: `Redirect: Trellis/worktree control-plane repair for stray
  classroom task`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: untracked
  `.trellis/tasks/05-27-classroom-bitwise-xor-slice-workflow/`.
- Initial HEAD: `b8cc337d rvv: add runtime avl vl control owner`.

## What I Already Know

- The stray classroom task is `status: in_progress`, `creator: codex`,
  `assignee: codex`, priority `P2`, title `classroom: add bitwise xor slice
  workflow`.
- Its PRD targets a classroom worktree and branch
  `classroom/rvv-slices-2026`, not the current main serial RVV loop.
- Its `implement.jsonl` and `check.jsonl` still contain only placeholder
  template records.
- `.trellis/.current-task` is absent at the start of this repair.
- `.trellis/workspace/codex/index.md` and the tail of
  `.trellis/workspace/codex/journal-16.md` record the previous Stage2 RVV
  runtime AVL/VL task as complete.
- The only external reference found for the classroom task is in the archived
  previous task PRD, where it is documented as an initial stray state left
  intact rather than as active work.

## Requirements

- Determine whether the classroom task is live human-owned work or stale Codex
  control-plane residue.
- If evidence shows stale residue, remove only the stray untracked classroom
  task directory.
- Preserve the completed archived RVV task and do not edit compiler/test/source
  files.
- Keep `.trellis/.current-task` clear or pointing only to this repair task
  while the repair is active.
- Finish and archive this repair task after validation so the repository has no
  active Trellis task.
- Leave `git status --short` clean after the cleanup commit.

## Acceptance Criteria

- [x] Evidence records the stray classroom task title, status, creator, assignee,
  and mismatch with the current main RVV loop.
- [x] Evidence records whether `.trellis/.current-task`, workspace index,
  workspace journal, or archived task state references the classroom task.
- [x] If the classroom task is stale residue, the untracked classroom task
  directory is removed.
- [x] No RVV production/compiler behavior, generated artifact, or experiment
  file is changed.
- [x] Trellis task context validates for this repair task.
- [x] `git diff --check` passes.
- [x] `git status --short` is clean after the repair commit.

## Repair Result

- Classification: stale Codex control-plane residue, not human-owned active
  work.
- Reason: the task title/PRD target a classroom worktree and
  `classroom/rvv-slices-2026`; the context JSONL files are placeholders; no
  `.trellis/.current-task` exists at repair start; workspace index/journal show
  the previous RVV owner completed; the only external reference is archived
  prior-task documentation that explicitly left the classroom task intact.
- Action: removed only
  `.trellis/tasks/05-27-classroom-bitwise-xor-slice-workflow/`.
- Validation so far: `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-27-trellis-control-plane-stray-classroom-task-repair`
  passed; `git diff --check` passed; `git status --short -- ':!.trellis/tasks/**'`
  showed no non-task changes.

## Out of Scope

- Implementing classroom bitwise XOR.
- Starting any new RVV route-family owner.
- Editing compiler implementation, tests, generated artifacts, or experiment
  data.
- Running broad test matrices such as `check-tianchenrv`.
- Updating project specs unless a durable Trellis control-plane rule is strictly
  required.

## Technical Notes

- Read `.trellis/spec/index.md`; no compiler-layer spec is modified or needed
  because this task is control-plane hygiene only.
- Read `.trellis/tasks/05-27-classroom-bitwise-xor-slice-workflow/task.json`.
- Read the classroom task PRD and context files to confirm it is unrelated and
  still only template-context seeded.
- Read `.trellis/workspace/codex/index.md` and the tail of
  `.trellis/workspace/codex/journal-16.md`.
- Read archived previous task metadata at
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-vl-control-runtime-avl-boundary-owner/`.
