# Redirect: clean stray RVV gearbox wording/spec worktree leak

## Goal

Restore repository and Trellis control-plane coherence after commit `ccd029b2`
by classifying the dirty RVV Gearbox wording/spec/artifact/archive residue and
either removing it or finishing it as one truthful, bounded documentation task.
This task must not open a new RVV compiler feature, route, pass, artifact
evidence, performance claim, or Stage2 coverage owner.

## What I Already Know

- The round began on `main` with HEAD `ccd029b2 rvv: prove product dequant
  clamp executable abi`.
- No `.trellis/.current-task` existed, so this redirect task was created from
  the Hermes brief before source edits.
- The dirty pre-existing paths were:
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`, and
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-gearbox-pass-pipeline-wording/`.
- The archived wording task records a small documentation/spec calibration, not
  a compiler implementation task.
- The added wording mostly reinforces existing authority boundaries: typed
  selected-body facts and RVV provider validation remain authority; artifacts,
  metadata, route ids, q8/q4 labels, benchmark names, and llama.cpp names do
  not.
- A stale fact was found in the wording residue: it names `dec212f4` as the
  product-dequant-clamp evidence commit, but `dec212f4` is not an ancestor of
  current HEAD and the current branch's completed clamp evidence commit is
  `ccd029b2`.

## Classification

- `.trellis/spec/extension-plugins/rvv-plugin.md`: valid to retain if it stays
  as current-implementation calibration and does not claim completed
  autotuning, route support, executable support, or artifact authority.
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`: valid to
  retain if it only clarifies plugin-local pass-pipeline semantics and layered
  tuning modes without claiming all modes exist today.
- `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`: valid to retain
  only as an interpretation artifact with explicit current-state calibration;
  it must not be treated as route, pipeline, performance, or evidence authority.
- `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-gearbox-pass-pipeline-wording/`:
  valid to retain as the archived owner of the wording delta, but its stale
  evidence commit reference must be repaired if the wording is committed.

## Requirements

- Keep this round bounded to repository/Trellis coherence for the named dirty
  paths and the new redirect task files.
- If retaining the wording residue, repair stale commit references from the
  non-current `dec212f4` lineage to current HEAD `ccd029b2` where the text means
  current product-dequant-clamp evidence.
- Preserve explicit mirror/authority language:
  - typed `tcrv_rvv` body and provider facts are the source of route facts;
  - Gearbox/resource candidate facts only matter when consumed by selected-body
    realization, provider planning, or target artifact validation;
  - artifact metadata, names, route ids, q8/q4 labels, benchmark names, and
    llama.cpp names are not authority.
- Do not edit compiler production source, tests, pass registration, route
  providers, artifact generation logic, or runtime evidence scripts unless the
  wording review discovers a direct contradiction introduced by the dirty files.
- Finish and archive this redirect task if the final worktree is committed
  cleanly.

## Acceptance Criteria

- [x] Focused diff review classifies all four dirty path groups as retained or
      removed.
- [x] Any retained wording is truthful against current HEAD and uses
      current-implementation calibration instead of completed-autotuner or
      performance wording.
- [x] The stale `dec212f4` clamp evidence reference is removed or corrected to
      current `ccd029b2` evidence where appropriate.
- [x] No compiler source, tests, pass registration, route provider, artifact
      generation, or runtime evidence code is changed.
- [x] `git diff --check` passes.
- [x] `git diff --cached --check` passes before commit.
- [x] Bounded old-authority scan over touched added lines shows no new positive
      legacy `i32m1`, descriptor, source-front-door, direct-C, common-EmitC RVV
      semantic authority, or artifact/status route authority.
- [x] Final `git status --short` is clean after one coherent commit.

## Definition Of Done

- The worktree is clean and future Direction Briefs start from committed
  evidence only.
- The retained or removed status of each dirty path group is recorded in this
  task's completion notes and the workspace journal.
- One coherent commit records the repaired wording residue, this redirect task,
  and Trellis bookkeeping.

## Out Of Scope

- New Stage2 RVV coverage, MAcc/reduction/dequant/mask expansion, source-front
  positive routes, pass-pipeline redesign, autotuning databases, broad spec
  rewrites, dashboards, report indexes, performance runs, or `ssh rvv` runtime
  claims.
- Treating the Gearbox v3 artifact, archived task status, metadata fields,
  emission-plan fields, or route ids as route, pipeline, executable, progress,
  or evidence authority.

## Technical Notes

- Initial inspection commands:
  `pwd`, `git status --short`, `git log --oneline -8`.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Related archive read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-resource-selected-dequant-clamp-f32-executable-abi/`.
- Dirty wording owner read:
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-gearbox-pass-pipeline-wording/`.

## Completion Notes

- Retained `.trellis/spec/extension-plugins/rvv-plugin.md`: the added text is
  current-implementation calibration for the Gearbox MVP pass and
  low-precision resource-candidate provider contract. It explicitly rejects
  completed-autotuner, performance, llama.cpp parity, and metadata-authority
  readings.
- Retained `.trellis/spec/variant-pipeline/generation-selection-tuning.md`: the
  added text clarifies RVV plugin-local pass-pipeline semantics and layered
  static/offline/JIT tuning modes without claiming all modes exist today.
- Retained `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`: the added
  section marks the artifact as target-direction interpretation and calibrates
  current implementation status; it now cites current branch evidence commits
  `6877677e` and `ccd029b2`.
- Retained `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-gearbox-pass-pipeline-wording/`:
  the archive is a valid owner for the original wording delta; its PRD evidence
  commit reference was repaired from stale `dec212f4` to current `ccd029b2`.
- No compiler source, tests, pass registration, route provider, artifact
  generation, runtime evidence script, or `ssh rvv` claim changed in this
  redirect task.
- Spec update judgment: the retained dirty spec wording is the spec update for
  this coherence task; no additional spec layer or guide update is needed.
- Verification run before archive/commit: `get_context.py --mode packages`,
  `task.py validate`, `git diff --check`, stale commit scan, and bounded added
  line old-authority scan. The old-authority scan only matched explicit
  negative authority wording.
- Post-archive self-repair: validating the archived task initially found
  `implement.jsonl` and `check.jsonl` still pointing at the pre-archive PRD
  path. Both entries were corrected to the archive path and validation was
  rerun successfully before the final amended commit.
- Final clean worktree status is verified after the coherent commit and recorded
  in the final report.
