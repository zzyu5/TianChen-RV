# Repair Hermes Task Discovery Toward Larger Production Owners

## Goal

Repair the Hermes/Codex supervisor task-discovery mechanism so automatic task
selection prefers large strategic production-capability owners over adjacent
artifact-evidence seams. The immediate failure mode is that Hermes correctly
rejects tiny helpers in prose, but still keeps choosing small Stage 2 RVV
generated-bundle evidence closeouts because each one can locally claim an
executable seam. A merely "compiler spine" owner is still too conservative if
it can be decomposed into one narrow route family or one ABI fixture at a time.
The durable fix must change the canonical supervisor prompt and review
packaging so Hermes can detect recent evidence-only drift, create or continue a
macro-task / campaign owner when needed, and keep that large owner active
across rounds until milestone-level exit gates are met.

## What I Already Know

* The current stopped loop is `20260608T051150Z`.
* The latest completed worker round is r0010 with commit
  `e1de6a7f rvv: record runtime scalar reduce add evidence`.
* STOP was written with reason
  `human_requested_diagnose_hermes_direction_before_continuing`, so the loop
  should not continue automatically while this task-discovery repair is being
  designed.
* Recent Hermes choices followed a locally plausible but globally too-small
  chain:
  source-front-door authority demotion -> segment2 memory evidence ->
  runtime-scalar segment2 evidence/provider checks -> standalone min/max
  evidence -> standalone reduce-add evidence.
* Several of those rounds explicitly said production source did not need
  changes and mainly archived generated-bundle / `ssh rvv` evidence.
* The user does not want only a one-shot next-round steering patch. The request
  is to fix the base prompt / task discovery mechanism so Hermes naturally
  chooses suitable larger tasks.
* The user clarified that "larger" must mean larger than a one-round compiler
  spine step. Hermes may and should choose large multi-round tasks when the
  alternative is a sequence of small seams.
* `scripts/codex_serial_supervisor_prompt.md` is the canonical Codex worker
  base prompt. It constrains worker execution after Hermes has selected a
  Direction Brief.
* Actual task selection is primarily controlled by the Hermes review prompt
  built in `scripts/codex_serial_supervisor.py` around the review-prompt
  generation path, especially the Review Job, Stage Decision Rules, Owner
  Selection Rules, and Required next_prompt Shape sections.
* `.trellis/spec/implementation-stack/supervision-loop.md` already says Hermes
  should prefer larger real compiler owners, must not decompose coherent
  compiler steps into helper/evidence-only rounds, and must use an anti-stall
  rule when recent rounds do not make an end-to-end path closer to completion.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` already says
  Stage 2 must not degrade into unbounded generated-bundle evidence closeouts,
  and should advance typed primitive coverage, selected-body realization,
  resource-aware tuning, or measured performance comparison paths.
* The current Hermes prompt contains anti-small-task wording, but does not
  operationalize it with a recent-round drift summary, hard evidence-only
  quotas, or a Stage 2 priority ladder.

## Assumptions

* The correct repair should be committed source/spec changes, not a transient
  `manual_steering_once.md` only.
* The mechanism should remain serial and Hermes-owned: Hermes still chooses one
  owner, Codex executes the current step, and the runner records evidence.
* A single owner may now be a macro-owner or campaign that intentionally spans
  multiple Codex rounds. "Single owner" does not mean "must be finishable in
  one round."
* The Codex-visible Trellis task must also be a macro-task. Hermes must not
  decompose a campaign into a stream of newly created small Trellis tasks, and
  Codex must not finish/archive the macro-task until the macro-level gates are
  actually satisfied.
* We should not add new JSON output fields unless the runner validation path is
  deliberately updated. The safer MVP is to strengthen prompt content and
  review evidence packaging while preserving the existing strict JSON shape.
* Evidence closeouts are still valid when they are the single blocker after a
  production compiler change or when new runtime/performance behavior is
  claimed. The fix should prevent unbounded evidence chains, not ban evidence.

## Requirements

* Update the Hermes review prompt in `scripts/codex_serial_supervisor.py` so
  task selection has an explicit "macro-owner first" algorithm, not only
  preference prose.
* Add a macro-task mode to the review prompt semantics without changing the
  JSON schema: Hermes must be able to choose a large owner, state the campaign
  milestone, specify the current round's slice, and require Codex to keep the
  Trellis task open when the macro-owner is not complete.
* Add a Codex macro-task continuation contract:
  when `.trellis/.current-task` points at an in-progress macro-task and the
  task PRD has incomplete campaign-level acceptance criteria, Codex must
  continue that task rather than create a new small task. It may update the PRD
  with the current milestone, execute one slice, and leave a truthful
  continuation point, but it must not archive the macro-task after a partial
  milestone.
* Treat "bounded enough for one round" as a rule for the current slice, not the
  whole owner. The owner itself may span multiple related compiler,
  realization, route, artifact, benchmark, and validation surfaces.
* Define a size floor for post-drift Hermes owners:
  - not a single op variant;
  - not one generated-bundle ABI fixture;
  - not one fail-closed case;
  - not one provider test;
  - not one adjacent route-family seam;
  - must instead advance a named production capability family, pass pipeline,
    primitive surface, or performance-comparison workflow with explicit
    milestone gates.
* Add a recent-round drift summary to Hermes review input or prompt context.
  It should summarize at least the last several rounds in the active loop:
  round id, commit title/hash, changed-file categories, whether production
  compiler source changed, whether the round was metadata/task archive only,
  whether the review reason called it metadata-only/evidence closeout, and the
  selected next-owner title/reason.
* Add a hard evidence-closeout rule:
  generated-bundle / `ssh rvv` evidence may be selected only when it validates
  a production path changed in the same or immediately previous round, or is a
  named blocker for a larger owner. After one evidence closeout for a module,
  the next owner must normally upgrade to production compiler surface,
  selected-body realization, resource-aware planning, typed primitive
  coverage, or measured performance path.
* Add a recent-drift rule:
  if the last N rounds, initially three, are metadata-only, archive/journal
  heavy, no-production-source-change, or repeated evidence closeouts, Hermes
  must choose a macro production-capability owner or set `continue=false` for
  human steering. It must not select another adjacent route-family evidence
  seam by default.
* Add a campaign-continuation rule:
  when a macro-owner is active, Hermes must not switch to a neighboring seam
  after each substep. It must continue the same macro-owner until an explicit
  milestone exit gate is satisfied, the PRD is proven stale, or human steering
  redirects it.
* Add a review-default rule:
  if the last worker advanced a macro-task but did not complete the macro-level
  acceptance gates, Hermes must emit a continuation Direction Brief for the
  same Trellis task and the next unfinished milestone. It must not reinterpret
  the just-finished slice as a complete module and select a new adjacent
  artifact/evidence seam.
* Add a Stage 2 priority ladder after Stage 1 is clean:
  1. RVV production-kernel capability campaign: selected-body realization /
     Gearbox resource-aware pass pipeline plus the low-precision contraction
     primitive and measurement surfaces needed to approach production RVV
     kernels.
  2. RVV plugin-local selected-body realization / Gearbox resource-aware pass
     structure when performance-sensitive or low-precision work is the blocker.
  3. Low-precision / quantized contraction primitive surface foundation:
     typed i8/u8 vector/config, i8/u8 load facts, i8*i8 widening product,
     widening reduction / `vwredsum`-style provider route facts, and
     fail-closed provider/target validation.
  4. Typed primitive coverage gaps that unblock structured-kernel classes.
  5. Generated-bundle / `ssh rvv` evidence only when it is the single blocker
     for a newly changed production path.
* Update `scripts/codex_serial_supervisor_prompt.md` so the Codex worker, when
  receiving a Hermes Direction Brief, repairs too-small PRDs into a macro-task
  with milestone gates rather than completing a tiny evidence-only task by
  default. Codex must not finish/archive the macro-task after one slice unless
  the macro-level acceptance criteria are actually met.
* Update worker finish/archive rules so a worker can end a round with a clean
  commit and an active macro-task when the macro-task is incomplete. That final
  report must include completed milestone(s), remaining milestones, next exact
  continuation point, and checks run. A clean commit is allowed without task
  archive when the macro-task is intentionally continuing.
* Update `.trellis/spec/implementation-stack/supervision-loop.md` and/or
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md` only if the
  durable rule needs to be clarified beyond what is already written.
* Add focused tests for the prompt/review-packaging behavior. These can be
  script-level tests or unit-style checks around the prompt builder, but must
  verify that recent evidence-only drift appears in Hermes context and that the
  prompt contains the hard escalation rule.
* Preserve existing strict Hermes JSON output compatibility unless the runner
  is intentionally updated and tested.

## Next Macro-Owner Direction After Mechanism Repair

After this task-discovery repair is committed, the next supervisor review
should not continue with another standalone reduction, segment2, generated
bundle, or artifact ABI evidence seam. The next Direction Brief should select a
macro-owner:

```text
RVV production-kernel capability campaign:
Gearbox resource-aware selected-body realization + low-precision contraction
primitive surface + measured same-target comparison path
```

This is intentionally larger than a one-round compiler-spine task. Hermes
should require Codex to create or repair one macro PRD and keep it active across
rounds. The campaign should have milestone gates such as:

1. resource-aware selected-body realization / Gearbox pass structure:
   build/prune/select/realize phases, resource facts, and provider-consumed
   plan or realized `tcrv_rvv` structure;
2. low-precision contraction primitive surface:
   typed i8/u8 vector/config, i8/u8 loads, i8*i8 widening product, widening
   reduction / `vwredsum`-style route facts, and fail-closed validation;
3. generated artifact and runtime correctness evidence for the selected
   primitive path when executable behavior is claimed;
4. measured same-target comparison harness or evidence path for production-like
   RVV kernels, with llama.cpp-style q8/q4 examples treated as pressure tests,
   not route authority.

The first round under this macro-owner may implement one slice, but the task
must not be considered complete merely because that slice has an archived
evidence bundle. Completion requires the campaign-level gates to be met or a
truthful continuation point to remain active.

The expected loop shape is:

```text
Hermes review:
  sees macro-task active and campaign gates incomplete
    -> continue same macro-task
    -> name next unfinished milestone

Codex worker:
  reads same macro-task PRD
    -> implements one coherent milestone slice
    -> updates PRD/checklist/journal with partial progress
    -> commits the slice if coherent
    -> leaves .trellis/.current-task active unless campaign gates are complete

Next Hermes review:
  sees macro-task still active
    -> continues same campaign
```

This is the opposite of the current bad pattern:

```text
Hermes selects small seam
  -> Codex creates small task
  -> Codex archives it after evidence
  -> Hermes selects neighboring small seam
```

If Hermes proves that Gearbox realization is blocked by missing primitive
surface, that is not a reason to choose another evidence seam. It should remain
inside the same macro-owner and execute the primitive-surface milestone first:

```text
Stage 2 RVV low-precision / quantized contraction primitive surface foundation
```

That fallback should target typed i8/u8 vector/config, i8/u8 loads, i8*i8
widening product, and widening reduction/provider validation as a coherent
submodule. It must not become a q8-named route, llama.cpp-named artifact, or
hand-written wrapper.

## Acceptance Criteria

* [x] PRD, `implement.jsonl`, and `check.jsonl` capture the mechanism repair
      scope and relevant specs.
* [x] Current supervisor status is verified before changing loop control.
* [x] Hermes review prompt generation has a durable anti-micro-task algorithm:
      recent-drift rule, evidence-closeout limit, macro-owner mode, campaign
      continuation rule, and Stage 2 priority ladder.
* [x] Hermes review input or prompt includes recent-round drift evidence so
      Hermes can apply the rule without broad repo archaeology.
* [x] Codex worker base prompt tells workers to repair too-small evidence-only
      PRDs into macro-tasks with milestone gates when the Direction Brief is
      undersized.
* [x] Codex worker base prompt tells workers not to finish/archive a macro-task
      after one bounded slice unless macro-level acceptance gates are actually
      met.
* [x] Codex worker base prompt permits a clean committed worker round that
      leaves the current macro-task active with a precise continuation point.
* [x] Hermes review prompt defaults to continuing an active incomplete
      macro-task instead of selecting a new task.
* [x] Review evidence packaging exposes current macro-task status, completed
      milestone checklist, and remaining milestone checklist to Hermes.
* [x] Existing strict JSON review output remains compatible, or any schema
      change is explicitly tested.
* [x] Focused tests validate prompt/rendered context behavior.
* [x] A one-shot next-round steering file is prepared only after the durable
      prompt/mechanism repair, pointing Hermes at the RVV production-kernel
      capability campaign, with Gearbox selected-body realization and
      low-precision contraction primitive surface as campaign milestones.
* [x] The stopped supervisor can be resumed with official Hermes review of the
      latest completed run, not by feeding Codex a stale `round_XXXX_next_prompt`.
* [x] `git diff --check`, relevant script tests, and any affected supervisor
      tests pass.
* [x] Final report states files changed, exact rule added, next-round steering
      content/path, restart/resume command, and final loop/git state.

## Out Of Scope

* Do not implement the Gearbox pass in this task. This task fixes selection
  mechanism and prepares the macro-owner.
* Do not implement low-precision contraction primitives in this task.
* Do not restart the loop into automatic execution until the durable prompt
  repair and one-shot steering are ready.
* Do not edit raw loop artifacts except for official control files such as
  `manual_steering_once.md` when intentionally resuming.
* Do not broaden Hermes into a multi-agent selector or external dashboard.
* Do not force every Hermes owner to be finishable in one round. This task is
  specifically about allowing and preferring macro-owners when repeated small
  tasks are causing drift.
* Do not force every Codex worker round to finish/archive a Trellis task. For
  macro-tasks, a correct worker outcome can be "milestone slice committed,
  macro-task still active, next continuation point recorded."
* Do not remove the requirement for real `ssh rvv` evidence when runtime,
  correctness, or performance is claimed.

## Technical Notes

Inspected files and evidence:

* `scripts/codex_serial_supervisor.py`
* `scripts/codex_serial_supervisor_prompt.md`
* `.trellis/spec/implementation-stack/supervision-loop.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `artifacts/tmp/hermes_codex_supervisor/loops/20260608T051150Z/events.jsonl`
* `artifacts/tmp/hermes_codex_supervisor/loops/20260608T051150Z/round_0009_next_prompt.md`
* `artifacts/tmp/hermes_codex_supervisor/runs/20260608T051150Z-r0010-20260608T082357Z/`
* Memory notes on q8/q4 being examples of broader Stage 2 low-precision and
  selected-body realization maturity gaps.

Important implementation areas:

* Hermes review prompt builder in `scripts/codex_serial_supervisor.py` around
  `build_hermes_review_prompt`.
* Review input / live summary packaging around `# Hermes Supervisor Review
  Input`.
* Worker base prompt in `scripts/codex_serial_supervisor_prompt.md`.
* Supervision-loop spec if prompt behavior needs a durable project contract.
