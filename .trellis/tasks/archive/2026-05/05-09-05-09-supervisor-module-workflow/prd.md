# Optimize Hermes/Codex Module-Sized Trellis Workflow

## Goal

Refactor the serial Hermes/Codex supervision workflow so Hermes acts as the
planner/reviewer for module-sized engineering tasks and Codex executes the
current Trellis task with a short base prompt. This task only changes workflow
prompts, runner prompt text, and necessary Trellis workflow/spec guidance. It
must not implement TianChen-RV compiler functionality.

## Requirements

- Shorten `scripts/codex_serial_supervisor_prompt.md` into an execution-focused
  Codex base prompt.
- Keep the Codex base prompt focused on repository root, single-worker serial
  execution, repo/task/spec reading, Trellis one-round workflow, technology
  stack, short red lines, validation, task update, finish/archive, commit, and
  final report.
- Move complex direction selection to the Hermes review prompt.
- Update the Hermes review prompt in `scripts/codex_serial_supervisor.py` so
  Hermes chooses exactly one module-sized task brief instead of asking Codex to
  pick among candidates.
- Require Hermes next prompts to look like current task briefs: owner, why now,
  specs/code to read, functional module to finish, explicit non-goals, minimal
  validation, and continuation behavior if unfinished.
- Add anti-stall review logic: when several rounds do not move an end-to-end
  compiler path closer to `TianChen-RV MLIR -> selected boundary ->
  plugin-owned lowering/emission -> artifact/runtime evidence`, Hermes must
  stop micro-tuning and choose a larger module owner.
- Preserve runner evidence packaging, strict JSON output, repo_audit /
  review_input priority, manual steering handling, and existing CLI surfaces.
- Update durable Trellis workflow or supervision guidance only where needed to
  record the module-sized Trellis loop behavior.

## Trellis Round Contract

The Codex base prompt must describe this one-round flow:

1. brainstorm / research: understand the current task, read relevant specs and
   code, and add context when needed;
2. PRD: if the current task has no clear PRD, create or repair it before coding;
3. implementation: implement a coherent module from the PRD, not a single
   isolated helper;
4. check / self-repair: run relevant checks, repair failures, rerun;
5. minimal validation: only validate the changed module behavior;
6. task status update: keep task context/status truthful;
7. finish/archive: finish/archive when complete according to repository Trellis
   conventions and record journal;
8. commit: create one coherent commit, or clearly leave the task open with the
   next continuation point.

## Out Of Scope

- No TianChen-RV compiler feature implementation.
- No new RVV, Scalar, IME, Offload, dialect, pass, lowering, or target exporter
  functionality.
- No broad test matrix or hardware experiment.
- No current-state or maturity report written into durable specs.
- No changes that remove useful runner evidence packaging or break existing CLI
  options.

## Acceptance Criteria

- [x] Codex base prompt is clearly shorter and execution-focused.
- [x] Codex base prompt includes the Trellis one-round workflow.
- [x] Hermes review prompt selects one module-sized owner and does not delegate
      task choice to Codex.
- [x] Hermes review prompt includes anti-stall checks and larger-owner recovery.
- [x] Hermes next prompt requirements cover unfinished task continuation.
- [x] Trellis workflow/supervision guidance records the new workflow behavior.
- [x] `python3 -m py_compile scripts/codex_serial_supervisor.py` passes.
- [x] Prompt/no-exec render checks pass and show repo_audit/review_input are
      still packaged.
- [x] One coherent commit records the workflow update.

## Technical Notes

- Relevant files:
  - `scripts/codex_serial_supervisor_prompt.md`
  - `scripts/codex_serial_supervisor.py`
  - `.trellis/spec/implementation-stack/supervision-loop.md`
  - `.trellis/workflow.md`
- The previous Scalar worker committed before this task began; this task should
  not touch compiler implementation files.
