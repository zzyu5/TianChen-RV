# Supervisor resume from existing Codex review run

## Goal

Allow the serial supervisor to resume from the exact point where a loop paused
after a Codex run: run Hermes official review on that existing run directory,
then continue with Hermes' `next_prompt`. This avoids inserting a human-written
delta or launching a bridge Codex worker before review.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The loop paused after
  `artifacts/tmp/hermes_codex_supervisor/runs/20260513T-extension-family-construction-template-r0017-20260513T170820Z`.
* That run already packaged `review_input.md`, snapshots, stderr/stdout, and
  repo audit evidence, so Hermes can review it directly.
* Current `loop` starts with a worker turn and has no CLI path to begin with
  review of an existing run.
* The user explicitly does not want a hand-written `--initial-delta` bridge.

## Requirements

* Add a runner control-plane option to `loop`/`start` that accepts an existing
  Codex `run_dir` as the first official Hermes review target.
* The resume path must call the same official `hermes_review` machinery as a
  normal post-worker review.
* The resume path must not launch Codex before Hermes review.
* If Hermes returns a valid `next_prompt`, the next Codex round should use that
  prompt as the task brief.
* If Hermes fails, the runner must fail closed as usual.
* Do not implement compiler functionality.

## Acceptance Criteria

* [x] CLI has a documented `--resume-review-run-dir` loop option.
* [x] `start` forwards that option to the detached loop process.
* [x] `loop --resume-review-run-dir <run>` writes a `resume_review_start`
      event and official review artifacts before launching any worker.
* [x] Next worker round begins after the reviewed round index when the run id
      contains `-rNNNN-`.
* [x] No human-written delta is required for this resume path.
* [x] `python3 -m py_compile scripts/codex_serial_supervisor.py` passes.
* [x] A no-exec/lightweight resume render check proves the option parses and
      reaches review setup without starting a worker first.

## Non-Goals

* No compiler code changes.
* No Hermes prompt rewrite.
* No change to one-shot steering semantics.

## Completion Notes

* Added `--resume-review-run-dir` to loop/start CLI.
* The loop now resolves the existing run directory, runs official Hermes review
  on it, writes `resume_review_start` / `hermes_review_finished`, and only then
  launches the next worker from Hermes' `next_prompt`.
* When the reviewed run id contains `-rNNNN-`, the next worker round continues
  after that round index.
* Dry check used `--review-no-llm --max-rounds 17` against r0017 and confirmed
  no worker round launched before review.

## Checks

```bash
python3 -m py_compile scripts/codex_serial_supervisor.py
python3 scripts/codex_serial_supervisor.py loop --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --base-prompt scripts/codex_serial_supervisor_prompt.md --loop-id 20260514T-resume-review-run-drycheck --review-no-llm --max-rounds 17 --resume-review-run-dir artifacts/tmp/hermes_codex_supervisor/runs/20260513T-extension-family-construction-template-r0017-20260513T170820Z
python3 ./.trellis/scripts/task.py validate 05-14-supervisor-resume-from-existing-review-run
git diff --check
```
